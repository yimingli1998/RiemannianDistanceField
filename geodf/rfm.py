"""A small, dependency-free Riemannian eikonal solver on 2-D grids (RFM baseline, Sec. 4.1).

The reference implementation of Riemannian fast marching is ``agd`` (Mirebeau's
HamiltonFastMarching), whose CPU binary is only distributed through conda.  To keep
the tutorial pip-installable we implement the discrete update of Eq. (18) ourselves:

    U(q) = min_{y on stencil}  U(y) + || q - y ||_{G(q)}

where y ranges over the *segments* between angularly consecutive stencil vectors
(semi-Lagrangian update with linear interpolation of U along each segment, which
has a closed-form minimiser).  Using all primitive vectors of Chebyshev radius
<= 3 (32 directions) makes the scheme accurate for strongly anisotropic metrics.
The update is iterated over the whole grid (vectorised Bellman/Jacobi sweeps)
until it reaches its fixed point -- the same fixed point fast marching visits in a
single ordered pass.  On a 200x200 grid this takes a few seconds.

``tests/`` compares the result with a reference field computed with ``agd``
(``tests/data/agd_fmm_values.npy``).
"""
from dataclasses import dataclass
from math import gcd

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator


def _stencil(radius):
    vecs = [(i, j) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)
            if (i, j) != (0, 0) and gcd(abs(i), abs(j)) == 1]
    vecs.sort(key=lambda v: np.arctan2(v[1], v[0]))
    return vecs


@dataclass
class Grid2D:
    """Cell-centred grid, identical to ``agd``'s ``SetRect`` convention."""
    bounds: tuple  # ((x0, x1), (y0, y1))
    n: tuple       # (nx, ny)

    def __post_init__(self):
        if np.ndim(self.n) == 0:
            self.n = (int(self.n), int(self.n))
        (x0, x1), (y0, y1) = self.bounds
        self.h = np.array([(x1 - x0) / self.n[0], (y1 - y0) / self.n[1]])
        self.x = x0 + (np.arange(self.n[0]) + 0.5) * self.h[0]
        self.y = y0 + (np.arange(self.n[1]) + 0.5) * self.h[1]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def points(self):
        return np.stack([self.X, self.Y], -1)  # (nx, ny, 2)


def _shift(A, di, dj, fill=np.inf, periodic=(False, False)):
    """B[i, j] = A[i + di, j + dj] (out of range -> fill, unless periodic)."""
    out = A
    for axis, d, per in ((0, di, periodic[0]), (1, dj, periodic[1])):
        if d == 0:
            continue
        out = np.roll(out, -d, axis=axis)
        if not per:
            idx = [slice(None)] * A.ndim
            idx[axis] = slice(-d, None) if d > 0 else slice(None, -d)
            out = out.copy()
            out[tuple(idx)] = fill
    return out


class GridField:
    """A distance field sampled on a grid + its geodesic flow and a backtracking routine.

    Used for RFM, and for the GRT baseline (any grid-valued U works).
    """

    def __init__(self, grid, U, G, source, Ginv_grid=None):
        self.grid, self.U, self.G, self.source = grid, U, G, np.asarray(source, float)
        if Ginv_grid is None:
            pts = torch.as_tensor(grid.points.reshape(-1, 2), dtype=torch.float64)
            Ginv_grid = np.linalg.inv(G(pts).detach().numpy()).reshape(grid.n[0], grid.n[1], 2, 2)
        Uf = np.where(np.isfinite(U), U, np.nanmax(np.where(np.isfinite(U), U, np.nan)) * 2)
        gx, gy = np.gradient(Uf, grid.h[0], grid.h[1], edge_order=2)
        self.grad = np.stack([gx, gy], -1)                       # dU/dq  (nx, ny, 2)
        self.flow = np.einsum("xyij,xyj->xyi", Ginv_grid, self.grad)  # V = G^-1 dU (Eq. 8)
        self._U_itp = RegularGridInterpolator((grid.x, grid.y), Uf, bounds_error=False, fill_value=None)
        self._g_itp = RegularGridInterpolator((grid.x, grid.y), self.grad, bounds_error=False, fill_value=None)

    def value(self, q):
        return self._U_itp(np.atleast_2d(q))

    def _direction(self, q):
        """Unit-G-speed descent direction -V/|V|_G at a batch of points (N, 2)."""
        g = self._g_itp(q)
        Gq = self.G(torch.as_tensor(q, dtype=torch.float64)).detach().numpy()
        v = np.linalg.solve(Gq, g[..., None])[..., 0]
        nrm = np.sqrt(np.clip(np.einsum("ni,nij,nj->n", v, Gq, v), 1e-12, None))
        return -v / nrm[:, None]

    def backtrack(self, q0, step=0.01, max_steps=20000, tol=None):
        """Integrate dq/ds = -V/|V|_G from q0 to the source (Eq. 10), RK2, unit G-speed.

        ``q0`` is one start (2,) -> returns one path (T, 2), or a batch (N, 2) -> list of
        paths (all integrated together, vectorised).  Paths end with the source point.
        """
        tol = tol if tol is not None else 2 * max(self.grid.h)
        single = np.ndim(q0) == 1
        q = np.atleast_2d(np.array(q0, float))
        done = np.zeros(len(q), bool)
        hist, n_steps = [q.copy()], np.zeros(len(q), int)
        for _ in range(max_steps):
            done |= np.linalg.norm(q - self.source, axis=1) < tol
            if done.all():
                break
            act = ~done
            qa = q[act]
            k1 = self._direction(qa)
            k2 = self._direction(qa + 0.5 * step * k1)
            q = q.copy()
            q[act] = qa + step * k2
            n_steps += act
            hist.append(q.copy())
        H = np.stack(hist, 1)
        paths = [np.vstack([H[i, : n_steps[i] + 1], self.source]) for i in range(len(q))]
        return paths[0] if single else paths


def solve(G, bounds, n, source, mask=None, radius=3, periodic=(False, False),
          max_iter=5000, tol=1e-9, verbose=False):
    """Solve ||grad U||_{G^-1} = 1, U(source) = 0 on a cell-centred grid.

    G       : metric callable (torch, (...,2) -> (...,2,2)).
    source  : a point (2,) or several seed points (k, 2) (U = distance to the nearest seed).
    mask    : optional bool array (nx, ny); True cells are obstacles (U = inf).
    Returns a :class:`GridField` (values in ``.U[i, j]`` at ``(grid.x[i], grid.y[j])``).
    """
    grid = Grid2D(bounds, n)
    pts = torch.as_tensor(grid.points.reshape(-1, 2), dtype=torch.float64)
    Gg = G(pts).detach().numpy().reshape(grid.n[0], grid.n[1], 2, 2)
    Ginv = np.linalg.inv(Gg)
    source = np.asarray(source, float)

    vecs = _stencil(radius)
    segs = []
    for k in range(len(vecs)):
        e0, e1 = np.array(vecs[k]), np.array(vecs[(k + 1) % len(vecs)])
        a = e0 * grid.h
        b = (e1 - e0) * grid.h
        C = np.einsum("i,xyij,j->xy", a, Gg, a)
        B = np.einsum("i,xyij,j->xy", a, Gg, b)
        A = np.einsum("i,xyij,j->xy", b, Gg, b)
        segs.append((e0, e1, A, B, C, np.sqrt(C), np.sqrt(A + 2 * B + C)))

    U = np.full(grid.n, np.inf)
    frozen = np.zeros(grid.n, bool)
    # initialise cells around the source(s) with the local (constant-metric) distance
    for src in np.atleast_2d(source):
        d = grid.points - src
        Gs = G(torch.as_tensor(src[None], dtype=torch.float64)).detach().numpy()[0]
        local = np.sqrt(np.einsum("xyi,ij,xyj->xy", d, Gs, d))
        near = (np.abs(d[..., 0]) <= 1.5 * grid.h[0]) & (np.abs(d[..., 1]) <= 1.5 * grid.h[1])
        U[near] = np.minimum(U[near], local[near])
        frozen |= near
    if mask is not None:
        U[mask] = np.inf
        frozen |= mask

    for it in range(max_iter):
        with np.errstate(invalid="ignore"):          # inf - inf on not-yet-reached cells is expected
            U, change, newly, done = _sweep(U, segs, frozen, periodic, tol)
        if verbose and it % 50 == 0:
            print(f"iter {it}: max change {change:.2e}, reached {np.isfinite(U).mean():.3f}")
        if done:
            break
    return GridField(grid, U, G, source, Ginv)


def _sweep(U, segs, frozen, periodic, tol):
    """One vectorised Bellman update of every cell (see the module docstring)."""
    best = U.copy()
    for e0, e1, A, B, C, n0, n1 in segs:
        U0 = _shift(U, *e0, periodic=periodic)
        U1 = _shift(U, *e1, periodic=periodic)
        cand = np.minimum(U0 + n0, U1 + n1)
        both = np.isfinite(U0) & np.isfinite(U1)
        delta = np.where(both, U1 - U0, 0.0)
        disc = A - delta ** 2
        ok = both & (disc > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            root = np.sqrt(np.clip(A * C - B ** 2, 0, None) / np.where(ok, disc, 1.0))
            t = (-B - delta * root) / A
            ok &= (t > 0) & (t < 1)
            tt = np.where(ok, t, 0.0)
            dist = np.sqrt(np.clip(C + 2 * B * tt + A * tt ** 2, 0, None))
            f = np.where(ok, U0 + tt * delta + dist, np.inf)
        cand = np.minimum(cand, f)
        best = np.minimum(best, cand)
    best[frozen] = U[frozen]
    change = np.nanmax(np.abs(np.where(np.isfinite(best) & np.isfinite(U), best - U, 0.0)))
    newly = np.isfinite(best) & ~np.isfinite(U)
    return best, change, newly, (change < tol and not newly.any())
