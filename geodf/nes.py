"""Neural Riemannian Eikonal Solver (NES) and its conditioned variants (Section 4.2-4.3).

Three models share one MLP backbone:

* :class:`NES`            U(qs, qe)    = |qe - qs| * softplus( (u(qs,qe) + u(qe,qs)) / 2 )      Eqs. (20)-(21)
* :class:`NES` (cond_dim>0) U(qs, qe | w) -- metric-conditioned C-NES, w appended to the input   Eq. (28)
* :class:`TaskNES`        U(xs, qe)    = |f(qe) - xs| * softplus( u(xs, qe) )                     Eq. (25)

Training minimises the eikonal residual (Eq. 22) and, optionally, the
Laplace-Beltrami regulariser (Eq. 23).
"""
import time

import torch
import torch.nn as nn

from .metrics import christoffel


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128, 128), act=nn.Tanh):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            lin = nn.Linear(d, h)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, act()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NES(nn.Module):
    """Symmetric, factorised two-point distance U(qs, qe [| w])."""

    def __init__(self, lo, hi, hidden=(128, 128, 128), cond_lo=None, cond_hi=None, metric=None):
        """``metric``: optional G(q).  If given, the factor of Eq. (21) is the metric-aware
        distance d_G = sqrt(dq^T G((qs+qe)/2) dq) instead of |qe - qs|.  It is still
        symmetric and vanishes only at qs = qe, but is exact to first order near the
        source, which matters for strongly anisotropic metrics (e.g. the pull-back J^T J,
        where the Euclidean factor leads NES to a wrong, non-viscosity solution)."""
        super().__init__()
        self.metric = metric
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))
        self.dim = self.lo.numel()
        self.cond_dim = 0 if cond_lo is None else len(cond_lo)
        if self.cond_dim:
            self.register_buffer("clo", torch.as_tensor(cond_lo, dtype=torch.float32))
            self.register_buffer("chi", torch.as_tensor(cond_hi, dtype=torch.float32))
        self.mlp = MLP(2 * self.dim + self.cond_dim, hidden)

    def _norm(self, q):
        return 2 * (q - self.lo) / (self.hi - self.lo) - 1

    def raw(self, qs, qe, w=None):
        x = [self._norm(qs), self._norm(qe)]
        if self.cond_dim:
            x.append(2 * (w - self.clo) / (self.chi - self.clo) - 1)
        return self.mlp(torch.cat(x, -1))[..., 0]

    def scale(self, qs, qe, w=None):
        """softplus(u_sym): the 'local speed' factor multiplying the Euclidean distance."""
        return nn.functional.softplus(0.5 * (self.raw(qs, qe, w) + self.raw(qe, qs, w)))

    def factor(self, qs, qe):
        d = qe - qs
        if self.metric is None:
            return torch.linalg.norm(d, dim=-1)
        Gm = self.metric(0.5 * (qs + qe))
        return torch.sqrt(torch.einsum("ni,nij,nj->n", d, Gm, d).clamp_min(1e-12))

    def forward(self, qs, qe, w=None):
        return self.factor(qs, qe) * self.scale(qs, qe, w)


class TaskNES(nn.Module):
    """C-NES conditioned on a task-space boundary: U(xs, qe) = |f(qe) - xs| softplus(u(xs, qe)).

    With ``periodic=True`` the joint angles enter the MLP as (sin q, cos q).  The
    Euclidean factor |f(q) - xs| is already 2*pi-periodic for revolute joints, so the
    whole field is then defined on the torus and backtracking may pass through
    q = +-pi, like the true distance to the IK set.
    """

    def __init__(self, fk, x_lo, x_hi, q_lo, q_hi, hidden=(128, 128, 128), periodic=True):
        super().__init__()
        self.fk = fk
        self.periodic = periodic
        self.register_buffer("xlo", torch.as_tensor(list(x_lo), dtype=torch.float32))
        self.register_buffer("xhi", torch.as_tensor(list(x_hi), dtype=torch.float32))
        self.register_buffer("qlo", torch.as_tensor(list(q_lo), dtype=torch.float32))
        self.register_buffer("qhi", torch.as_tensor(list(q_hi), dtype=torch.float32))
        nq = len(q_lo)
        self.mlp = MLP(len(x_lo) + (2 * nq if periodic else nq), hidden)

    def forward(self, xs, qe):
        x = 2 * (xs - self.xlo) / (self.xhi - self.xlo) - 1
        q = torch.cat([torch.sin(qe), torch.cos(qe)], -1) if self.periodic \
            else 2 * (qe - self.qlo) / (self.qhi - self.qlo) - 1
        u = self.mlp(torch.cat([x, q], -1))[..., 0]
        return torch.linalg.norm(self.fk(qe) - xs, dim=-1) * nn.functional.softplus(u)


# ----------------------------------------------------------------------------- losses
def grad_U(model, src, qe, w=None, create_graph=False):
    """Returns U and dU/dqe (qe must not require grad beforehand)."""
    qe = qe.detach().requires_grad_(True)
    U = model(src, qe) if w is None else model(src, qe, w)
    g = torch.autograd.grad(U.sum(), qe, create_graph=create_graph)[0]
    return U, g, qe


def eikonal_loss(model, src, qe, Ginv, w=None, lb_weight=0.0, G=None):
    """Eq. (22) (+ lb_weight * Eq. (23)).  Ginv = G(qe)^-1."""
    need_hess = lb_weight > 0
    U, g, qe_ = grad_U(model, src, qe, w, create_graph=True)
    norm = torch.sqrt(torch.einsum("ni,nij,nj->n", g, Ginv, g).clamp_min(1e-12))
    loss = ((norm - 1.0) ** 2).mean()
    if need_hess:
        H = torch.stack([torch.autograd.grad(g[:, i].sum(), qe_, create_graph=True)[0]
                         for i in range(g.shape[-1])], -2)
        Gamma = christoffel(G, qe.detach())
        lb = torch.einsum("nij,nij->n", Ginv, H - torch.einsum("nkij,nk->nij", Gamma, g))
        loss = loss + lb_weight * (lb ** 2).mean()
    return loss


def train(model, sample_batch, metric_inv, steps=15000, lr=3e-3, lb_weight=0.0, G=None,
          log_every=500, lr_decay=True, verbose=True):
    """Generic trainer.

    sample_batch() -> (src, qe, w)  (w may be None)
    metric_inv(qe, w) -> G(qe)^-1
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if lr_decay else None
    history, t0 = [], time.time()
    for it in range(steps):
        src, qe, w = sample_batch()
        loss = eikonal_loss(model, src, qe, metric_inv(qe, w), w, lb_weight, G)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        history.append(loss.item())
        if verbose and (it % log_every == 0 or it == steps - 1):
            print(f"step {it:6d}  loss {loss.item():.2e}  ({time.time() - t0:.0f}s)")
    return history


# ----------------------------------------------------------------------------- geodesic flow
def geodesic_flow(model, src, q, G, w=None, normalize=True):
    """V(q) = G^-1 dU/dq (Eq. 8), optionally rescaled to unit G-speed.  Returns (U, V)."""
    with torch.enable_grad():
        U, g, _ = grad_U(model, src, q, w)
    Gq = G(q) if w is None else G(q, w)
    V = torch.linalg.solve(Gq, g[..., None])[..., 0]
    if normalize:
        V = V / torch.sqrt(torch.einsum("ni,nij,nj->n", V, Gq, V).clamp_min(1e-12))[:, None]
    return U.detach(), V.detach()


def backtrack(model, src, q0, G, w=None, step=0.01, max_steps=3000, tol=0.05, goal_fn=None, adaptive=True):
    """Follow -V from q0 towards the source (Eq. 10), batched.

    Each trajectory is frozen once it is within ``tol`` of its goal; ``goal_fn(q)``
    returns that distance (default |q - src|, and the path is then closed with
    the source point).  For C-NES pass e.g. ``goal_fn=lambda q: |fk(q) - xs|``.
    ``step`` is a Riemannian arc length (the flow has unit G-speed).  With
    ``adaptive=True`` each step is capped at U(q)/2 -- U decreases at unit rate along
    the flow, so this never overshoots the source / boundary set.
    Returns a list of (T_i, d) numpy arrays.
    """
    append_source = goal_fn is None
    goal_fn = goal_fn or (lambda q: torch.linalg.norm(q - src, dim=-1))
    q = q0.clone()
    done = torch.zeros(q.shape[0], dtype=torch.bool)
    paths, lengths = [q.clone()], torch.zeros(q.shape[0], dtype=torch.long)
    for k in range(max_steps):
        done |= goal_fn(q) < tol
        if done.all():
            break
        U, V = geodesic_flow(model, src, q, G, w)
        h = torch.clamp(0.5 * U, min=0.05 * step, max=step)[:, None] if adaptive else step
        q = torch.where(done[:, None], q, q - h * V)
        lengths += (~done).long()
        paths.append(q.clone())
    P = torch.stack(paths, 1)
    out = []
    for i in range(q.shape[0]):
        p = P[i, : lengths[i] + 1]
        if append_source:
            p = torch.cat([p, src[i: i + 1]], 0)   # close the last gap linearly
        out.append(p.numpy())
    return out


# ----------------------------------------------------------------------------- helpers
def uniform_pair_sampler(lo, hi, batch, cond_sampler=None):
    """Fresh uniformly distributed (source, goal) pairs in the box [lo, hi] at every step.

    For 2-D problems uniform sampling covers the domain well; RM-MALA (``geodf.sampling``,
    Appendix D) matters for high-dimensional configuration spaces.
    """
    lo = torch.as_tensor(lo, dtype=torch.float32)
    hi = torch.as_tensor(hi, dtype=torch.float32)

    def sample():
        a = lo + (hi - lo) * torch.rand(batch, lo.numel())
        b = lo + (hi - lo) * torch.rand(batch, lo.numel())
        return a, b, None if cond_sampler is None else cond_sampler(batch)

    return sample


def pair_sampler(pool, batch, cond_sampler=None):
    """Random (source, goal) pairs from a pool of configurations (e.g. RM-MALA samples)."""
    pool = torch.as_tensor(pool, dtype=torch.float32)

    def sample():
        i = torch.randint(len(pool), (batch,))
        j = torch.randint(len(pool), (batch,))
        w = None if cond_sampler is None else cond_sampler(batch)
        return pool[i], pool[j], w

    return sample


def train_or_load(model, path, train_fn, retrain=False):
    """Load ``path`` if it exists (and not ``retrain``), otherwise call ``train_fn()`` and save.

    Returns the training history (list of losses, possibly empty) and the wall time.
    """
    import os
    if os.path.exists(path) and not retrain:
        ck = torch.load(path)
        model.load_state_dict(ck["state"])
        print(f"loaded {path} (trained in {ck['time']:.0f}s)")
        return ck["history"], ck["time"]
    t0 = time.time()
    history = train_fn()
    dt = time.time() - t0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state": model.state_dict(), "history": history, "time": dt}, path)
    return history, dt
