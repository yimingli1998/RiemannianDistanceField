"""Riemannian metrics of Section 2.3.

Every metric is a callable ``G(q) -> (..., d, d)`` (torch, differentiable).
"""
import torch


def kinetic_energy(robot):
    """G_ke(q) = M(q)  (Eq. 11)."""
    return robot.inertia


def jacobi(robot, H, form="paper"):
    """Jacobi metric for a conservative system at total energy H.

    form="paper": G = 2 (H - P(q)) M(q)        (Eq. 13 / 61) -- the Jacobi metric.  Its
                  geodesics are the free (torque-free) motions at energy H and the
                  length of a path is Maupertuis' action  int p^T dq.  Used throughout.
    form="time" : G = M(q) / (2 (H - P(q)))    -- a different metric: the length of a
                  path under it is the travel time at energy H (Eq. 66), and its
                  geodesics are minimum-time paths, *not* Jacobi geodesics.  Kept only
                  for comparison.
    """
    if form not in ("paper", "time"):
        raise ValueError(form)
    if H <= robot.P_max:
        raise ValueError("H must be larger than max P(q) so that the metric stays positive definite")

    def G(q):
        M = robot.inertia(q)
        k = (2.0 * (H - robot.potential(q)))[..., None, None]
        return k * M if form == "paper" else M / k

    return G


def jacobi_speed(robot, H, form="paper"):
    """Factor turning a unit-G-speed direction into the physical velocity at energy H.

    A motion at energy H has 1/2 qd^T M qd = H - P(q).  For G = 2(H-P)M a unit-G-speed
    vector v has 1/2 v^T M v = 1/(4(H-P)), so qd = 2(H - P(q)) v.  (For the "time"
    form unit G-speed already is the physical speed and the factor is 1.)
    """

    def k(q):
        if form == "time":
            return torch.ones(q.shape[:-1], dtype=q.dtype)
        return 2.0 * (H - robot.potential(q))

    return k


def jacobi_conditioned(robot, form="paper"):
    """Jacobi metric with the energy H passed explicitly: G(q, H) (for C-NES, Sec. 4.3.2)."""

    def G(q, H):
        M = robot.inertia(q)
        k = (2.0 * (H - robot.potential(q)))[..., None, None]
        return k * M if form == "paper" else M / k

    return G


def pullback(robot, G_x=None, reg=1e-3):
    """G_pb(q) = J^T G_x J + reg*I  (Eqs. 14-15). G_x defaults to identity."""

    def G(q):
        J = robot.jacobian(q)
        Gx = torch.eye(J.shape[-2], dtype=q.dtype) if G_x is None else G_x
        return J.transpose(-1, -2) @ Gx @ J + reg * torch.eye(J.shape[-1], dtype=q.dtype)

    return G


def stability_aware(robot, eps=2.0):
    """G_sta(q) = M(q) (1 + eps * P_tilde(q))  (Eq. 17 / 31)."""

    def G(q):
        P_t = (robot.potential(q) - robot.P_min) / (robot.P_max - robot.P_min)
        return robot.inertia(q) * (1.0 + eps * P_t)[..., None, None]

    return G


def euclidean(dim=2):
    def G(q):
        return torch.eye(dim, dtype=q.dtype).expand(q.shape[:-1] + (dim, dim))

    return G


def obstacle_metric(sdf, alpha=5.0, tau=0.5, phi_min=0.1, literal=False, eps=1e-6):
    """Distance-aware metric of Eq. (16) for a 2-D signed distance function ``sdf``.

    phi(x) = sigmoid(alpha (tau - d)) (1 - phi_min) + phi_min   in [phi_min, 1]
    r1 = grad d / |grad d| (obstacle normal), r2 = r1 rotated by 90 deg.

    ``literal=True`` : G = phi r1 r1^T + r2 r2^T                    (Eq. 16 as printed)
    default         : G = (phi / phi_min) r1 r1^T + r2 r2^T

    Read literally, Eq. (16) makes motion along the normal *cheaper* far from
    obstacles (phi -> phi_min) and isotropic close to them, so geodesics hug the
    obstacles.  The behaviour described in the text -- motion towards obstacles is
    penalised close to them, the metric is isotropic far away, and geodesics bend
    around obstacles with clearance -- is obtained by measuring the normal cost
    relative to phi_min, which is the default here (see notebook 6).
    """

    grad_sdf = torch.func.vmap(torch.func.grad(lambda p: sdf(p[None])[0]))

    def G(x):
        d = sdf(x)
        n = grad_sdf(x.reshape(-1, x.shape[-1])).reshape(x.shape)   # composable with vmap/jacrev
        nn = n.norm(dim=-1, keepdim=True)
        r1 = torch.where(nn > eps, n / nn.clamp_min(eps), torch.tensor([1.0, 0.0], dtype=x.dtype))
        r2 = torch.stack([-r1[..., 1], r1[..., 0]], -1)
        phi = torch.sigmoid(alpha * (tau - d)) * (1 - phi_min) + phi_min
        w = phi if literal else phi / phi_min
        return (w[..., None, None] * r1[..., :, None] * r1[..., None, :]
                + r2[..., :, None] * r2[..., None, :])

    return G


def christoffel(G, q):
    """Christoffel symbols of the second kind Gamma^k_ij(q), shape (..., k, i, j).

    Uses autograd on the metric, so any differentiable metric works.
    """
    d = q.shape[-1]
    qf = q.reshape(-1, d)
    dG = torch.func.vmap(torch.func.jacrev(lambda x: G(x[None])[0]))(qf)  # (N, d, d, d): dG[n,a,b,c] = d G_ab / d q_c
    Ginv = torch.linalg.inv(G(qf))
    # Gamma^k_ij = 1/2 G^{kl} (d_i G_lj + d_j G_li - d_l G_ij)
    A = torch.einsum("nlji->nlij", dG)      # d_i G_lj
    B = dG                                  # d_j G_li
    C = torch.einsum("nijl->nlij", dG)      # d_l G_ij
    Gamma = 0.5 * torch.einsum("nkl,nlij->nkij", Ginv, A + B - C)
    return Gamma.reshape(q.shape[:-1] + (d, d, d))


def metric_length(G, path):
    """Riemannian length of a discrete path (N, d) (midpoint rule, Eq. 3)."""
    path = torch.as_tensor(path).double()
    dq = path[1:] - path[:-1]
    mid = 0.5 * (path[1:] + path[:-1])
    Gm = G(mid)
    return torch.sqrt(torch.einsum("ni,nij,nj->n", dq, Gm, dq).clamp_min(0)).sum()
