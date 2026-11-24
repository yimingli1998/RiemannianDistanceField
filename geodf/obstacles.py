"""Obstacles for the eikonal problems of Figs. 10-11.

RFM can treat obstacles as cells with infinite cost (zero speed).  A neural
solver needs a smooth problem, so -- as in NTFields (Ni & Qureshi 2022/2023) --
obstacles enter through a speed field s(q) in [s_min, 1]:

    G_s(q) = G(q) / s(q)^2,   s = s_min + (1 - s_min) * smoothstep(clearance / d_safe)

i.e. motion close to (or inside) obstacles is 1/s_min times more expensive.
Defaults s_min = 0.2, d_safe = 0.5: with a sharper field (s_min = 0.1, d_safe = 0.3)
NES smooths the barrier too much and some paths cut through obstacles.
"""
import torch


def smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x * x * (3 - 2 * x)


def speed(clearance, d_safe=0.5, s_min=0.2):
    return s_min + (1 - s_min) * smoothstep(clearance / d_safe)


def speed_scaled(G, clearance_fn, d_safe=0.5, s_min=0.2):
    """Metric G / s(q)^2 for a clearance function q -> (...)."""

    def Gs(q):
        s = speed(clearance_fn(q), d_safe, s_min)
        return G(q) / (s ** 2)[..., None, None]

    return Gs


def disc_sdf(centers, radii):
    """Signed distance to a union of discs in the plane: x (...,2) -> (...)."""
    C = torch.as_tensor(centers, dtype=torch.get_default_dtype())
    R = torch.as_tensor(radii, dtype=torch.get_default_dtype())

    def sdf(x):
        return (torch.linalg.norm(x[..., None, :] - C.to(x.dtype), dim=-1) - R.to(x.dtype)).min(-1).values

    return sdf


def _seg_point_dist(a, b, c):
    ab = b - a
    t = ((c - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)
    t = t.clamp(0, 1)
    return torch.linalg.norm(a + t[..., None] * ab - c, dim=-1)


def arm_clearance(robot, centers, radii):
    """Signed clearance between the arm links (segments) and discs: q (...,2) -> (...)."""
    def clearance(q):
        J = robot.joints(q)                                   # (..., 3, 2)
        out = None
        for c, r in zip(centers, radii):
            c = torch.as_tensor(c, dtype=q.dtype)
            for k in range(2):
                d = _seg_point_dist(J[..., k, :], J[..., k + 1, :], c) - r
                out = d if out is None else torch.minimum(out, d)
        return out

    return clearance
