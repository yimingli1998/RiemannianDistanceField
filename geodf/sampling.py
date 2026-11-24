"""Geometry-aware sampling with RM-MALA (Appendix D, Algorithm 2).

Target density w.r.t. the Lebesgue measure: rho(q) ∝ sqrt(|G(q)|)   (Eq. 80 with p = const),
i.e. points are uniform w.r.t. the Riemannian volume.
"""
import torch
from torch.func import jacrev, vmap


def wrap(q, periodic, lo, hi):
    """Map periodic coordinates back to [lo, hi) (atan2 trick of Algorithm 2)."""
    out = q.clone()
    for i, per in enumerate(periodic):
        if per:
            span = hi[i] - lo[i]
            out[:, i] = torch.remainder(q[:, i] - lo[i], span) + lo[i]
    return out


def rm_mala(G, lo, hi, n_chains=1000, n_burn=200, n_samples=100, eps=0.3,
            periodic=None, log_target=None, generator=None):
    """Riemannian-manifold MALA.

    G          : metric (d,) -> (d,d), batched over the first dim.
    lo, hi     : box bounds (sequences of length d).
    periodic   : per-dimension flags; non-periodic dims are reflected into the box.
    log_target : log rho(q); default 0.5 * log det G(q).
    Returns (n_samples * n_chains, d) samples and the acceptance rate.
    """
    lo = torch.as_tensor(lo, dtype=torch.get_default_dtype())
    hi = torch.as_tensor(hi, dtype=torch.get_default_dtype())
    d = lo.numel()
    periodic = [True] * d if periodic is None else periodic
    if log_target is None:
        def log_target(q):
            return 0.5 * torch.logdet(G(q))

    def single_G(x):
        return G(x[None])[0]

    def single_L(x):
        return log_target(x[None])[0]

    dG = vmap(jacrev(single_G))        # (N, d, d, d): dG[n,a,b,j] = d G_ab / d q_j
    dL = vmap(jacrev(single_L))        # (N, d)

    def mean_cov(q):
        Gq = G(q)
        Gi = torch.linalg.inv(Gq)
        dGq = dG(q)
        grad_L = dL(q)
        # Eq. (86)
        t1 = 0.5 * torch.einsum("nij,nj->ni", Gi, grad_L)
        GidG = torch.einsum("nab,nbcj->nacj", Gi, dGq)              # G^-1 dG/dq_j
        t2 = torch.einsum("nacj,nci->naij", GidG, Gi)               # G^-1 dG_j G^-1  (indices a,i,j)
        t2 = torch.einsum("nijj->ni", t2)                           # sum_j (.)_ij
        tr = torch.einsum("naaj->nj", GidG)                         # Tr(G^-1 dG_j)
        t3 = 0.5 * torch.einsum("nij,nj->ni", Gi, tr)
        mu = q + eps ** 2 * (t1 - t2 + t3)
        cov = eps ** 2 * Gi
        return mu, 0.5 * (cov + cov.transpose(-1, -2))

    import itertools
    span = hi - lo

    def images(x):
        """All points that wrapping / reflection maps onto x (one step away from the box)."""
        per_dim = []
        for i, per in enumerate(periodic):
            xi = x[:, i]
            per_dim.append([xi, xi + span[i], xi - span[i]] if per else [xi, 2 * lo[i] - xi, 2 * hi[i] - xi])
        return [torch.stack(c, -1) for c in itertools.product(*per_dim)]

    def log_normal(x, mu, cov):
        """Density of the *wrapped / reflected* Gaussian proposal at x: sum over the pre-images of x.

        Evaluating the plain Gaussian at the wrapped point would break detailed balance for
        proposals that cross the boundary of the box.
        """
        dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
        return torch.logsumexp(torch.stack([dist.log_prob(y) for y in images(x)]), 0)

    q = lo + (hi - lo) * torch.rand(n_chains, d, generator=generator)
    samples, accepted, total = [], 0, 0
    for it in range(n_burn + n_samples):
        mu, cov = mean_cov(q)
        L = torch.linalg.cholesky(cov)
        z = torch.randn(n_chains, d, generator=generator)
        q_new = mu + (L @ z[..., None])[..., 0]
        q_new = wrap(q_new, periodic, lo, hi)
        for i, per in enumerate(periodic):     # reflect non-periodic dims
            if not per:
                x = q_new[:, i]
                x = torch.where(x < lo[i], 2 * lo[i] - x, x)
                q_new[:, i] = torch.where(x > hi[i], 2 * hi[i] - x, x)
        mu_new, cov_new = mean_cov(q_new)
        log_alpha = (log_target(q_new) + log_normal(q, mu_new, cov_new)
                     - log_target(q) - log_normal(q_new, mu, cov))
        acc = torch.rand(n_chains, generator=generator).log() <= log_alpha
        q = torch.where(acc[:, None], q_new, q)
        if it >= n_burn:
            samples.append(q.clone())
            accepted += acc.sum().item()
            total += n_chains
    return torch.cat(samples), accepted / max(total, 1)
