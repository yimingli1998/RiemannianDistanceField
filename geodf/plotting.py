"""Small plotting helpers shared by the tutorial scripts."""
import numpy as np
import torch
import matplotlib.pyplot as plt

PI = np.pi


def setup_ax(ax, bounds=((-PI, PI), (-PI, PI)), labels=("$q_1$", "$q_2$"), title=None, equal=True):
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_xlabel(labels[0], fontsize=13)
    ax.set_ylabel(labels[1], fontsize=13)
    if equal:
        ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=14)


def metric_ellipses(ax, G, bounds=((-PI, PI), (-PI, PI)), density=13, scale=None, color="gray", **kw):
    """Draw the unit balls of G^-1 (the 'isocontours of inverse matrices' of Fig. 3a)."""
    xs = np.linspace(bounds[0][0], bounds[0][1], density + 2)[1:-1]
    ys = np.linspace(bounds[1][0], bounds[1][1], density + 2)[1:-1]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    q = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float64)
    Ginv = torch.linalg.inv(G(q)).detach().numpy()
    w, V = np.linalg.eigh(Ginv)
    R = V * np.sqrt(np.clip(w, 0, None))[:, None, :]
    cell = min(xs[1] - xs[0], ys[1] - ys[0])
    if scale is None:  # typical ellipse fits in a cell; outliers (near-singular G) are clipped
        scale = 0.45 * cell / np.sqrt(np.percentile(w.max(-1), 90))
    t = np.linspace(0, 2 * np.pi, 40)
    circ = np.stack([np.cos(t), np.sin(t)])
    for c, r in zip(q.numpy(), R):
        e = scale * r @ circ
        rad = np.linalg.norm(e, axis=0).max()
        e = e * min(1.0, 0.5 * cell / rad) + c[:, None]
        ax.plot(e[0], e[1], color=color, lw=1, **kw)
        ax.plot(*c, ".", color=color, ms=2)


def field(ax, X, Y, U, levels=20, cmap="coolwarm", lines=True):
    cf = ax.contourf(X, Y, U, levels=levels, cmap=cmap)
    if lines:
        ax.contour(X, Y, U, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
    return cf


def quiver(ax, X, Y, V, sub=10, color="k", normalize=True, **kw):
    U_, V_ = V[..., 0], V[..., 1]
    if normalize:
        n = np.sqrt(U_ ** 2 + V_ ** 2) + 1e-12
        U_, V_ = U_ / n, V_ / n
    s = slice(sub // 2, None, sub)
    return ax.quiver(X[s, s], Y[s, s], U_[s, s], V_[s, s], color=color, **kw)


def draw_arm(ax, robot, q, color="k", alpha=1.0, lw=2, ee=True):
    P = robot.joints(torch.as_tensor(np.atleast_2d(q), dtype=torch.float64)).numpy()
    for p in P:
        ax.plot(p[:, 0], p[:, 1], "-", color=color, alpha=alpha, lw=lw, solid_capstyle="round")
        ax.plot(p[:2, 0], p[:2, 1], "o", color=color, alpha=alpha, ms=3)
        if ee:
            ax.plot(p[2, 0], p[2, 1], "o", color=color, alpha=alpha, ms=4, mfc="w")


def arm_motion(ax, robot, path, color, n=8, lim=4.3):
    idx = np.linspace(0, len(path) - 1, n).astype(int)
    for k, i in enumerate(idx):
        draw_arm(ax, robot, path[i], color=color, alpha=0.15 + 0.85 * k / (n - 1))
    ee = robot.fk(torch.as_tensor(path, dtype=torch.float64)).numpy()
    ax.plot(ee[:, 0], ee[:, 1], "--", color=color, lw=1.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.plot(0, 0, "ks", ms=6)


def draw_cartpole(ax, x, theta, l=1.0, color="k", alpha=1.0):
    ax.add_patch(plt.Rectangle((x - 0.2, -0.1), 0.4, 0.2, fc=color, alpha=alpha * 0.6, ec=color))
    tip = (x + l * np.sin(theta), l * np.cos(theta))
    ax.plot([x, tip[0]], [0, tip[1]], "-", color=color, alpha=alpha, lw=2)
    ax.plot(*tip, "o", color=color, alpha=alpha, ms=5)


def savefig(fig, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("saved", path)


def plot_wrapped(ax, path, *args, **kw):
    """Plot a joint-space path on the torus: wrap to [-pi, pi) and break the line at the seams."""
    p = (np.asarray(path) + PI) % (2 * PI) - PI
    jump = np.any(np.abs(np.diff(p, axis=0)) > PI, axis=1)
    segs = np.split(p, np.where(jump)[0] + 1)
    lines = []
    for k, s in enumerate(segs):
        lines += ax.plot(s[:, 0], s[:, 1], *args, **(kw if k == 0 else {**kw, "label": None}))
    return lines
