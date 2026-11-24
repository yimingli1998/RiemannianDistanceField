"""Analytic models of the two 2-DoF systems used in the paper.

* :class:`PlanarArm` -- 2-link planar arm, point masses at the link ends
  (Section 5.1, Eq. 29).  Joint angles are *relative*: q2 is measured w.r.t. link 1.
  Gravity acts along -y, so P(q) = g (m1 y1 + m2 y2).
* :class:`CartPole` -- cart + point-mass pendulum (Section 5.3.2).  State is
  (x, theta) with theta = 0 pointing up, positive anti-clockwise.

All functions are batched over the leading dimensions and written in plain torch,
so they are differentiable (needed for Christoffel symbols / autograd losses).
"""
from dataclasses import dataclass

import torch

GRAVITY = 9.81


@dataclass
class PlanarArm:
    l1: float = 2.0
    l2: float = 2.0
    m1: float = 1.0
    m2: float = 1.0
    g: float = GRAVITY
    q_min: float = -torch.pi
    q_max: float = torch.pi

    # ------------------------------------------------------------------ kinematics
    def fk(self, q):
        """End-effector position, shape (..., 2)."""
        q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
        x = self.l1 * torch.cos(q1) + self.l2 * torch.cos(q12)
        y = self.l1 * torch.sin(q1) + self.l2 * torch.sin(q12)
        return torch.stack([x, y], -1)

    def joints(self, q):
        """Base, elbow and end-effector positions, shape (..., 3, 2) (for drawing)."""
        q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
        base = torch.zeros(q.shape[:-1] + (2,), dtype=q.dtype)
        elbow = torch.stack([self.l1 * torch.cos(q1), self.l1 * torch.sin(q1)], -1)
        ee = elbow + torch.stack([self.l2 * torch.cos(q12), self.l2 * torch.sin(q12)], -1)
        return torch.stack([base, elbow, ee], -2)

    def ik(self, x):
        """Both analytic IK solutions for an end-effector position x=(x, y): array (2, 2) (or (0, 2))."""
        import numpy as np
        x, y = float(x[0]), float(x[1])
        c2 = (x * x + y * y - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        if abs(c2) > 1:
            return np.zeros((0, 2))
        sols = []
        for q2 in (np.arccos(c2), -np.arccos(c2)):
            q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
            sols.append([(q1 + np.pi) % (2 * np.pi) - np.pi, q2])
        return np.array(sols)

    def jacobian(self, q):
        """d fk / d q, shape (..., 2, 2)."""
        q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
        s1, c1, s12, c12 = torch.sin(q1), torch.cos(q1), torch.sin(q12), torch.cos(q12)
        J = torch.stack([
            torch.stack([-self.l1 * s1 - self.l2 * s12, -self.l2 * s12], -1),
            torch.stack([self.l1 * c1 + self.l2 * c12, self.l2 * c12], -1),
        ], -2)
        return J

    # ------------------------------------------------------------------ dynamics
    def inertia(self, q):
        """Mass matrix M(q) of Eq. (29), shape (..., 2, 2)."""
        m1, m2, l1, l2 = self.m1, self.m2, self.l1, self.l2
        c2 = torch.cos(q[..., 1])
        a = (m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * c2
        b = m2 * l2 ** 2 + m2 * l1 * l2 * c2
        d = m2 * l2 ** 2 * torch.ones_like(c2)
        return torch.stack([torch.stack([a, b], -1), torch.stack([b, d], -1)], -2)

    def coriolis(self, q, dq):
        """Coriolis/centrifugal vector C(q, dq) dq, shape (..., 2)."""
        h = self.m2 * self.l1 * self.l2 * torch.sin(q[..., 1])
        dq1, dq2 = dq[..., 0], dq[..., 1]
        return torch.stack([-h * (2 * dq1 * dq2 + dq2 ** 2), h * dq1 ** 2], -1)

    def potential(self, q):
        """Potential energy P(q), shape (...)."""
        q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
        y1 = self.l1 * torch.sin(q1)
        y2 = y1 + self.l2 * torch.sin(q12)
        return self.g * (self.m1 * y1 + self.m2 * y2)

    def gravity(self, q):
        """Generalised gravity g(q) = dP/dq, shape (..., 2)."""
        q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
        g2 = self.m2 * self.g * self.l2 * torch.cos(q12)
        g1 = (self.m1 + self.m2) * self.g * self.l1 * torch.cos(q1) + g2
        return torch.stack([g1, g2], -1)

    @property
    def P_max(self):
        return self.g * (self.m1 * self.l1 + self.m2 * (self.l1 + self.l2))

    @property
    def P_min(self):
        return -self.P_max

    def inverse_dynamics(self, q, dq, ddq, gravity=True):
        tau = (self.inertia(q) @ ddq.unsqueeze(-1)).squeeze(-1) + self.coriolis(q, dq)
        return tau + self.gravity(q) if gravity else tau

    def forward_dynamics(self, q, dq, tau, gravity=True):
        rhs = tau - self.coriolis(q, dq)
        if gravity:
            rhs = rhs - self.gravity(q)
        return torch.linalg.solve(self.inertia(q), rhs.unsqueeze(-1)).squeeze(-1)


@dataclass
class CartPole:
    """Cart (mass mc) with a point-mass pendulum (mass mp, length l)."""
    mc: float = 1.0
    mp: float = 1.0
    l: float = 1.0
    g: float = GRAVITY

    def inertia(self, s):
        # pole tip at (x + l sin(theta), l cos(theta))
        # => T = 1/2 (mc+mp) dx^2 + mp l cos(theta) dx dtheta + 1/2 mp l^2 dtheta^2
        theta = s[..., 1]
        a = (self.mc + self.mp) * torch.ones_like(theta)
        b = self.mp * self.l * torch.cos(theta)
        d = self.mp * self.l ** 2 * torch.ones_like(theta)
        return torch.stack([torch.stack([a, b], -1), torch.stack([b, d], -1)], -2)

    def potential(self, s):
        return self.mp * self.g * self.l * torch.cos(s[..., 1])

    @property
    def P_max(self):
        return self.mp * self.g * self.l

    @property
    def P_min(self):
        return -self.P_max
