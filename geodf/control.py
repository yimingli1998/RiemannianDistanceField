"""Turning geodesic flows into robot motion (Section 6.2) and the evaluation metrics (Sec. 6.3).

* :func:`blended_velocity`  -- Eq. (33)
* :func:`kinematic_qp`      -- Eq. (34); with diagonal Q and box constraints the QP
  is separable, so its exact solution is a clip.
* :func:`kinodynamic_qp`    -- Eq. (37); solved with SLSQP in the 2 accelerations
  (tau = M qdd + C qd + g is substituted into the objective).
* :func:`rollout_velocity` / :func:`rollout_torque` -- closed-loop simulation.
* :func:`path_policy`       -- the same blended rule but following a *given path*
  (e.g. to execute a straight line or a grid-solver path with the same controller).
* :func:`evaluate`          -- geodesic length, L2 torque, active torque (Eq. 40).
"""
import numpy as np
import torch
from scipy.optimize import minimize

from .metrics import metric_length


def blended_velocity(U, V, q, q_goal, beta=5.0):
    """qdot_des = -lambda V + (1 - lambda)(q_goal - q),  lambda = beta U / (1 + beta U)."""
    lam = beta * U / (1 + beta * U)
    return -lam[:, None] * V + (1 - lam)[:, None] * (q_goal - q)


def kinematic_qp(qd_des, q, dt, qd_lim, q_lo, q_hi, qd_prev=None, qdd_lim=np.inf):
    """Eq. (34) with diagonal Q: exact solution = clip to the intersection of the boxes.

    Besides the velocity / position limits of Eq. (34) we also bound the change of
    velocity per step (|qd - qd_prev| <= qdd_lim dt), which keeps the start from
    rest physically plausible.
    """
    lo = np.maximum(-qd_lim, (q_lo - q) / dt)
    hi = np.minimum(qd_lim, (q_hi - q) / dt)
    if qd_prev is not None:
        lo = np.maximum(lo, qd_prev - qdd_lim * dt)
        hi = np.minimum(hi, qd_prev + qdd_lim * dt)
    return np.clip(qd_des, lo, hi)


def kinodynamic_qp(robot, q, qd, qdd_des, dt, tau_lim, qdd_lim, qd_lim, q_lo, q_hi, gravity=True):
    """Eq. (37) for one robot state (numpy in, numpy out). Returns (tau, qdd)."""
    qt, qdt = torch.as_tensor(q[None]), torch.as_tensor(qd[None])
    M = robot.inertia(qt)[0].numpy()
    h = robot.coriolis(qt, qdt)[0].numpy() + (robot.gravity(qt)[0].numpy() if gravity else 0.0)
    tau_des = M @ qdd_des + h
    # bounds on qdd from acceleration / velocity / position limits
    lo = np.maximum.reduce([-qdd_lim * np.ones(2), (-qd_lim - qd) / dt, 2 * (q_lo - q - dt * qd) / dt ** 2])
    hi = np.minimum.reduce([qdd_lim * np.ones(2), (qd_lim - qd) / dt, 2 * (q_hi - q - dt * qd) / dt ** 2])
    x0 = np.clip(qdd_des, lo, hi)
    tau0 = M @ x0 + h
    if np.all(np.abs(tau0) <= tau_lim) and np.allclose(x0, qdd_des):
        return tau0, x0
    cons = [{"type": "ineq", "fun": lambda x: tau_lim - (M @ x + h)},
            {"type": "ineq", "fun": lambda x: tau_lim + (M @ x + h)}]
    res = minimize(lambda x: np.sum((M @ x + h - tau_des) ** 2), x0, method="SLSQP",
                   bounds=list(zip(lo, hi)), constraints=cons)
    return M @ res.x + h, res.x


QDD_LIM = 20.0   # rad/s^2, shared by both controllers
QD_LIM = 5.0     # rad/s


def rollout_velocity(policy, q0, q_goal, dt=0.005, T=5.0, tol=0.02, qd_lim=QD_LIM, qdd_lim=QDD_LIM,
                     q_lo=-np.pi, q_hi=np.pi):
    """Kinematic control: q_{t+1} = q_t + dt * QP(qdot_des).  policy(q) -> qdot_des."""
    q, traj = np.array(q0, float), [np.array(q0, float)]
    qd = np.zeros_like(q)
    for _ in range(int(T / dt)):
        if np.linalg.norm(q - q_goal) < tol:
            break
        qd = kinematic_qp(policy(q), q, dt, qd_lim, q_lo, q_hi, qd, qdd_lim)
        q = q + dt * qd
        traj.append(q.copy())
    return np.array(traj)


def rollout_torque(robot, policy, q0, q_goal, dt=0.005, T=5.0, tol=0.02, tau_lim=500.0, qdd_lim=QDD_LIM,
                   qd_lim=QD_LIM, q_lo=-np.pi, q_hi=np.pi, gravity=True, dt_track=0.05):
    """Torque control: qdd_des = (qdot_des - qd)/dt_track (Eq. 35), QP (Eq. 37), forward dynamics.

    ``dt_track`` is the control period used in Eq. (35); the simulation itself runs at ``dt``.
    (Using the 5 ms simulation step in Eq. 35 makes the tracking bang-bang.)
    """
    q, qd = np.array(q0, float), np.zeros(2)
    traj, taus = [q.copy()], []
    for _ in range(int(T / dt)):
        if np.linalg.norm(q - q_goal) < tol:
            break
        qdd_des = (policy(q) - qd) / dt_track
        tau, _ = kinodynamic_qp(robot, q, qd, qdd_des, dt, tau_lim, qdd_lim, qd_lim, q_lo, q_hi, gravity)
        qdd = robot.forward_dynamics(torch.as_tensor(q[None]), torch.as_tensor(qd[None]),
                                     torch.as_tensor(tau[None]), gravity)[0].numpy()
        qd = qd + dt * qdd
        q = q + dt * qd
        traj.append(q.copy())
        taus.append(tau)
    return np.array(traj), np.array(taus)


class path_policy:
    """Blended-velocity policy that follows a precomputed path.

    The geodesic direction -V is replaced by the unit-G-speed tangent of the path
    a short look-ahead ahead of the closest path point, and U by the remaining
    path length -- so path-based methods are executed by exactly the same
    controllers as NES.
    """

    def __init__(self, path, G, q_goal, beta=5.0, lookahead=0.05, speed=None):
        """``speed(q)``: optional factor turning the unit-G-speed tangent into a physical
        velocity (e.g. ``metrics.jacobi_speed``); ``beta`` as in Eq. (33)."""
        self.path = np.asarray(path, float)
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        mid = torch.as_tensor(0.5 * (self.path[1:] + self.path[:-1]))
        d = torch.as_tensor(np.diff(self.path, axis=0))
        self.Gseg = np.sqrt(np.einsum("ni,nij,nj->n", d.numpy(), G(mid).numpy(), d.numpy()))
        self.rem = np.concatenate([np.cumsum(self.Gseg[::-1])[::-1], [0]])
        self.G, self.goal, self.beta, self.la = G, np.asarray(q_goal, float), beta, lookahead
        self.speed = speed
        self.k = 0

    def __call__(self, q):
        # monotone progress along the path
        window = slice(self.k, min(self.k + 200, len(self.path)))
        self.k = window.start + int(np.argmin(np.linalg.norm(self.path[window] - q, axis=1)))
        j = int(np.searchsorted(self.s, self.s[self.k] + self.la))
        j = min(max(j, self.k + 1), len(self.path) - 1)
        v = torch.as_tensor((self.path[j] - q)[None])
        Gq = self.G(torch.as_tensor(q[None])).numpy()[0]
        v = v.numpy()[0]
        v = v / np.sqrt(max(v @ Gq @ v, 1e-12))
        if self.speed is not None:
            v = v * float(self.speed(torch.as_tensor(q[None]))[0])
        U = np.array([self.rem[self.k]])
        return blended_velocity(U, -v[None], q[None], self.goal[None], self.beta)[0]


def finite_diff_torque(robot, traj, dt, gravity=True):
    q = torch.as_tensor(traj)
    qd = torch.as_tensor(np.gradient(traj, dt, axis=0))
    qdd = torch.as_tensor(np.gradient(qd.numpy(), dt, axis=0))
    return robot.inverse_dynamics(q, qd, qdd, gravity).numpy(), qd.numpy()


def evaluate(robot, G, traj, dt, taus=None, gravity=True):
    """Geodesic length (Eq. 3), L2 torque and active torque (Eq. 40)."""
    length = metric_length(G, torch.as_tensor(traj)).item()
    qd = np.gradient(traj, dt, axis=0)
    if taus is None:
        taus, _ = finite_diff_torque(robot, traj, dt, gravity)
    n = min(len(taus), len(qd))
    taus, qd = taus[:n], qd[:n]
    l2 = np.sqrt(np.sum(np.sum(taus ** 2, -1)) * dt)
    active = np.sqrt(np.sum(np.sum(taus * qd, -1) ** 2) * dt)
    return dict(length=length, l2_torque=l2, active_torque=active, duration=len(traj) * dt)
