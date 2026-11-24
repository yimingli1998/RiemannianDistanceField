"""Sanity checks for the tutorial library.  Run from the repository root:  python -m pytest tests -q"""
import os

import numpy as np
import pytest
import torch

from geodf import metrics, nes, rfm
from geodf.robots import PlanarArm

pi = np.pi
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
torch.set_default_dtype(torch.float64)


def test_inertia_matches_eq29():
    arm = PlanarArm()
    q = torch.tensor([[0.3, 0.7]])
    c2 = np.cos(0.7)
    M = arm.inertia(q)[0].numpy()
    assert np.allclose(M, [[12 + 8 * c2, 4 + 4 * c2], [4 + 4 * c2, 4]])


def test_christoffel_matches_coriolis():
    arm = PlanarArm()
    q, v = torch.tensor([[0.3, 0.7]]), torch.tensor([[0.4, -1.1]])
    Gam = metrics.christoffel(arm.inertia, q)[0]
    lhs = torch.einsum("kij,i,j->k", Gam, v[0], v[0])
    rhs = torch.linalg.solve(arm.inertia(q)[0], arm.coriolis(q, v)[0])
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_gravity_is_gradient_of_potential():
    arm = PlanarArm()
    q = torch.tensor([[0.3, -1.2]], requires_grad=True)
    g = torch.autograd.grad(arm.potential(q).sum(), q)[0]
    assert torch.allclose(g, arm.gravity(q.detach()))


def test_rfm_constant_metric_is_exact():
    Gc = torch.tensor([[20.0, 8.0], [8.0, 4.0]])
    G = lambda q: Gc.expand(q.shape[:-1] + (2, 2))
    s = rfm.solve(G, ((-pi, pi), (-pi, pi)), 100, (-0.5, 0.8))
    d = s.grid.points - np.array([-0.5, 0.8])
    exact = np.sqrt(np.einsum("xyi,ij,xyj->xy", d, Gc.numpy(), d))
    assert np.abs(s.U - exact).max() / exact.max() < 5e-3


def test_rfm_matches_agd_reference():
    # reference field computed with agd (Mirebeau's HamiltonFastMarching, first-order scheme):
    # 200x200, source (-0.5, 0.8), kinetic-energy metric
    ref = np.load(os.path.join(DATA, "agd_fmm_values.npy"))
    s = rfm.solve(PlanarArm().inertia, ((-pi, pi), (-pi, pi)), 200, (-0.5, 0.8))
    # agd (first-order scheme) is ~0.07 above the converged solution
    assert np.abs(s.U - ref).mean() < 0.1
    assert np.abs(s.U - ref).max() < 0.2


def test_jacobi_forms():
    arm = PlanarArm()
    q = torch.tensor([[0.2, 0.5]])
    H = 1.2 * arm.P_max
    k = 2 * (H - arm.potential(q))[0]
    assert torch.allclose(metrics.jacobi(arm, H, "paper")(q)[0], k * arm.inertia(q)[0])
    assert torch.allclose(metrics.jacobi(arm, H, "time")(q)[0], arm.inertia(q)[0] / k)
    with pytest.raises(ValueError):
        metrics.jacobi(arm, 0.9 * arm.P_max)


def test_nes_is_symmetric_and_nonnegative():
    torch.set_default_dtype(torch.float32)
    m = nes.NES([-pi, -pi], [pi, pi], hidden=(32, 32))
    a, b = torch.rand(10, 2) * 6 - 3, torch.rand(10, 2) * 6 - 3
    assert torch.allclose(m(a, b), m(b, a), atol=1e-6)
    assert (m(a, b) >= 0).all() and torch.allclose(m(a, a), torch.zeros(10))
    torch.set_default_dtype(torch.float64)
