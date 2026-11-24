# A Riemannian Take on Distance Fields and Geodesic Flows in Robotics — 2D tutorial

A hands-on introduction to the method of the paper
[A Riemannian Take on Distance Fields and Geodesic Flows in Robotics](https://arxiv.org/abs/2412.05197)
(see also the [project website](https://sites.google.com/view/geodf)).  Every notebook is executed,
so you can read the results and figures directly on GitHub, or run it yourself on a laptop CPU.

> The notebooks and the `geodf` library in this repository were rewritten by
> [Claude](https://www.anthropic.com/claude) (Anthropic's AI assistant) from the authors' research
> code, as a clean and self-contained tutorial of the method.

## The idea

Describe what makes a motion expensive with a **Riemannian metric** $G(q)$ — a positive-definite
matrix at every point of a space, saying how costly it is to move in each direction.  The cheapest
motions are the **geodesics**, and all of them are encoded in a **distance field** $U$ that solves
the Riemannian eikonal equation $\|\nabla U\|_{G^{-1}}=1$.  Its gradient gives a **geodesic flow**:
from any point, the direction of the optimal motion.

* The space can be anything with coordinates: a plane or a workspace, the joint space of an arm,
  the state space of a cart-pole, ...
* The metric encodes the task: inertia (low kinetic effort), energy conservation, end-effector
  distance, stability, distance to obstacles, ...
* The distance field can be computed on a grid (Riemannian fast marching) or learned with a neural
  network for **all pairs** of points (the Neural Riemannian Eikonal Solver, NES), which scales to
  high dimensions, is differentiable, and is fast enough to be used as a feedback policy.

The tutorial uses 2-D examples so that everything can be plotted; the method itself is not tied
to two dimensions or to joint space (the paper applies it to a 7-joint arm).

## Notebooks

| | Notebook | What you learn |
|---|---|---|
| 0 | [What can it do?](00_what_can_it_do.ipynb) | seven examples, one figure each: navigation among obstacles, low-effort arm motions, reaching without IK, straight end-effector paths, a stable cart-pole, a family of metrics in one network, a reactive controller |
| 1 | [Riemannian metrics and geodesics](01_riemannian_metrics_and_geodesics.ipynb) | metrics on the plane and on a robot's configuration space, path length, geodesics by shooting |
| 2 | [The eikonal equation and fast marching](02_eikonal_equation_and_fast_marching.ipynb) | distance fields, the Riemannian eikonal equation, a grid solver, the geodesic flow and backtracking |
| 3 | [The Neural Riemannian Eikonal Solver](03_neural_eikonal_solver.ipynb) | a network that is a distance by construction, physics-informed training, geodesics between any pairs |
| 4 | [Conditioned NES](04_conditioned_nes.ipynb) | goals given in another space (reaching without IK); one network for a family of metrics |
| 5 | [From geodesic flows to robot motion](05_from_geodesics_to_robot_control.ipynb) | Maupertuis' principle (Jacobi metric), a feedback policy, velocity/torque control, robustness to pushes |
| 6 | [Designing metrics for tasks](06_designing_metrics_for_tasks.ipynb) | end-effector distance, stability, obstacles; how to parameterise NES for difficult metrics |
| A | [Geometry-aware sampling](A_geometry_aware_sampling.ipynb) | RM-MALA: sampling uniformly w.r.t. the Riemannian volume |

Start with notebook 0 to see what the method does, then read 1-6 in order.

## Running the notebooks

Everything installs with `pip` (Python ≥ 3.9; NumPy, SciPy, Matplotlib, PyTorch, JupyterLab).  No
GPU, conda or compiled solver is needed: the grid solver used in the notebooks is implemented in
`geodf/rfm.py` in plain NumPy (the [agd](https://github.com/Mirebeau/AdaptiveGridDiscretizations)
library is *not* required; a field computed with it is only stored in `tests/data/` as a reference
for the tests).

```bash
pip install -r requirements.txt
```

```bash
jupyter lab
```

Open the notebooks from the repository root.  Trained networks are included in `checkpoints/`,
so every notebook runs in a few seconds to a couple of minutes on a CPU.  To train a network
yourself, set `RETRAIN = True` in the notebook (about 2 minutes per network on 4 CPU cores).

## The code: `geodf/`

A small library (~1k lines of PyTorch / NumPy) used by the notebooks:

| File | Content |
|---|---|
| `robots.py` | 2-link planar arm (inertia, gravity, dynamics, kinematics, IK) and cart-pole |
| `metrics.py` | kinetic-energy, Jacobi, pull-back, stability-aware and obstacle metrics; Christoffel symbols; path length |
| `rfm.py` | Riemannian fast-marching grid solver, geodesic flow and backtracking |
| `nes.py` | NES, conditioned NES, eikonal loss, training, geodesic flow and backtracking |
| `obstacles.py` | signed distances and the speed field used for obstacles |
| `control.py` | blended geodesic policy, velocity- and torque-level QP controllers, simulation |
| `sampling.py` | RM-MALA sampler |
| `plotting.py` | figures |

`python -m pytest tests -q` runs a few sanity checks (dynamics, Christoffel symbols, the grid
solver against an exact solution and against a reference field computed with
[agd](https://github.com/Mirebeau/AdaptiveGridDiscretizations)).

## About this code

This repository is a rewrite, by Claude (Anthropic), of the authors' research code into a tutorial:
the solvers, the neural eikonal solver, the robot models and the notebooks were re-implemented and
re-trained from scratch.  Numbers in the notebooks can therefore differ slightly from those
reported in the paper.

## License

Copyright (c) 2024 Idiap Research Institute, https://idiap.ch/
