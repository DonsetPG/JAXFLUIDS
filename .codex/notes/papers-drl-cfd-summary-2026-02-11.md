# DRL+CFD Papers Summary (for JAX differentiable replacement)

Date: 2026-02-11
Sources:
- `/Users/paulgarnier/github/phd/JAXFLUIDS/papers_to_read/viq.pdf`
- `/Users/paulgarnier/github/phd/JAXFLUIDS/papers_to_read/rabault.pdf`
- `/Users/paulgarnier/github/phd/JAXFLUIDS/papers_to_read/garnier.pdf`

## 1) Viquerat et al. (`viq.pdf`) - direct shape optimization with DRL

- Problem: generate 2D obstacle shapes that maximize lift-to-drag ratio in CFD.
- Environment:
  - Shape represented by constrained control points and reconstructed with Bezier curves.
  - CFD run (FEniCS in paper) evaluates aerodynamic metrics.
- Agent setup in paper:
  - PPO with fully connected ANN (two hidden layers, 512 units each).
  - "Degenerate" DRL: one action per episode (direct optimizer behavior).
- Objective/reward:
  - Baseline reward is lift-to-drag relative to cylinder reference:
    - `r_t = <Cl/|Cd|> - <Cl/|Cd|>_cyl`
  - Temporal averaging is done over later part of simulation window.
- Key relevance for JAX plan:
  - This is closest to replacing DRL with direct gradient ascent over shape parameters.
  - One-step episode design maps naturally to differentiable optimization over design variables.

## 2) Rabault et al. (`rabault.pdf`) - active flow control with two jets on a cylinder

- Problem: reduce drag (and stabilize wake) around a cylinder in 2D unsteady flow.
- Environment:
  - Reynolds number `Re = 100`.
  - Two synthetic jets (top/bottom of cylinder) with controllable mass flow rates.
  - Observation from velocity probes in wake region.
  - Unsteady CFD with fixed timestep (`dt = 5e-3` in non-dimensional units).
- Agent setup in paper:
  - PPO + fully connected ANN.
  - Episode-based control from an initialized vortex-shedding state.
- Objective/reward:
  - `r_t = -<C_D>_T - 0.2*|<C_L>_T|`
  - Sliding average over one shedding cycle.
- Reported result (paper claim):
  - About 8% drag reduction with low actuation level (~0.5% normalized mass flow in established regime).
- Key relevance for JAX plan:
  - Replace policy learning with direct differentiation through flow rollout wrt jet control sequence or low-dimensional jet parameters.

## 3) Garnier et al. (`garnier.pdf`) - review + two-cylinder positioning case

- Scope:
  - Broad review of DRL in fluid mechanics, including control and optimization tasks.
  - Includes synthesis of prior works plus additional case studies.
- Most relevant case for our roadmap:
  - Two-cylinder setup (square main cylinder + small cylindrical control cylinder with variable position).
  - Goal: place small cylinder to reduce combined drag relative to main cylinder baseline.
  - Reported comparisons against classical adjoint results; similar optima reported at `Re = 40` and `Re = 100`.
  - Includes direct optimization framing and transfer-learning experiments across `Re = 10, 40, 100`.
- Key relevance for JAX plan:
  - Position optimization over `(x, y)` is a direct fit for gradient-based optimization if CFD pipeline is differentiable w.r.t. geometry/placement.

## Cross-paper extraction for implementation

- Case A (shape optimization): optimize shape control parameters against lift/drag objective.
- Case B (jet control): optimize jet actuation parameters/trajectory against drag/lift objective.
- Case C (small-cylinder placement): optimize control-cylinder position `(x, y)` against total drag objective.

All three are design/control optimization problems where DRL can be replaced by direct differentiation if:
- the simulation graph is differentiable through state rollout and objective aggregation; and
- geometry/control parameterization is differentiable and numerically stable.
