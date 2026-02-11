# Rabault Jet-Optimization Environment (Re=100)

This example recreates the 2D cylinder configuration from:

- Rabault et al., *Artificial Neural Networks Trained Through Deep Reinforcement Learning Discover Control Strategies for Active Flow Control* (2018, arXiv:1808.07664)

Implemented paper parameters:

- Cylinder diameter: `D = 1` (radius `R = 0.5`)
- Domain size: `Lx = 22`, `H = 4.1`
- Domain extents: `x in [-2, 20]`, `y in [-2, 2.1]`
- Cylinder offset relative to domain centerline: `0.05` in `y`
- Mean inflow velocity: `U_bar = 1`
- Density: `rho = 1`
- Reynolds number: `Re = 100` (via `mu = 0.01`)
- Jets:
  - centers at `90 deg` and `270 deg`
  - angular width `10 deg`
  - synthetic condition `Q1 + Q2 = 0`
  - limit `|Q*| <= 0.06`

Mesh choice (reasonable starting point):

- `Nx = 280`, `Ny = 160`, with piecewise stretching around the cylinder/wake region.

## Run

```bash
cd examples/examples_2D/22_rabault_jet_optimization
../../../venvjax/bin/python run_rabault_jet.py --q1-star 0.0 --end-time 8.0 --save-dt 0.2
```

`--q1-star` is the normalized jet command `Q1*`; `Q2*` is set automatically to `-Q1*`.
Jet forcing is automatically enabled only when `|q1-star| > 0`.

## Outputs

Saved under:

- `results/<case_name>/domain/*.h5`
- `results/<case_name>/domain/data_time_series.xdmf`
- `results/<case_name>/drag_lift_timeseries.csv`
- `results/<case_name>/drag_lift_summary.json`
- `results/<case_name>/drag_lift_coefficients.png`
- `results/<case_name>/rabault_parameters.json`

## Notes

- Drag/lift are computed by integrating pressure + viscous traction around the immersed cylinder interface (from level-set output).
- Because this is a Cartesian immersed-boundary setup, jet actuation is implemented as localized synthetic forcing around the cylinder slots while keeping the paper’s physical scales and jet parameters.
