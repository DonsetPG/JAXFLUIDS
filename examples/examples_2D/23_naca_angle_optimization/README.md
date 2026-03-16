# NACA angle-of-attack optimization

This example optimizes the angle of attack of a rotated NACA airfoil with the
differentiable `feed_forward()` solver path in JAX-FLUIDS.

What it does:

- builds a differentiable rotated NACA level-set field on the simulation grid
- runs a Reynolds-100 viscous flow case
- computes the reward `mean(Cl / |Cd|)` over the second half of the rollout
- differentiates the reward with respect to the angle of attack
- updates the angle with Adam-style gradient ascent until convergence
- reruns the best angle as a full simulation and writes figures, CSVs, HDF5, and optional XDMF

Quick start:

```bash
cd /Users/paulgarnier/github/phd/JAXFLUIDS/examples/examples_2D/23_naca_angle_optimization
python optimize_naca_angle.py optimize --output-dir runs/default_opt
```

Single forward simulation at a fixed angle:

```bash
cd /Users/paulgarnier/github/phd/JAXFLUIDS/examples/examples_2D/23_naca_angle_optimization
python optimize_naca_angle.py simulate --angle-deg 6.0 --output-dir runs/angle_6deg
```

Useful options:

- `--initial-angle-deg`, `--min-angle-deg`, `--max-angle-deg`: angle setup
- `--sim-time`: rollout horizon used both in optimization and the final simulation
- `--max-iters`: optimization budget
- `--learning-rate-deg`: Adam step size in degrees
- `--solver-precision single|double`: precision override
- `--skip-final-simulation`: optimize only, without the final disk-writing rerun
- `--no-generate-xdmf`: skip XDMF conversion

Main outputs:

- `history.csv`: optimization metrics by iteration
- `final_summary.json`: best angle, reward, figures, and final simulation paths
- `figures/optimization_history.png`: reward, force, angle, gradient history
- `figures/airfoil_overlay.png`: initial vs optimized airfoil orientation
- `best_iter_timeseries.csv`: best differentiable rollout force history
- `final_simulation/simulation_summary.json`: full-simulation outputs for the optimized angle

Notes:

- The case setup file contains a placeholder level set; the script injects the actual rotated NACA geometry at runtime.
- The default flow setup uses `rho = 1`, `U = 1`, `mu = 0.01`, `c = 1`, so `Re = 100`.
