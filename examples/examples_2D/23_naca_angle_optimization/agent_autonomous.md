# Autonomous runbook

## Goal

Optimize the angle of attack of a NACA airfoil in JAX-FLUIDS by backpropagating
through the differentiable solver and maximizing:

`reward = mean_second_half(Cl / |Cd|)`

The implementation lives in:

- `/Users/paulgarnier/github/phd/JAXFLUIDS/examples/examples_2D/23_naca_angle_optimization/optimize_naca_angle.py`

## Main entrypoints

Full optimization plus final simulation:

```bash
cd /Users/paulgarnier/github/phd/JAXFLUIDS/examples/examples_2D/23_naca_angle_optimization
python optimize_naca_angle.py optimize --output-dir runs/gpu_opt
```

Single forward simulation at a fixed angle:

```bash
cd /Users/paulgarnier/github/phd/JAXFLUIDS/examples/examples_2D/23_naca_angle_optimization
python optimize_naca_angle.py simulate --angle-deg 6.0 --output-dir runs/angle_6deg
```

Fast smoke run on a weak CPU:

```bash
cd /Users/paulgarnier/github/phd/JAXFLUIDS/examples/examples_2D/23_naca_angle_optimization
JAX_PLATFORMS=cpu python optimize_naca_angle.py optimize \
  --output-dir runs/smoke \
  --sim-time 1.0 \
  --max-iters 2 \
  --learning-rate-deg 0.5 \
  --skip-final-simulation \
  --no-generate-xdmf
```

## What the script does

1. Builds a rotated NACA contour in JAX and converts it to a differentiable signed-distance field on the simulation grid.
2. Uses `SimulationManager.feed_forward()` so the flow rollout stays differentiable.
3. Computes `Cd`, `Cl`, and `Cl / |Cd|` over time from the pressure and viscous stresses.
4. Uses `jax.value_and_grad()` to differentiate the reward with respect to the angle.
5. Updates the angle with Adam-style gradient ascent until the step, reward change, and gradient all become small.
6. Reruns the best angle as a conventional simulation to write HDF5 outputs, figures, CSVs, and optional XDMF.

## Output files to inspect

The optimization output directory contains:

- `environment.json`
  - first place to check backend/device detection and precision
- `run_config.json`
  - exact CLI arguments used for the run
- `history.csv`
  - per-iteration reward, Cd, Cl, gradient, step, and non-finite counters
- `final_summary.json`
  - best angle, best reward, figure paths, and the final simulation summary path
- `figures/optimization_history.png`
  - reward, force, angle, and gradient history
- `figures/airfoil_overlay.png`
  - initial vs optimized airfoil orientation
- `best_iter_timeseries.csv`
  - force history from the best differentiable rollout

The final conventional rerun is recorded under:

- `final_simulation/simulation_summary.json`
  - includes the resolved simulation output path
- `final_simulation/results/<case>/output.log`
  - JAX-FLUIDS runtime log for the final simulation
- `final_simulation/results/<case>/domain/*.h5`
  - flow fields
- `final_simulation/results/<case>/domain/data_time_series.xdmf`
  - ParaView timeseries if XDMF generation is enabled

## GPU notes

This code is backend-agnostic. It runs on CPU here, but it should move to GPU
automatically when the target environment has a GPU-enabled JAX install.

Recommended GPU launch:

```bash
cd /Users/paulgarnier/github/phd/JAXFLUIDS/examples/examples_2D/23_naca_angle_optimization
XLA_PYTHON_CLIENT_PREALLOCATE=false python optimize_naca_angle.py optimize --output-dir runs/gpu_opt
```

If something goes wrong on GPU, inspect these in order:

1. `environment.json`
   - verify `jax_backend` is `gpu`
   - verify the expected GPU appears in `jax_devices`
2. `history.csv`
   - check `grad_source`, `reward_is_finite`, `grad_is_finite`, and the non-finite counters
3. `final_summary.json`
   - confirms the best angle found and whether the final simulation was run
4. `final_simulation/simulation_summary.json`
   - contains the actual output path selected by JAX-FLUIDS
5. `final_simulation/results/<case>/output.log`
   - look here for the solver-side failure details

Important local note:

- Apple Metal was not a reliable backend for this workflow during setup checks.
- For local debugging on this machine, prefer `JAX_PLATFORMS=cpu`.
- For the real optimization run, prefer a CUDA-backed GPU environment.

If GPU memory or compile pressure is too high, reduce one or more of:

- `--sim-time`
- `--max-iters`
- `--geometry-samples`
- `--inner-steps`

If numerical stability is an issue, try:

- `--solver-precision double`
- smaller `--learning-rate-deg`
- tighter angle bounds via `--min-angle-deg` and `--max-angle-deg`

If throughput matters more than precision, try:

- `--solver-precision single`

## Expected success signal

A healthy optimization run should leave:

- finite rewards in `history.csv`
- a sensible best angle in `final_summary.json`
- `figures/optimization_history.png`
- `final_simulation/figures/final_simulation_report.png`

If those four are present and finite, the pipeline is working end to end.
