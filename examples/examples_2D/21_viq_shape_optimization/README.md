# VIQ-style shape optimization baseline

This case mirrors the CFD setup in `viq.pdf`:

- Domain size: `Lx = 45`, `Ly = 30`
- Typical obstacle size: `R_ref = 1`
- Inflow: `U_in = 1`
- Density: `rho = 1`
- Reynolds number: `Re = 200` (via `mu = 0.01` from `Re = 2*rho*U*R/mu`)
- Top/bottom: free-slip proxy (`SYMMETRY`)
- Outflow: traction-free proxy (`ZEROGRADIENT`)
- Obstacle wall: no-slip immersed solid via level-set

Run:

```bash
cd examples/examples_2D/21_viq_shape_optimization
../../../../venvjax/bin/python run_viq_shape.py
```

Outputs are saved under:

- `results/viq_shape_re200_baseline/domain/*.h5`
- `results/viq_shape_re200_baseline/domain/data_time_series.xdmf` (open this in ParaView)
- `results/viq_shape_re200_baseline/drag_lift_timeseries.csv`
- `results/viq_shape_re200_baseline/drag_lift_summary.json`
- `results/viq_shape_re200_baseline/drag_lift_coefficients.png`

XDMF generation options:

- Native during simulation: set `"output.is_xdmf": true` in `numerical_setup.json`.
- Convert an existing `domain/*.h5` folder after the run:

```bash
cd examples/examples_2D/21_viq_shape_optimization
SKIP_SIMULATION=1 MAKE_XDMF_ONLY=1 ../../../venvjax/bin/python run_viq_shape.py
```

## 4-point Bezier shape generation

Generate 10 paper-style Bezier shapes (4 control points each):

```bash
cd examples/examples_2D/21_viq_shape_optimization
../../../venvjax/bin/python generate_bezier_shapes.py
```

Outputs:

- `generated_shapes_4pt_bezier/shape_XX_levelset.h5` (levelset usable by JAXFluids)
- `generated_shapes_4pt_bezier/shape_XX.png` (preview)
- `generated_shapes_4pt_bezier/shape_XX.json` (control points + paths)
- `generated_shapes_4pt_bezier/manifest.csv`

## Differentiability check + first gradient step

Run one finite-difference differentiability check and one gradient-ascent update
on the VIQ reward `mean(Cl/|Cd|)` with 4 optimized Bezier points:

```bash
cd examples/examples_2D/21_viq_shape_optimization
../../../venvjax/bin/python optimize_bezier_shape_viq.py
```

Outputs:

- `optimization_4pt_bezier/optimization_summary.json`
- `optimization_4pt_bezier/base/`
- `optimization_4pt_bezier/plus/`
- `optimization_4pt_bezier/minus/`
- `optimization_4pt_bezier/after_step_1/`
