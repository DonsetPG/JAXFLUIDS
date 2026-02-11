from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids import InitializationManager, InputManager, SimulationManager
from jaxfluids.feed_forward.data_types import FeedForwardSetup

from bezier_shape import render_shape_preview, write_levelset_h5


def clamp_points_to_ring_jax(points: jax.Array, r_min: float, r_max: float) -> jax.Array:
    radii = jnp.linalg.norm(points, axis=1) + 1.0e-14
    clamped = jnp.clip(radii, r_min, r_max)
    return points * (clamped / radii)[:, None]


def cubic_bezier_jax(
    p0: jax.Array,
    p1: jax.Array,
    p2: jax.Array,
    p3: jax.Array,
    t: jax.Array,
) -> jax.Array:
    omt = 1.0 - t
    return (
        (omt**3)[:, None] * p0[None, :]
        + (3.0 * omt**2 * t)[:, None] * p1[None, :]
        + (3.0 * omt * t**2)[:, None] * p2[None, :]
        + (t**3)[:, None] * p3[None, :]
    )


def bezier_closed_from_four_points_jax(
    control_points: jax.Array,
    samples_per_segment: int,
    tension: float,
) -> jax.Array:
    t = jnp.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    segments = []
    for i in range(4):
        p_im1 = control_points[(i - 1) % 4]
        p_i = control_points[i]
        p_ip1 = control_points[(i + 1) % 4]
        p_ip2 = control_points[(i + 2) % 4]
        tangent_i = 0.5 * (p_ip1 - p_im1)
        tangent_ip1 = 0.5 * (p_ip2 - p_i)
        h1 = p_i + (tension / 3.0) * tangent_i
        h2 = p_ip1 - (tension / 3.0) * tangent_ip1
        segments.append(cubic_bezier_jax(p_i, h1, h2, p_ip1, t))
    contour = jnp.concatenate(segments, axis=0)
    contour = jnp.concatenate([contour, contour[:1]], axis=0)
    return contour


def signed_distance_from_contour_jax(
    contour: jax.Array,
    control_points: jax.Array,
    x: jax.Array,
    y: jax.Array,
    softmin_temperature: float = 0.02,
    inside_edge_sharpness: float = 30.0,
    inside_score_sharpness: float = 20.0,
) -> jax.Array:
    xx, yy = jnp.meshgrid(x, y, indexing="ij")

    p0 = contour[:-1]  # (S,2)
    p1 = contour[1:]   # (S,2)
    x0, y0 = p0[:, 0], p0[:, 1]
    x1, y1 = p1[:, 0], p1[:, 1]

    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy + 1.0e-14

    xg = xx[..., None]
    yg = yy[..., None]
    t = ((xg - x0) * vx + (yg - y0) * vy) / denom
    t = jnp.clip(t, 0.0, 1.0)
    cx = x0 + t * vx
    cy = y0 + t * vy
    dist = jnp.sqrt((xg - cx) ** 2 + (yg - cy) ** 2 + 1.0e-14)

    # Smooth min distance to polyline.
    d = -softmin_temperature * jax.nn.logsumexp(-dist / softmin_temperature, axis=-1)

    # Smooth inside/outside using the control polygon half-space scores.
    q0 = control_points
    q1 = jnp.roll(control_points, shift=-1, axis=0)
    ex = q1[:, 0] - q0[:, 0]
    ey = q1[:, 1] - q0[:, 1]
    cross_poly = ex * (yg - q0[:, 1]) - ey * (xg - q0[:, 0])
    inside_each = jax.nn.sigmoid(inside_edge_sharpness * cross_poly)
    inside_score = jnp.mean(inside_each, axis=-1)
    inside_prob = jax.nn.sigmoid(inside_score_sharpness * (inside_score - 0.5))
    sign = 1.0 - 2.0 * inside_prob

    return sign * d


def prepare_feedforward_compatible_numerical_setup(numerical_setup: Dict) -> Dict:
    num = copy.deepcopy(numerical_setup)
    levelset = num.setdefault("levelset", {})
    levelset.setdefault("interface_flux", {})["is_cell_based_computation"] = False
    levelset.setdefault("extension", {}).setdefault("primitives", {}).setdefault("interpolation", {})[
        "is_cell_based_computation"
    ] = False
    levelset.setdefault("extension", {}).setdefault("solids", {}).setdefault("interpolation", {})[
        "is_cell_based_computation"
    ] = False
    levelset.setdefault("mixing", {}).setdefault("conservatives", {})["is_cell_based_computation"] = False
    levelset.setdefault("mixing", {}).setdefault("solids", {})["is_cell_based_computation"] = False
    return num


def compute_cd_cl_series(
    primitives_t: jax.Array,
    levelset: jax.Array,
    x: jax.Array,
    y: jax.Array,
    dynamic_viscosity: float,
) -> Tuple[jax.Array, jax.Array]:
    # Static geometry terms from levelset.
    dphi_dx = jnp.gradient(levelset, x, axis=0)
    dphi_dy = jnp.gradient(levelset, y, axis=1)
    normal_norm = jnp.sqrt(dphi_dx**2 + dphi_dy**2) + 1.0e-14
    normal_x = dphi_dx / normal_norm
    normal_y = dphi_dy / normal_norm

    dx = jnp.abs(jnp.gradient(x))
    dy = jnp.abs(jnp.gradient(y))
    cell_area = dx[:, None] * dy[None, :]
    interface_eps = 1.5 * jnp.minimum(jnp.min(dx), jnp.min(dy))
    # Smooth Gaussian delta avoids hard thresholding that kills gradients.
    delta = jnp.exp(-(levelset / interface_eps) ** 2) / (jnp.sqrt(jnp.pi) * interface_eps)
    surface_weight = delta * normal_norm

    def one_snapshot(pr: jax.Array) -> Tuple[jax.Array, jax.Array]:
        u = pr[1, :, :, 0]
        v = pr[2, :, :, 0]
        p = pr[4, :, :, 0]

        du_dx = jnp.gradient(u, x, axis=0)
        du_dy = jnp.gradient(u, y, axis=1)
        dv_dx = jnp.gradient(v, x, axis=0)
        dv_dy = jnp.gradient(v, y, axis=1)

        tau_xx = -p + 2.0 * dynamic_viscosity * du_dx
        tau_xy = dynamic_viscosity * (du_dy + dv_dx)
        tau_yy = -p + 2.0 * dynamic_viscosity * dv_dy

        traction_x = tau_xx * normal_x + tau_xy * normal_y
        traction_y = tau_xy * normal_x + tau_yy * normal_y

        force_x = jnp.sum(traction_x * surface_weight * cell_area)
        force_y = jnp.sum(traction_y * surface_weight * cell_area)
        return force_x, force_y

    force_x, force_y = jax.vmap(one_snapshot, in_axes=0, out_axes=0)(primitives_t)
    coeff_scale = 1.0  # 0.5 * rho * U^2 * S_ref with rho=1,U=1,S=2
    return force_x / coeff_scale, force_y / coeff_scale


def build_objective(
    sim_manager: SimulationManager,
    primes_init: jax.Array,
    x_cells: jax.Array,
    y_cells: jax.Array,
    dt: float,
    t0: float,
    feed_forward_setup: FeedForwardSetup,
    r_min: float,
    r_max: float,
    bezier_samples: int,
) -> callable:
    def objective(theta: jax.Array):
        theta = clamp_points_to_ring_jax(theta, r_min, r_max)
        contour = bezier_closed_from_four_points_jax(theta, bezier_samples, tension=1.0)
        levelset_xy = signed_distance_from_contour_jax(contour, theta, x_cells, y_cells)
        levelset = levelset_xy[..., None]

        solution, times = sim_manager.feed_forward(
            batch_primes_init=primes_init[None, ...],
            physical_timestep_size=jnp.array([dt]),
            t_start=jnp.array([t0]),
            feed_forward_setup=feed_forward_setup,
            batch_levelset_init=levelset[None, ...],
        )

        primitives = solution["primitives"][0]  # (T, 5, Nx, Ny, Nz)
        levelset_single = solution["levelset"][0, :, :, 0]

        cd, cl = compute_cd_cl_series(
            primitives_t=primitives,
            levelset=levelset_single,
            x=x_cells,
            y=y_cells,
            dynamic_viscosity=0.01,
        )

        ratio = cl / (jnp.abs(cd) + 1.0e-12)
        half_idx = ratio.shape[0] // 2
        reward = jnp.mean(ratio[half_idx:])
        aux = {
            "reward": reward,
            "mean_cd_second_half": jnp.mean(cd[half_idx:]),
            "mean_cl_second_half": jnp.mean(cl[half_idx:]),
            "end_time": times[0, -1],
            "levelset_xy": levelset_xy,
            "theta": theta,
        }
        return reward, aux

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Full differentiable VIQ optimization with 4 Bezier points.")
    parser.add_argument("--case-setup", default="case_setup_viq_re200.json")
    parser.add_argument("--numerical-setup", default="numerical_setup.json")
    parser.add_argument("--output-dir", default="full_diff_opt_t100")
    parser.add_argument("--sim-time", type=float, default=100.0)
    parser.add_argument("--num-iters", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-step-norm", type=float, default=0.05)
    parser.add_argument("--ring-r-min", type=float, default=0.7)
    parser.add_argument("--ring-r-max", type=float, default=1.3)
    parser.add_argument("--bezier-samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jit", action="store_true", help="JIT compile value+grad function.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "iter_shapes").mkdir(parents=True, exist_ok=True)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    case_dict = json.loads(Path(args.case_setup).read_text())
    numerical_dict = json.loads(Path(args.numerical_setup).read_text())
    numerical_ff_dict = prepare_feedforward_compatible_numerical_setup(numerical_dict)

    input_manager = InputManager(case_dict, numerical_ff_dict)
    initialization_manager = InitializationManager(input_manager)
    sim_manager = SimulationManager(input_manager)
    jxf_buffers = initialization_manager.initialization()

    nhx, nhy, nhz = sim_manager.domain_information.domain_slices_conservatives
    primes_init = jxf_buffers.simulation_buffers.material_fields.primitives[..., nhx, nhy, nhz]
    x_raw, y_raw, _ = sim_manager.domain_information.get_global_cell_centers_unsplit()
    x_cells = jnp.asarray(x_raw).reshape(-1)
    y_cells = jnp.asarray(y_raw).reshape(-1)

    dt = float(jxf_buffers.time_control_variables.physical_timestep_size)
    t0 = float(jxf_buffers.time_control_variables.physical_simulation_time)

    total_steps = int(np.ceil(args.sim_time / dt))
    outer_steps = int(np.ceil(total_steps / args.inner_steps))
    ff_setup = FeedForwardSetup(
        outer_steps=outer_steps,
        inner_steps=args.inner_steps,
        is_scan=True,
        is_checkpoint_inner_step=True,
        is_checkpoint_integration_step=False,
        is_include_t0=True,
        is_include_halos=False,
    )

    objective = build_objective(
        sim_manager=sim_manager,
        primes_init=primes_init,
        x_cells=x_cells,
        y_cells=y_cells,
        dt=dt,
        t0=t0,
        feed_forward_setup=ff_setup,
        r_min=args.ring_r_min,
        r_max=args.ring_r_max,
        bezier_samples=args.bezier_samples,
    )
    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    if args.jit:
        value_and_grad = jax.jit(value_and_grad)

    theta = jnp.array(
        [
            [1.05, 0.00],
            [0.05, 1.00],
            [-1.00, 0.10],
            [0.00, -0.95],
        ],
        dtype=jnp.float64,
    )
    theta = clamp_points_to_ring_jax(theta, args.ring_r_min, args.ring_r_max)

    run_config = {
        "case_setup": str(Path(args.case_setup).resolve()),
        "numerical_setup": str(Path(args.numerical_setup).resolve()),
        "sim_time": args.sim_time,
        "dt_used_in_feed_forward": dt,
        "total_steps": total_steps,
        "outer_steps": outer_steps,
        "inner_steps": args.inner_steps,
        "num_iters": args.num_iters,
        "learning_rate": args.learning_rate,
        "max_step_norm": args.max_step_norm,
        "ring_r_min": args.ring_r_min,
        "ring_r_max": args.ring_r_max,
        "bezier_samples": args.bezier_samples,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    history_csv = out_dir / "history.csv"
    with history_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iter",
                "reward",
                "mean_cd_second_half",
                "mean_cl_second_half",
                "grad_norm",
                "step_norm",
                "wall_seconds",
                "sim_end_time",
            ]
        )

    history = []
    for it in range(args.num_iters):
        t_start_iter = time.time()
        (reward, aux), grad = value_and_grad(theta)
        reward = jnp.nan_to_num(reward, nan=-1.0e6, posinf=1.0e6, neginf=-1.0e6)
        aux = {
            "mean_cd_second_half": jnp.nan_to_num(aux["mean_cd_second_half"], nan=1.0e6, posinf=1.0e6, neginf=-1.0e6),
            "mean_cl_second_half": jnp.nan_to_num(aux["mean_cl_second_half"], nan=0.0, posinf=1.0e6, neginf=-1.0e6),
            "end_time": jnp.nan_to_num(aux["end_time"], nan=0.0, posinf=0.0, neginf=0.0),
            "levelset_xy": jnp.nan_to_num(aux["levelset_xy"], nan=0.0, posinf=0.0, neginf=0.0),
            "theta": jnp.nan_to_num(aux["theta"], nan=0.0, posinf=0.0, neginf=0.0),
        }
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        reward_f = float(reward)
        grad_norm = float(jnp.linalg.norm(grad))

        step = args.learning_rate * grad
        step_norm = float(jnp.linalg.norm(step))
        if step_norm > args.max_step_norm and step_norm > 0.0:
            step = step * (args.max_step_norm / (step_norm + 1.0e-14))
            step_norm = args.max_step_norm

        theta = clamp_points_to_ring_jax(theta + step, args.ring_r_min, args.ring_r_max)
        elapsed = time.time() - t_start_iter

        theta_np = np.asarray(theta)
        levelset_xy_np = np.asarray(aux["levelset_xy"])
        iter_dir = out_dir / "iter_shapes" / f"iter_{it:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        write_levelset_h5(levelset_xy_np, iter_dir / "levelset.h5")
        contour_np = np.asarray(
            bezier_closed_from_four_points_jax(theta, args.bezier_samples, tension=1.0)
        )
        render_shape_preview(contour_np, theta_np, iter_dir / "shape.png")

        row = {
            "iter": it,
            "reward": reward_f,
            "mean_cd_second_half": float(aux["mean_cd_second_half"]),
            "mean_cl_second_half": float(aux["mean_cl_second_half"]),
            "grad_norm": grad_norm,
            "step_norm": step_norm,
            "wall_seconds": elapsed,
            "sim_end_time": float(aux["end_time"]),
            "theta": theta_np.tolist(),
        }
        history.append(row)
        with history_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row["iter"],
                    row["reward"],
                    row["mean_cd_second_half"],
                    row["mean_cl_second_half"],
                    row["grad_norm"],
                    row["step_norm"],
                    row["wall_seconds"],
                    row["sim_end_time"],
                ]
            )
        (iter_dir / "metrics.json").write_text(json.dumps(row, indent=2))
        print(
            f"[iter {it:03d}] reward={row['reward']:.6f} "
            f"Cd={row['mean_cd_second_half']:.6f} Cl={row['mean_cl_second_half']:.6f} "
            f"|grad|={row['grad_norm']:.6e} step={row['step_norm']:.6e} "
            f"t={row['wall_seconds']:.2f}s"
        )

    final = {
        "final_theta": np.asarray(theta).tolist(),
        "history": history,
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final, indent=2))
    print(f"Saved optimization outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
