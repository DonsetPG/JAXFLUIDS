from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids import InitializationManager, InputManager, SimulationManager
from jaxfluids.feed_forward.data_types import FeedForwardSetup

from bezier_shape import render_shape_preview, write_levelset_h5


def maybe_init_wandb(args: argparse.Namespace, run_config: Dict[str, Any], out_dir: Path):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging requested but `wandb` is not installed. "
            "Install it with: pip install wandb"
        ) from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        mode=args.wandb_mode,
        tags=tags if tags else None,
        dir=str(out_dir.resolve()),
        config=run_config,
    )
    return run


def make_wandb_iter_payload(row: Dict[str, Any], theta_np: np.ndarray) -> Dict[str, float]:
    payload: Dict[str, float] = {
        "reward": float(row["reward"]),
        "drag": float(row["mean_cd_second_half"]),
        "lift": float(row["mean_cl_second_half"]),
        "mean_cd_second_half": float(row["mean_cd_second_half"]),
        "mean_cl_second_half": float(row["mean_cl_second_half"]),
        "grad_norm": float(row["grad_norm"]),
        "step_norm": float(row["step_norm"]),
        "wall_seconds": float(row["wall_seconds"]),
        "sim_end_time": float(row["sim_end_time"]),
    }
    for i in range(4):
        payload[f"control_point_{i}_x"] = float(theta_np[i, 0])
        payload[f"control_point_{i}_y"] = float(theta_np[i, 1])
    return payload


def maybe_apply_solver_precision_override(numerical_setup: Dict[str, Any], solver_precision: str) -> Dict[str, Any]:
    if solver_precision == "from-setup":
        return numerical_setup
    num = copy.deepcopy(numerical_setup)
    precision = num.setdefault("precision", {})
    if solver_precision == "single":
        precision["is_double_precision_compute"] = False
        precision["is_double_precision_output"] = False
    elif solver_precision == "double":
        precision["is_double_precision_compute"] = True
        precision["is_double_precision_output"] = True
    else:
        raise ValueError(f"Unknown solver_precision: {solver_precision}")
    return num


def finite_float_or_none(value: Any) -> Any:
    value_f = float(value)
    return value_f if np.isfinite(value_f) else None


def build_objective_value_fn(objective_fn: Callable, do_jit: bool) -> Callable[[jax.Array], jax.Array]:
    def objective_value(theta_batch: jax.Array) -> jax.Array:
        reward, _ = objective_fn(theta_batch)
        return reward

    return jax.jit(objective_value) if do_jit else objective_value


def build_backward_diagnostic_grad_fns(objective_fn: Callable) -> Dict[str, Callable[[jax.Array], jax.Array]]:
    return {
        "reward": jax.grad(lambda theta_batch: objective_fn(theta_batch)[0]),
        "mean_cd_second_half": jax.grad(
            lambda theta_batch: jnp.sum(objective_fn(theta_batch)[1]["mean_cd_second_half"])
        ),
        "mean_cl_second_half": jax.grad(
            lambda theta_batch: jnp.sum(objective_fn(theta_batch)[1]["mean_cl_second_half"])
        ),
    }


def diagnose_backward_terms(
    theta_batch: jax.Array,
    grad_fns: Dict[str, Callable[[jax.Array], jax.Array]],
) -> Dict[str, Any]:
    batch_size = int(theta_batch.shape[0])
    diagnostics: Dict[str, Any] = {
        "first_nonfinite_term": None,
        "first_nonfinite_term_by_candidate": ["none"] * batch_size,
        "term_nonfinite_counts": {},
        "term_nonfinite_counts_by_candidate": {},
        "term_errors": {},
    }
    order = ["reward", "mean_cd_second_half", "mean_cl_second_half"]
    for term in order:
        fn = grad_fns[term]
        try:
            grad_term = fn(theta_batch)
            grad_term_np = np.asarray(grad_term)
            counts_by_candidate = np.sum(
                ~np.isfinite(grad_term_np.reshape((batch_size, -1))),
                axis=1,
            ).astype(np.int64)
            count = int(np.sum(counts_by_candidate))
            diagnostics["term_nonfinite_counts"][term] = count
            diagnostics["term_nonfinite_counts_by_candidate"][term] = counts_by_candidate.tolist()
            for cand in range(batch_size):
                if diagnostics["first_nonfinite_term_by_candidate"][cand] == "none" and counts_by_candidate[cand] > 0:
                    diagnostics["first_nonfinite_term_by_candidate"][cand] = term
            if diagnostics["first_nonfinite_term"] is None and count > 0:
                diagnostics["first_nonfinite_term"] = term
        except Exception as exc:  # pragma: no cover - runtime diagnostics path
            diagnostics["term_nonfinite_counts"][term] = None
            diagnostics["term_nonfinite_counts_by_candidate"][term] = [None] * batch_size
            diagnostics["term_errors"][term] = f"{type(exc).__name__}: {exc}"
            for cand in range(batch_size):
                if diagnostics["first_nonfinite_term_by_candidate"][cand] == "none":
                    diagnostics["first_nonfinite_term_by_candidate"][cand] = term
            if diagnostics["first_nonfinite_term"] is None:
                diagnostics["first_nonfinite_term"] = term
    if diagnostics["first_nonfinite_term"] is None:
        diagnostics["first_nonfinite_term"] = "none"
    return diagnostics


def estimate_fd_grad_batch(
    theta_batch: jax.Array,
    nonfinite_candidate_mask: np.ndarray,
    objective_value_fn: Callable[[jax.Array], jax.Array],
    r_min: float,
    r_max: float,
    eps: float,
    mode: str,
    spsa_samples: int,
    rng: np.random.Generator,
) -> Tuple[jax.Array, Dict[str, Any]]:
    grad_batch = jnp.zeros_like(theta_batch)
    eval_count = 0
    bad_eval_count = 0
    eval_count_by_candidate = np.zeros((theta_batch.shape[0],), dtype=np.int64)
    bad_eval_count_by_candidate = np.zeros((theta_batch.shape[0],), dtype=np.int64)
    used_candidates: List[int] = []

    candidate_indices = [int(i) for i in np.where(nonfinite_candidate_mask)[0].tolist()]
    for cand in candidate_indices:
        used_candidates.append(cand)
        theta_cand = theta_batch[cand]
        grad_cand = jnp.zeros_like(theta_cand)

        if mode == "spsa":
            if spsa_samples < 1:
                raise ValueError("spsa_samples must be >= 1")
            for _ in range(spsa_samples):
                delta_np = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=theta_cand.shape)
                delta = jnp.asarray(delta_np, dtype=theta_cand.dtype)
                theta_plus_cand = clamp_points_to_ring_jax(theta_cand + eps * delta, r_min, r_max)
                theta_minus_cand = clamp_points_to_ring_jax(theta_cand - eps * delta, r_min, r_max)
                theta_plus = theta_batch.at[cand].set(theta_plus_cand)
                theta_minus = theta_batch.at[cand].set(theta_minus_cand)
                f_plus = objective_value_fn(theta_plus)
                f_minus = objective_value_fn(theta_minus)
                eval_count += 2
                eval_count_by_candidate[cand] += 2
                if not (jnp.isfinite(f_plus) and jnp.isfinite(f_minus)):
                    bad_eval_count += 1
                    bad_eval_count_by_candidate[cand] += 1
                    continue
                slope = (f_plus - f_minus) / (2.0 * eps)
                grad_cand = grad_cand + slope * delta
            grad_cand = grad_cand / max(spsa_samples, 1)

        elif mode == "coordinate":
            for i in range(theta_cand.shape[0]):
                for j in range(theta_cand.shape[1]):
                    theta_plus_cand = clamp_points_to_ring_jax(
                        theta_cand.at[i, j].add(eps), r_min, r_max
                    )
                    theta_minus_cand = clamp_points_to_ring_jax(
                        theta_cand.at[i, j].add(-eps), r_min, r_max
                    )
                    theta_plus = theta_batch.at[cand].set(theta_plus_cand)
                    theta_minus = theta_batch.at[cand].set(theta_minus_cand)
                    f_plus = objective_value_fn(theta_plus)
                    f_minus = objective_value_fn(theta_minus)
                    eval_count += 2
                    eval_count_by_candidate[cand] += 2
                    if not (jnp.isfinite(f_plus) and jnp.isfinite(f_minus)):
                        bad_eval_count += 1
                        bad_eval_count_by_candidate[cand] += 1
                        continue
                    grad_cand = grad_cand.at[i, j].set((f_plus - f_minus) / (2.0 * eps))
        else:
            raise ValueError(f"Unknown FD mode: {mode}")

        grad_batch = grad_batch.at[cand].set(grad_cand)

    info = {
        "fd_used_candidates": used_candidates,
        "fd_eval_count": eval_count,
        "fd_bad_eval_count": bad_eval_count,
        "fd_eval_count_by_candidate": eval_count_by_candidate.tolist(),
        "fd_bad_eval_count_by_candidate": bad_eval_count_by_candidate.tolist(),
        "fd_mode": mode,
        "fd_eps": eps,
        "fd_spsa_samples": spsa_samples,
    }
    return grad_batch, info


def maybe_make_initial_theta_batch(
    parallel_sims: int,
    seed: int,
    theta_dtype: jnp.dtype,
    r_min: float,
    r_max: float,
    init_noise: float,
) -> jax.Array:
    base_theta = jnp.array(
        [
            [1.05, 0.00],
            [0.05, 1.00],
            [-1.00, 0.10],
            [0.00, -0.95],
        ],
        dtype=theta_dtype,
    )
    base_theta = clamp_points_to_ring_jax(base_theta, r_min, r_max)
    theta_batch = jnp.broadcast_to(base_theta[None, ...], (parallel_sims, 4, 2))
    if parallel_sims == 1 or init_noise <= 0.0:
        return theta_batch

    key = jax.random.PRNGKey(seed)
    noise = init_noise * jax.random.normal(key, shape=theta_batch.shape, dtype=theta_dtype)
    theta_batch = theta_batch + noise
    theta_batch = theta_batch.at[0].set(base_theta)
    theta_batch = jax.vmap(clamp_points_to_ring_jax, in_axes=(0, None, None))(theta_batch, r_min, r_max)
    return theta_batch


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


def build_objective_batched(
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
    def finite_mean(x: jax.Array, axis: int) -> Tuple[jax.Array, jax.Array]:
        finite = jnp.isfinite(x)
        valid_count = jnp.sum(finite, axis=axis)
        total = jnp.sum(jnp.where(finite, x, 0.0), axis=axis)
        mean = total / jnp.maximum(valid_count, 1)
        mean = jnp.where(valid_count > 0, mean, jnp.nan)
        return mean, valid_count

    def objective(theta_batch: jax.Array):
        theta_batch = jax.vmap(clamp_points_to_ring_jax, in_axes=(0, None, None))(theta_batch, r_min, r_max)

        def one_levelset(theta: jax.Array) -> jax.Array:
            contour = bezier_closed_from_four_points_jax(theta, bezier_samples, tension=1.0)
            return signed_distance_from_contour_jax(contour, theta, x_cells, y_cells)

        levelset_xy_batch = jax.vmap(one_levelset, in_axes=0, out_axes=0)(theta_batch)
        levelset_batch = levelset_xy_batch[..., None]
        batch_size = theta_batch.shape[0]
        batch_primes_init = jnp.broadcast_to(primes_init[None, ...], (batch_size,) + primes_init.shape)
        dt_vec = jnp.full((batch_size,), dt)
        t0_vec = jnp.full((batch_size,), t0)

        solution, times = sim_manager.feed_forward(
            batch_primes_init=batch_primes_init,
            physical_timestep_size=dt_vec,
            t_start=t0_vec,
            feed_forward_setup=feed_forward_setup,
            batch_levelset_init=levelset_batch,
        )

        primitives_batch = solution["primitives"]  # (B, T, 5, Nx, Ny, Nz)
        levelset_single_batch = solution["levelset"][:, :, :, 0]  # (B, Nx, Ny)

        cd, cl = jax.vmap(
            compute_cd_cl_series,
            in_axes=(0, 0, None, None, None),
            out_axes=(0, 0),
        )(
            primitives_batch,
            levelset_single_batch,
            x_cells,
            y_cells,
            0.01,
        )

        ratio = cl / (jnp.abs(cd) + 1.0e-12)  # (B, T)
        half_idx = ratio.shape[1] // 2
        ratio_second_half = ratio[:, half_idx:]
        cd_second_half = cd[:, half_idx:]
        cl_second_half = cl[:, half_idx:]

        rewards, ratio_valid_count = finite_mean(ratio_second_half, axis=1)
        mean_cd_second_half, cd_valid_count = finite_mean(cd_second_half, axis=1)
        mean_cl_second_half, cl_valid_count = finite_mean(cl_second_half, axis=1)
        rewards_for_objective = jnp.where(jnp.isfinite(rewards), rewards, 0.0)
        reward = jnp.sum(rewards_for_objective)

        primitives_nonfinite_count = jnp.sum(~jnp.isfinite(primitives_batch), axis=(1, 2, 3, 4, 5))
        cd_nonfinite_count = jnp.sum(~jnp.isfinite(cd), axis=1)
        cl_nonfinite_count = jnp.sum(~jnp.isfinite(cl), axis=1)
        max_abs_primitives = jnp.max(
            jnp.nan_to_num(jnp.abs(primitives_batch), nan=0.0, posinf=1.0e30, neginf=1.0e30),
            axis=(1, 2, 3, 4, 5),
        )

        aux = {
            "reward": reward,
            "rewards": rewards,
            "mean_cd_second_half": mean_cd_second_half,
            "mean_cl_second_half": mean_cl_second_half,
            "ratio_valid_count_second_half": ratio_valid_count,
            "cd_valid_count_second_half": cd_valid_count,
            "cl_valid_count_second_half": cl_valid_count,
            "ratio_total_count_second_half": jnp.full_like(ratio_valid_count, ratio_second_half.shape[1]),
            "cd_nonfinite_count": cd_nonfinite_count,
            "cl_nonfinite_count": cl_nonfinite_count,
            "primitives_nonfinite_count": primitives_nonfinite_count,
            "max_abs_primitives": max_abs_primitives,
            "end_time": times[:, -1],
            "levelset_xy_batch": levelset_xy_batch,
            "theta_batch": theta_batch,
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
    parser.add_argument("--parallel-sims", type=int, default=1, help="Number of simulations to run in parallel.")
    parser.add_argument("--init-noise", type=float, default=0.05, help="Init noise for extra parallel candidates.")
    parser.add_argument("--inner-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-step-norm", type=float, default=0.05)
    parser.add_argument("--ring-r-min", type=float, default=0.7)
    parser.add_argument("--ring-r-max", type=float, default=1.3)
    parser.add_argument("--bezier-samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jit", action="store_true", help="JIT compile value+grad function.")
    parser.add_argument("--solver-precision", choices=["from-setup", "single", "double"], default="from-setup")
    parser.add_argument("--theta-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--checkpoint-integration-step", action="store_true")
    parser.add_argument("--checkpoint-inner-step", dest="checkpoint_inner_step", action="store_true")
    parser.add_argument("--no-checkpoint-inner-step", dest="checkpoint_inner_step", action="store_false")
    parser.set_defaults(checkpoint_inner_step=True)
    parser.add_argument("--save-every", type=int, default=1, help="Save shape artifacts every N iterations.")
    parser.add_argument("--skip-shape-artifacts", action="store_true", help="Skip PNG/H5 shape artifacts for speed.")
    parser.add_argument(
        "--fail-on-nonfinite-grad",
        action="store_true",
        help="Fail fast when non-finite gradients are detected and unresolved.",
    )
    parser.add_argument(
        "--enable-fd-fallback",
        action="store_true",
        help="Use finite-difference fallback when AD gradients are non-finite.",
    )
    parser.add_argument(
        "--disable-fd-fallback",
        dest="enable_fd_fallback",
        action="store_false",
        help="Disable finite-difference fallback.",
    )
    parser.set_defaults(enable_fd_fallback=True)
    parser.add_argument("--fd-eps", type=float, default=1.0e-3)
    parser.add_argument("--fd-mode", choices=["spsa", "coordinate"], default="spsa")
    parser.add_argument("--fd-spsa-samples", type=int, default=2)
    parser.add_argument(
        "--log-backward-diagnostics",
        action="store_true",
        help="Log which backward term first goes non-finite (reward/Cd/Cl).",
    )
    parser.add_argument(
        "--no-log-backward-diagnostics",
        dest="log_backward_diagnostics",
        action="store_false",
        help="Disable backward non-finite term diagnostics.",
    )
    parser.set_defaults(log_backward_diagnostics=False)
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="jaxfluids-viq-shape-opt")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-tags", default="viq,shape,opt,differentiable")
    args = parser.parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if args.num_iters < 1:
        raise ValueError("--num-iters must be >= 1")
    if args.parallel_sims < 1:
        raise ValueError("--parallel-sims must be >= 1")
    if args.fd_eps <= 0.0:
        raise ValueError("--fd-eps must be > 0")
    if args.fd_spsa_samples < 1:
        raise ValueError("--fd-spsa-samples must be >= 1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "iter_shapes").mkdir(parents=True, exist_ok=True)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    case_dict = json.loads(Path(args.case_setup).read_text())
    numerical_dict = json.loads(Path(args.numerical_setup).read_text())
    numerical_dict = maybe_apply_solver_precision_override(numerical_dict, args.solver_precision)
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
        is_checkpoint_inner_step=args.checkpoint_inner_step,
        is_checkpoint_integration_step=args.checkpoint_integration_step,
        is_include_t0=True,
        is_include_halos=False,
    )

    objective = build_objective_batched(
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
    objective_value_fn = build_objective_value_fn(objective, do_jit=args.jit)
    backward_diag_grad_fns = build_backward_diagnostic_grad_fns(objective) if args.log_backward_diagnostics else None

    theta_dtype = jnp.float32 if args.theta_dtype == "float32" else jnp.float64
    theta_batch = maybe_make_initial_theta_batch(
        parallel_sims=args.parallel_sims,
        seed=args.seed,
        theta_dtype=theta_dtype,
        r_min=args.ring_r_min,
        r_max=args.ring_r_max,
        init_noise=args.init_noise,
    )

    run_config = {
        "case_setup": str(Path(args.case_setup).resolve()),
        "numerical_setup": str(Path(args.numerical_setup).resolve()),
        "sim_time": args.sim_time,
        "dt_used_in_feed_forward": dt,
        "total_steps": total_steps,
        "outer_steps": outer_steps,
        "inner_steps": args.inner_steps,
        "num_iters": args.num_iters,
        "parallel_sims": args.parallel_sims,
        "init_noise": args.init_noise,
        "learning_rate": args.learning_rate,
        "max_step_norm": args.max_step_norm,
        "solver_precision": args.solver_precision,
        "theta_dtype": args.theta_dtype,
        "checkpoint_inner_step": args.checkpoint_inner_step,
        "checkpoint_integration_step": args.checkpoint_integration_step,
        "save_every": args.save_every,
        "skip_shape_artifacts": args.skip_shape_artifacts,
        "fail_on_nonfinite_grad": args.fail_on_nonfinite_grad,
        "enable_fd_fallback": args.enable_fd_fallback,
        "fd_eps": args.fd_eps,
        "fd_mode": args.fd_mode,
        "fd_spsa_samples": args.fd_spsa_samples,
        "log_backward_diagnostics": args.log_backward_diagnostics,
        "ring_r_min": args.ring_r_min,
        "ring_r_max": args.ring_r_max,
        "bezier_samples": args.bezier_samples,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
        "wandb_mode": args.wandb_mode,
        "wandb_tags": args.wandb_tags,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    wandb_run = maybe_init_wandb(args, run_config, out_dir)

    history_csv = out_dir / "history.csv"
    with history_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iter",
                "candidate",
                "reward",
                "reward_raw",
                "mean_cd_second_half",
                "mean_cd_second_half_raw",
                "mean_cl_second_half",
                "mean_cl_second_half_raw",
                "reward_is_finite",
                "cd_is_finite",
                "cl_is_finite",
                "ratio_valid_fraction_second_half",
                "primitives_nonfinite_count",
                "cd_nonfinite_count",
                "cl_nonfinite_count",
                "max_abs_primitives",
                "grad_raw_nonfinite_count",
                "grad_raw_norm_before_sanitize",
                "grad_unresolved_nonfinite_count",
                "grad_source",
                "fd_fallback_used",
                "fd_eval_count",
                "fd_bad_eval_count",
                "first_nonfinite_backward_term",
                "backward_reward_nonfinite_count",
                "backward_cd_nonfinite_count",
                "backward_cl_nonfinite_count",
                "grad_norm",
                "step_norm",
                "wall_seconds",
                "sim_end_time",
                "cp0_x",
                "cp0_y",
                "cp1_x",
                "cp1_y",
                "cp2_x",
                "cp2_y",
                "cp3_x",
                "cp3_y",
            ]
        )

    history = []
    for it in range(args.num_iters):
        t_start_iter = time.time()
        (reward_total_raw, aux_raw), grad_raw = value_and_grad(theta_batch)
        grad_ad_raw = grad_raw
        grad_ad_raw_np = np.asarray(grad_ad_raw)
        grad_ad_raw_nonfinite_counts_np = np.sum(
            ~np.isfinite(grad_ad_raw_np.reshape((args.parallel_sims, -1))),
            axis=1,
        )
        nonfinite_grad_mask_np = grad_ad_raw_nonfinite_counts_np > 0

        backward_diag: Dict[str, Any] | None = None
        backward_first_terms: List[str] = ["none"] * args.parallel_sims
        backward_reward_nonfinite_counts: List[Any] = [0] * args.parallel_sims
        backward_cd_nonfinite_counts: List[Any] = [0] * args.parallel_sims
        backward_cl_nonfinite_counts: List[Any] = [0] * args.parallel_sims
        if args.log_backward_diagnostics and backward_diag_grad_fns is not None and np.any(nonfinite_grad_mask_np):
            backward_diag = diagnose_backward_terms(theta_batch, backward_diag_grad_fns)
            backward_first_terms = list(backward_diag.get("first_nonfinite_term_by_candidate", backward_first_terms))
            term_counts_by_candidate = backward_diag.get("term_nonfinite_counts_by_candidate", {})
            backward_reward_nonfinite_counts = term_counts_by_candidate.get(
                "reward", backward_reward_nonfinite_counts
            )
            backward_cd_nonfinite_counts = term_counts_by_candidate.get(
                "mean_cd_second_half", backward_cd_nonfinite_counts
            )
            backward_cl_nonfinite_counts = term_counts_by_candidate.get(
                "mean_cl_second_half", backward_cl_nonfinite_counts
            )

        fd_info: Dict[str, Any] = {
            "fd_used_candidates": [],
            "fd_eval_count": 0,
            "fd_bad_eval_count": 0,
            "fd_eval_count_by_candidate": [0] * args.parallel_sims,
            "fd_bad_eval_count_by_candidate": [0] * args.parallel_sims,
            "fd_mode": args.fd_mode,
            "fd_eps": args.fd_eps,
            "fd_spsa_samples": args.fd_spsa_samples,
        }
        grad_effective_raw = grad_ad_raw
        if np.any(nonfinite_grad_mask_np) and args.enable_fd_fallback:
            rng = np.random.default_rng(args.seed + it)
            fd_grad_batch, fd_info = estimate_fd_grad_batch(
                theta_batch=theta_batch,
                nonfinite_candidate_mask=nonfinite_grad_mask_np,
                objective_value_fn=objective_value_fn,
                r_min=args.ring_r_min,
                r_max=args.ring_r_max,
                eps=args.fd_eps,
                mode=args.fd_mode,
                spsa_samples=args.fd_spsa_samples,
                rng=rng,
            )
            fallback_mask = jnp.asarray(nonfinite_grad_mask_np)[:, None, None]
            grad_effective_raw = jnp.where(fallback_mask, fd_grad_batch, grad_ad_raw)

        grad_effective_raw_np = np.asarray(grad_effective_raw)
        grad_effective_raw_nonfinite_counts_np = np.sum(
            ~np.isfinite(grad_effective_raw_np.reshape((args.parallel_sims, -1))),
            axis=1,
        )
        unresolved_grad_mask_np = grad_effective_raw_nonfinite_counts_np > 0
        if args.fail_on_nonfinite_grad and np.any(unresolved_grad_mask_np):
            bad_candidates = [int(i) for i in np.where(unresolved_grad_mask_np)[0].tolist()]
            first_terms = {int(i): backward_first_terms[int(i)] for i in bad_candidates}
            raise RuntimeError(
                "Non-finite gradient detected after fallback; "
                f"candidates={bad_candidates}, first_nonfinite_backward_term={first_terms}, "
                f"fd_info={fd_info}"
            )

        grad_sources = ["ad"] * args.parallel_sims
        fd_used_set = set(int(c) for c in fd_info.get("fd_used_candidates", []))
        for cand in range(args.parallel_sims):
            if cand in fd_used_set:
                grad_sources[cand] = "fd" if not unresolved_grad_mask_np[cand] else "fd_nonfinite"
            elif nonfinite_grad_mask_np[cand]:
                grad_sources[cand] = "ad_nonfinite"

        reward_total = jnp.nan_to_num(reward_total_raw, nan=-1.0e6, posinf=1.0e6, neginf=-1.0e6)
        aux = {
            "rewards": jnp.nan_to_num(aux_raw["rewards"], nan=-1.0e6, posinf=1.0e6, neginf=-1.0e6),
            "mean_cd_second_half": jnp.nan_to_num(aux_raw["mean_cd_second_half"], nan=1.0e6, posinf=1.0e6, neginf=-1.0e6),
            "mean_cl_second_half": jnp.nan_to_num(aux_raw["mean_cl_second_half"], nan=0.0, posinf=1.0e6, neginf=-1.0e6),
            "ratio_valid_count_second_half": aux_raw["ratio_valid_count_second_half"],
            "ratio_total_count_second_half": aux_raw["ratio_total_count_second_half"],
            "cd_nonfinite_count": aux_raw["cd_nonfinite_count"],
            "cl_nonfinite_count": aux_raw["cl_nonfinite_count"],
            "primitives_nonfinite_count": aux_raw["primitives_nonfinite_count"],
            "max_abs_primitives": jnp.nan_to_num(aux_raw["max_abs_primitives"], nan=1.0e30, posinf=1.0e30, neginf=1.0e30),
            "end_time": jnp.nan_to_num(aux_raw["end_time"], nan=0.0, posinf=0.0, neginf=0.0),
            "levelset_xy_batch": jnp.nan_to_num(aux_raw["levelset_xy_batch"], nan=0.0, posinf=0.0, neginf=0.0),
            "theta_batch": jnp.nan_to_num(aux_raw["theta_batch"], nan=0.0, posinf=0.0, neginf=0.0),
        }
        grad_batch = jnp.nan_to_num(grad_effective_raw, nan=0.0, posinf=0.0, neginf=0.0)

        step_batch = args.learning_rate * grad_batch
        step_norms = jnp.linalg.norm(step_batch.reshape((args.parallel_sims, -1)), axis=1)
        scales = jnp.where(
            step_norms > args.max_step_norm,
            args.max_step_norm / (step_norms + 1.0e-14),
            1.0,
        )
        step_batch = step_batch * scales[:, None, None]
        step_norms = jnp.linalg.norm(step_batch.reshape((args.parallel_sims, -1)), axis=1)
        theta_batch = jax.vmap(clamp_points_to_ring_jax, in_axes=(0, None, None))(
            theta_batch + step_batch, args.ring_r_min, args.ring_r_max
        )
        elapsed = time.time() - t_start_iter

        iter_dir = out_dir / "iter_shapes" / f"iter_{it:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        theta_np_batch = np.asarray(theta_batch)
        rewards_np = np.asarray(aux["rewards"])
        rewards_raw_np = np.asarray(aux_raw["rewards"])
        cd_np = np.asarray(aux["mean_cd_second_half"])
        cd_raw_np = np.asarray(aux_raw["mean_cd_second_half"])
        cl_np = np.asarray(aux["mean_cl_second_half"])
        cl_raw_np = np.asarray(aux_raw["mean_cl_second_half"])
        ratio_valid_count_np = np.asarray(aux["ratio_valid_count_second_half"])
        ratio_total_count_np = np.asarray(aux["ratio_total_count_second_half"])
        cd_nonfinite_count_np = np.asarray(aux["cd_nonfinite_count"])
        cl_nonfinite_count_np = np.asarray(aux["cl_nonfinite_count"])
        primitives_nonfinite_count_np = np.asarray(aux["primitives_nonfinite_count"])
        max_abs_primitives_np = np.asarray(aux["max_abs_primitives"])
        end_time_np = np.asarray(aux["end_time"])
        grad_norms_np = np.asarray(jnp.linalg.norm(grad_batch.reshape((args.parallel_sims, -1)), axis=1))
        grad_raw_nonfinite_counts_np = grad_ad_raw_nonfinite_counts_np
        grad_unresolved_nonfinite_counts_np = grad_effective_raw_nonfinite_counts_np
        grad_raw_norms_np = np.linalg.norm(
            np.nan_to_num(
                grad_effective_raw_np.reshape((args.parallel_sims, -1)),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            axis=1,
        )
        step_norms_np = np.asarray(step_norms)

        should_save_shape = (it % args.save_every == 0) or (it == args.num_iters - 1)
        levelset_np_batch = None
        if should_save_shape and not args.skip_shape_artifacts:
            levelset_np_batch = np.asarray(aux["levelset_xy_batch"])
        iter_metrics = []
        for cand in range(args.parallel_sims):
            cand_dir = iter_dir / f"cand_{cand:03d}"
            cand_dir.mkdir(parents=True, exist_ok=True)

            if should_save_shape and not args.skip_shape_artifacts:
                write_levelset_h5(levelset_np_batch[cand], cand_dir / "levelset.h5")
                contour_np = np.asarray(
                    bezier_closed_from_four_points_jax(jnp.asarray(theta_np_batch[cand]), args.bezier_samples, tension=1.0)
                )
                render_shape_preview(contour_np, theta_np_batch[cand], cand_dir / "shape.png")

            row = {
                "iter": it,
                "candidate": cand,
                "reward": float(rewards_np[cand]),
                "reward_raw": finite_float_or_none(rewards_raw_np[cand]),
                "mean_cd_second_half": float(cd_np[cand]),
                "mean_cd_second_half_raw": finite_float_or_none(cd_raw_np[cand]),
                "mean_cl_second_half": float(cl_np[cand]),
                "mean_cl_second_half_raw": finite_float_or_none(cl_raw_np[cand]),
                "reward_is_finite": bool(np.isfinite(rewards_raw_np[cand])),
                "cd_is_finite": bool(np.isfinite(cd_raw_np[cand])),
                "cl_is_finite": bool(np.isfinite(cl_raw_np[cand])),
                "ratio_valid_fraction_second_half": float(
                    ratio_valid_count_np[cand] / max(float(ratio_total_count_np[cand]), 1.0)
                ),
                "primitives_nonfinite_count": int(primitives_nonfinite_count_np[cand]),
                "cd_nonfinite_count": int(cd_nonfinite_count_np[cand]),
                "cl_nonfinite_count": int(cl_nonfinite_count_np[cand]),
                "max_abs_primitives": float(max_abs_primitives_np[cand]),
                "grad_raw_nonfinite_count": int(grad_raw_nonfinite_counts_np[cand]),
                "grad_raw_norm_before_sanitize": float(grad_raw_norms_np[cand]),
                "grad_unresolved_nonfinite_count": int(grad_unresolved_nonfinite_counts_np[cand]),
                "grad_source": grad_sources[cand],
                "fd_fallback_used": bool(cand in fd_used_set),
                "fd_eval_count": int(fd_info["fd_eval_count_by_candidate"][cand]),
                "fd_bad_eval_count": int(fd_info["fd_bad_eval_count_by_candidate"][cand]),
                "first_nonfinite_backward_term": str(backward_first_terms[cand]),
                "backward_reward_nonfinite_count": (
                    int(backward_reward_nonfinite_counts[cand])
                    if backward_reward_nonfinite_counts[cand] is not None
                    else None
                ),
                "backward_cd_nonfinite_count": (
                    int(backward_cd_nonfinite_counts[cand]) if backward_cd_nonfinite_counts[cand] is not None else None
                ),
                "backward_cl_nonfinite_count": (
                    int(backward_cl_nonfinite_counts[cand]) if backward_cl_nonfinite_counts[cand] is not None else None
                ),
                "grad_norm": float(grad_norms_np[cand]),
                "step_norm": float(step_norms_np[cand]),
                "wall_seconds": elapsed,
                "sim_end_time": float(end_time_np[cand]),
                "theta": theta_np_batch[cand].tolist(),
            }
            history.append(row)
            iter_metrics.append(row)
            with history_csv.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        row["iter"],
                        row["candidate"],
                        row["reward"],
                        row["reward_raw"],
                        row["mean_cd_second_half"],
                        row["mean_cd_second_half_raw"],
                        row["mean_cl_second_half"],
                        row["mean_cl_second_half_raw"],
                        row["reward_is_finite"],
                        row["cd_is_finite"],
                        row["cl_is_finite"],
                        row["ratio_valid_fraction_second_half"],
                        row["primitives_nonfinite_count"],
                        row["cd_nonfinite_count"],
                        row["cl_nonfinite_count"],
                        row["max_abs_primitives"],
                        row["grad_raw_nonfinite_count"],
                        row["grad_raw_norm_before_sanitize"],
                        row["grad_unresolved_nonfinite_count"],
                        row["grad_source"],
                        row["fd_fallback_used"],
                        row["fd_eval_count"],
                        row["fd_bad_eval_count"],
                        row["first_nonfinite_backward_term"],
                        row["backward_reward_nonfinite_count"],
                        row["backward_cd_nonfinite_count"],
                        row["backward_cl_nonfinite_count"],
                        row["grad_norm"],
                        row["step_norm"],
                        row["wall_seconds"],
                        row["sim_end_time"],
                        row["theta"][0][0],
                        row["theta"][0][1],
                        row["theta"][1][0],
                        row["theta"][1][1],
                        row["theta"][2][0],
                        row["theta"][2][1],
                        row["theta"][3][0],
                        row["theta"][3][1],
                    ]
                )
            (cand_dir / "metrics.json").write_text(json.dumps(row, indent=2))

        (iter_dir / "metrics_all_candidates.json").write_text(json.dumps(iter_metrics, indent=2))
        if wandb_run is not None:
            wb_payload: Dict[str, float] = {
                "reward_total": float(reward_total),
                "reward_mean": float(np.mean(rewards_np)),
                "reward_best": float(np.max(rewards_np)),
                "drag_mean": float(np.mean(cd_np)),
                "lift_mean": float(np.mean(cl_np)),
                "nonfinite_grad_candidates": float(np.sum(nonfinite_grad_mask_np)),
                "fd_fallback_candidates": float(len(fd_used_set)),
                "unresolved_nonfinite_grad_candidates": float(np.sum(unresolved_grad_mask_np)),
            }
            for cand in range(args.parallel_sims):
                wb_payload[f"cand_{cand}/reward"] = float(rewards_np[cand])
                wb_payload[f"cand_{cand}/drag"] = float(cd_np[cand])
                wb_payload[f"cand_{cand}/lift"] = float(cl_np[cand])
                for i in range(4):
                    wb_payload[f"cand_{cand}/control_point_{i}_x"] = float(theta_np_batch[cand, i, 0])
                    wb_payload[f"cand_{cand}/control_point_{i}_y"] = float(theta_np_batch[cand, i, 1])
            wandb_run.log(wb_payload, step=it)

        best_idx = int(np.argmax(rewards_np))
        invalid_count = int(np.sum(~np.isfinite(rewards_raw_np)))
        print(
            f"[iter {it:03d}] sims={args.parallel_sims} "
            f"reward_mean={np.mean(rewards_np):.6f} reward_best={np.max(rewards_np):.6f} "
            f"(cand={best_idx:03d}) invalid_rewards={invalid_count}/{args.parallel_sims} t={elapsed:.2f}s"
        )

    rewards_last = np.asarray([h["reward"] for h in history[-args.parallel_sims:]], dtype=np.float64)
    best_final_idx = int(np.argmax(rewards_last))
    best_global = history[-args.parallel_sims + best_final_idx]
    final = {
        "parallel_sims": args.parallel_sims,
        "final_theta_batch": np.asarray(theta_batch).tolist(),
        "best_final_candidate": best_final_idx,
        "best_final_reward": float(best_global["reward"]),
        "best_final_theta": best_global["theta"],
        "history": history,
    }
    if args.parallel_sims == 1:
        final["final_theta"] = final["final_theta_batch"][0]
    (out_dir / "final_summary.json").write_text(json.dumps(final, indent=2))
    if wandb_run is not None:
        wandb_run.summary["final_theta_batch"] = final["final_theta_batch"]
        wandb_run.summary["best_final_candidate"] = best_final_idx
        wandb_run.summary["best_final_reward"] = float(best_global["reward"])
        wandb_run.summary["num_iters"] = args.num_iters
        wandb_run.summary["history_rows"] = len(history)
        if history:
            wandb_run.summary["final_reward"] = float(best_global["reward"])
            wandb_run.summary["final_drag"] = float(best_global["mean_cd_second_half"])
            wandb_run.summary["final_lift"] = float(best_global["mean_cl_second_half"])
        wandb_run.finish()
    print(f"Saved optimization outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
