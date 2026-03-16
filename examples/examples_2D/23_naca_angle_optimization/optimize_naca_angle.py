from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import h5py
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jaxfluids import InitializationManager, InputManager, SimulationManager
from jaxfluids.feed_forward.data_types import FeedForwardSetup
from jaxfluids.levelset.creation.NACA_helper_functions import (
    five_digit_airfoil,
    four_digit_airfoil,
    thickness_distribution,
)
from jaxfluids_postprocess import create_xdmf_from_h5, load_data


NACA_DIGIT_FUNCTIONS: Dict[int, Callable[[str, jax.Array], Tuple[jax.Array, jax.Array]]] = {
    4: four_digit_airfoil,
    5: five_digit_airfoil,
}

HISTORY_FIELDS = [
    "iter",
    "angle_deg",
    "reward",
    "mean_cd_second_half",
    "mean_cl_second_half",
    "grad",
    "step_deg",
    "reward_is_finite",
    "grad_is_finite",
    "grad_source",
    "fd_eval_count",
    "reward_valid_fraction_second_half",
    "primitives_nonfinite_count",
    "cd_nonfinite_count",
    "cl_nonfinite_count",
    "max_abs_primitives",
    "wall_seconds",
    "sim_end_time",
]


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def maybe_apply_solver_precision_override(
    numerical_setup: Dict[str, Any],
    solver_precision: str,
) -> Dict[str, Any]:
    if solver_precision == "from-setup":
        return numerical_setup
    overridden = copy.deepcopy(numerical_setup)
    precision = overridden.setdefault("precision", {})
    if solver_precision == "single":
        precision["is_double_precision_compute"] = False
        precision["is_double_precision_output"] = False
    elif solver_precision == "double":
        precision["is_double_precision_compute"] = True
        precision["is_double_precision_output"] = True
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unsupported solver precision: {solver_precision}")
    return overridden


def prepare_feedforward_compatible_numerical_setup(numerical_setup: Dict[str, Any]) -> Dict[str, Any]:
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
    num.setdefault("output", {})["is_xdmf"] = False
    num.setdefault("output", {}).setdefault("logging", {})["frequency"] = 1000
    return num


def extract_reference_values(case_dict: Dict[str, Any], chord: float) -> Dict[str, float]:
    primitives = case_dict["initial_condition"]["primitives"]
    dynamic_viscosity = case_dict["material_properties"]["transport"]["dynamic_viscosity"]["value"]
    return {
        "rho_ref": float(primitives["rho"]),
        "u_ref": float(primitives["u"]),
        "dynamic_viscosity": float(dynamic_viscosity),
        "s_ref": float(chord),
        "reynolds_number": float(primitives["rho"]) * float(primitives["u"]) * float(chord) / float(dynamic_viscosity),
    }


def get_solver_dtype(sim_manager: SimulationManager) -> jnp.dtype:
    if sim_manager.numerical_setup.precision.is_double_precision_compute:
        return jnp.float64
    return jnp.float32


def write_levelset_h5(levelset_xy: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5file:
        h5file.create_dataset("levelset", data=np.asarray(levelset_xy, dtype=np.float64)[..., None])
    return out_path


def save_airfoil_preview(
    contour_xy: np.ndarray,
    angle_deg: float,
    out_path: Path,
    center: Tuple[float, float],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(contour_xy[:, 0], contour_xy[:, 1], color="#1f4e79", lw=2.2, label="airfoil contour")
    ax.scatter([center[0]], [center[1]], color="#d62728", s=36, label="rotation center")
    ax.axhline(0.0, color="#bbbbbb", lw=0.8)
    ax.axvline(0.0, color="#bbbbbb", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"NACA geometry at {angle_deg:.3f} deg")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_airfoil_overlay(
    contour_local: jax.Array,
    initial_angle_rad: float,
    best_angle_rad: float,
    center: Tuple[float, float],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    initial_contour = np.asarray(rotate_translate(contour_local, initial_angle_rad, center[0], center[1]))
    best_contour = np.asarray(rotate_translate(contour_local, best_angle_rad, center[0], center[1]))

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(initial_contour[:, 0], initial_contour[:, 1], color="#7f8c8d", lw=2.0, label="initial")
    ax.plot(best_contour[:, 0], best_contour[:, 1], color="#c0392b", lw=2.0, label="optimized")
    ax.scatter([center[0]], [center[1]], color="#1f1f1f", s=28, label="rotation center")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Initial vs optimized airfoil orientation")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_naca_contour_local(
    digit_code: str,
    chord: float,
    samples_per_side: int,
    pivot_fraction: float,
    dtype: jnp.dtype,
) -> jax.Array:
    if len(digit_code) not in NACA_DIGIT_FUNCTIONS:
        raise ValueError(f"NACA code '{digit_code}' must have 4 or 5 digits.")

    beta = jnp.linspace(0.0, jnp.pi, samples_per_side, dtype=dtype)
    chord_line = 0.5 * (1.0 - jnp.cos(beta))
    camber_line, theta = NACA_DIGIT_FUNCTIONS[len(digit_code)](digit_code, chord_line)
    thickness = jnp.asarray(int(digit_code[-2:]) / 100.0, dtype=dtype)
    y_t = thickness_distribution(thickness, chord_line)

    x_upper = (chord_line - y_t * jnp.sin(theta)) * chord
    y_upper = (camber_line + y_t * jnp.cos(theta)) * chord
    x_lower = (chord_line + y_t * jnp.sin(theta)) * chord
    y_lower = (camber_line - y_t * jnp.cos(theta)) * chord

    pivot_shift = jnp.asarray(chord * pivot_fraction, dtype=dtype)
    upper = jnp.stack([x_upper - pivot_shift, y_upper], axis=1)
    lower = jnp.stack([x_lower - pivot_shift, y_lower], axis=1)

    contour = jnp.concatenate([upper[::-1], lower[1:]], axis=0)
    contour = jnp.concatenate([contour, contour[:1]], axis=0)
    return contour


def rotate_translate(points: jax.Array, angle_rad: jax.Array, center_x: float, center_y: float) -> jax.Array:
    cos_alpha = jnp.cos(angle_rad)
    sin_alpha = jnp.sin(angle_rad)
    x_rot = cos_alpha * points[:, 0] - sin_alpha * points[:, 1] + center_x
    y_rot = sin_alpha * points[:, 0] + cos_alpha * points[:, 1] + center_y
    return jnp.stack([x_rot, y_rot], axis=1)


def signed_distance_from_contour_jax(
    contour: jax.Array,
    x: jax.Array,
    y: jax.Array,
    softmin_temperature: float,
    inside_edge_sharpness: float,
    inside_score_sharpness: float,
) -> jax.Array:
    xx, yy = jnp.meshgrid(x, y, indexing="ij")

    p0 = contour[:-1]
    p1 = contour[1:]
    x0 = p0[:, 0]
    y0 = p0[:, 1]
    x1 = p1[:, 0]
    y1 = p1[:, 1]

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
    unsigned_distance = -softmin_temperature * jax.nn.logsumexp(-dist / softmin_temperature, axis=-1)

    signed_area = 0.5 * jnp.sum(x0 * y1 - x1 * y0)
    orientation = jnp.where(signed_area >= 0.0, 1.0, -1.0)
    cross = orientation * (vx * (yg - y0) - vy * (xg - x0))
    inside_each = jax.nn.sigmoid(inside_edge_sharpness * cross)
    inside_score = jnp.mean(inside_each, axis=-1)
    inside_prob = jax.nn.sigmoid(inside_score_sharpness * (inside_score - 0.5))
    sign = 1.0 - 2.0 * inside_prob
    return sign * unsigned_distance


def build_naca_levelset(
    angle_rad: jax.Array,
    contour_local: jax.Array,
    x_cells: jax.Array,
    y_cells: jax.Array,
    center_x: float,
    center_y: float,
    softmin_temperature: float,
    inside_edge_sharpness: float,
    inside_score_sharpness: float,
) -> Tuple[jax.Array, jax.Array]:
    contour = rotate_translate(contour_local, angle_rad, center_x, center_y)
    levelset = signed_distance_from_contour_jax(
        contour=contour,
        x=x_cells,
        y=y_cells,
        softmin_temperature=softmin_temperature,
        inside_edge_sharpness=inside_edge_sharpness,
        inside_score_sharpness=inside_score_sharpness,
    )
    return contour, levelset


def compute_cd_cl_series(
    primitives_t: jax.Array,
    levelset_xy: jax.Array,
    x_cells: jax.Array,
    y_cells: jax.Array,
    dynamic_viscosity: float,
    rho_ref: float,
    u_ref: float,
    s_ref: float,
) -> Tuple[jax.Array, jax.Array]:
    dphi_dx = jnp.gradient(levelset_xy, x_cells, axis=0)
    dphi_dy = jnp.gradient(levelset_xy, y_cells, axis=1)
    normal_norm = jnp.sqrt(dphi_dx**2 + dphi_dy**2) + 1.0e-14
    normal_x = dphi_dx / normal_norm
    normal_y = dphi_dy / normal_norm

    dx = jnp.abs(jnp.gradient(x_cells))
    dy = jnp.abs(jnp.gradient(y_cells))
    cell_area = dx[:, None] * dy[None, :]
    interface_eps = 1.5 * jnp.minimum(jnp.min(dx), jnp.min(dy))
    delta = jnp.exp(-(levelset_xy / interface_eps) ** 2) / (jnp.sqrt(jnp.pi) * interface_eps)
    surface_weight = delta * normal_norm

    def one_snapshot(primitives: jax.Array) -> Tuple[jax.Array, jax.Array]:
        u = primitives[1, :, :, 0]
        v = primitives[2, :, :, 0]
        p = primitives[4, :, :, 0]

        du_dx = jnp.gradient(u, x_cells, axis=0)
        du_dy = jnp.gradient(u, y_cells, axis=1)
        dv_dx = jnp.gradient(v, x_cells, axis=0)
        dv_dy = jnp.gradient(v, y_cells, axis=1)

        tau_xx = -p + 2.0 * dynamic_viscosity * du_dx
        tau_xy = dynamic_viscosity * (du_dy + dv_dx)
        tau_yy = -p + 2.0 * dynamic_viscosity * dv_dy

        traction_x = tau_xx * normal_x + tau_xy * normal_y
        traction_y = tau_xy * normal_x + tau_yy * normal_y

        force_x = jnp.sum(traction_x * surface_weight * cell_area)
        force_y = jnp.sum(traction_y * surface_weight * cell_area)
        return force_x, force_y

    force_x, force_y = jax.vmap(one_snapshot, in_axes=0, out_axes=0)(primitives_t)
    coeff_scale = 0.5 * rho_ref * u_ref**2 * s_ref
    return force_x / coeff_scale, force_y / coeff_scale


def finite_mean(values: jax.Array) -> Tuple[jax.Array, jax.Array]:
    finite = jnp.isfinite(values)
    valid_count = jnp.sum(finite)
    total = jnp.sum(jnp.where(finite, values, 0.0))
    mean = total / jnp.maximum(valid_count, 1)
    mean = jnp.where(valid_count > 0, mean, jnp.nan)
    return mean, valid_count


def build_objective(
    sim_manager: SimulationManager,
    primes_init: jax.Array,
    x_cells: jax.Array,
    y_cells: jax.Array,
    dt: float,
    t0: float,
    feed_forward_setup: FeedForwardSetup,
    contour_local: jax.Array,
    center_x: float,
    center_y: float,
    softmin_temperature: float,
    inside_edge_sharpness: float,
    inside_score_sharpness: float,
    angle_min_rad: float,
    angle_max_rad: float,
    reference_values: Dict[str, float],
) -> Callable[[jax.Array], Tuple[jax.Array, Dict[str, jax.Array]]]:
    batch_primes_init = primes_init[None, ...]
    dt_vec = jnp.asarray([dt], dtype=primes_init.dtype)
    t0_vec = jnp.asarray([t0], dtype=primes_init.dtype)

    def objective(angle_rad: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        angle_rad = jnp.clip(angle_rad, angle_min_rad, angle_max_rad)
        _, levelset_xy = build_naca_levelset(
            angle_rad=angle_rad,
            contour_local=contour_local,
            x_cells=x_cells,
            y_cells=y_cells,
            center_x=center_x,
            center_y=center_y,
            softmin_temperature=softmin_temperature,
            inside_edge_sharpness=inside_edge_sharpness,
            inside_score_sharpness=inside_score_sharpness,
        )
        levelset_batch = levelset_xy[None, ..., None]

        solution, times = sim_manager.feed_forward(
            batch_primes_init=batch_primes_init,
            physical_timestep_size=dt_vec,
            t_start=t0_vec,
            feed_forward_setup=feed_forward_setup,
            batch_levelset_init=levelset_batch,
        )

        primitives_t = solution["primitives"][0]
        static_levelset_xy = solution["levelset"][0, :, :, 0]
        times_1d = times[0]

        cd_series, cl_series = compute_cd_cl_series(
            primitives_t=primitives_t,
            levelset_xy=static_levelset_xy,
            x_cells=x_cells,
            y_cells=y_cells,
            dynamic_viscosity=reference_values["dynamic_viscosity"],
            rho_ref=reference_values["rho_ref"],
            u_ref=reference_values["u_ref"],
            s_ref=reference_values["s_ref"],
        )
        reward_series = cl_series / (jnp.abs(cd_series) + 1.0e-12)

        half_idx = reward_series.shape[0] // 2
        reward_second_half = reward_series[half_idx:]
        cd_second_half = cd_series[half_idx:]
        cl_second_half = cl_series[half_idx:]

        reward, reward_valid_count = finite_mean(reward_second_half)
        mean_cd_second_half, cd_valid_count = finite_mean(cd_second_half)
        mean_cl_second_half, cl_valid_count = finite_mean(cl_second_half)

        aux = {
            "angle_rad": angle_rad,
            "mean_cd_second_half": mean_cd_second_half,
            "mean_cl_second_half": mean_cl_second_half,
            "reward_valid_count_second_half": reward_valid_count,
            "cd_valid_count_second_half": cd_valid_count,
            "cl_valid_count_second_half": cl_valid_count,
            "reward_total_count_second_half": jnp.asarray(reward_second_half.shape[0]),
            "cd_nonfinite_count": jnp.sum(~jnp.isfinite(cd_series)),
            "cl_nonfinite_count": jnp.sum(~jnp.isfinite(cl_series)),
            "primitives_nonfinite_count": jnp.sum(~jnp.isfinite(primitives_t)),
            "max_abs_primitives": jnp.max(
                jnp.nan_to_num(jnp.abs(primitives_t), nan=0.0, posinf=1.0e30, neginf=1.0e30)
            ),
            "cd_series": cd_series,
            "cl_series": cl_series,
            "reward_series": reward_series,
            "times": times_1d,
            "end_time": times_1d[-1],
        }
        return reward, aux

    return objective


def objective_value_sanitized(objective_fn: Callable[[jax.Array], Tuple[jax.Array, Dict[str, jax.Array]]], angle_rad: float) -> float:
    reward_raw, _ = objective_fn(jnp.asarray(angle_rad))
    reward_np = float(np.asarray(reward_raw))
    if math.isfinite(reward_np):
        return reward_np
    if math.isnan(reward_np):
        return -1.0e6
    return math.copysign(1.0e6, reward_np)


def finite_difference_grad(
    objective_fn: Callable[[jax.Array], Tuple[jax.Array, Dict[str, jax.Array]]],
    angle_rad: float,
    angle_min_rad: float,
    angle_max_rad: float,
    fd_eps_rad: float,
) -> Tuple[float, Dict[str, Any]]:
    angle_plus = min(angle_rad + fd_eps_rad, angle_max_rad)
    angle_minus = max(angle_rad - fd_eps_rad, angle_min_rad)
    reward_plus = objective_value_sanitized(objective_fn, angle_plus)
    reward_minus = objective_value_sanitized(objective_fn, angle_minus)
    denom = angle_plus - angle_minus
    if abs(denom) < 1.0e-14:
        return math.nan, {"fd_eval_count": 2, "reward_plus": reward_plus, "reward_minus": reward_minus}
    return (reward_plus - reward_minus) / denom, {
        "fd_eval_count": 2,
        "reward_plus": reward_plus,
        "reward_minus": reward_minus,
    }


def smoothed_delta(phi: np.ndarray, eps: float) -> np.ndarray:
    abs_phi = np.abs(phi)
    delta = np.zeros_like(phi)
    band = abs_phi <= eps
    delta[band] = 0.5 / eps * (1.0 + np.cos(np.pi * phi[band] / eps))
    return delta


def compute_force_coefficients_np(
    pressure: np.ndarray,
    velocity_u: np.ndarray,
    velocity_v: np.ndarray,
    levelset: np.ndarray,
    volume_fraction: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dynamic_viscosity: float,
    rho_ref: float,
    u_ref: float,
    s_ref: float,
) -> Tuple[float, float, float, float]:
    dphi_dx = np.gradient(levelset, x, axis=0, edge_order=2)
    dphi_dy = np.gradient(levelset, y, axis=1, edge_order=2)
    normal_norm = np.sqrt(dphi_dx**2 + dphi_dy**2) + 1.0e-14
    normal_x = dphi_dx / normal_norm
    normal_y = dphi_dy / normal_norm

    du_dx = np.gradient(velocity_u, x, axis=0, edge_order=2)
    du_dy = np.gradient(velocity_u, y, axis=1, edge_order=2)
    dv_dx = np.gradient(velocity_v, x, axis=0, edge_order=2)
    dv_dy = np.gradient(velocity_v, y, axis=1, edge_order=2)

    tau_xx = -pressure + 2.0 * dynamic_viscosity * du_dx
    tau_xy = dynamic_viscosity * (du_dy + dv_dx)
    tau_yy = -pressure + 2.0 * dynamic_viscosity * dv_dy

    traction_x = tau_xx * normal_x + tau_xy * normal_y
    traction_y = tau_xy * normal_x + tau_yy * normal_y

    dx = np.abs(np.gradient(x))
    dy = np.abs(np.gradient(y))
    cell_area = dx[:, None] * dy[None, :]

    interface_eps = 1.5 * min(np.min(dx), np.min(dy))
    delta = smoothed_delta(levelset, interface_eps)
    fluid_side = np.where(volume_fraction > 0.0, 1.0, 0.0)
    surface_weight = delta * normal_norm * fluid_side

    force_x = np.sum(traction_x * surface_weight * cell_area)
    force_y = np.sum(traction_y * surface_weight * cell_area)
    coeff_scale = 0.5 * rho_ref * u_ref**2 * s_ref
    c_d = force_x / coeff_scale
    c_l = force_y / coeff_scale
    return float(force_x), float(force_y), float(c_d), float(c_l)


def write_history_csv(history: List[Dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in HISTORY_FIELDS})
    return out_path


def write_force_timeseries_csv(
    times: np.ndarray,
    cd_series: np.ndarray,
    cl_series: np.ndarray,
    reward_series: np.ndarray,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time", "Cd", "Cl", "Cl_over_abs_Cd"])
        for idx in range(len(times)):
            writer.writerow([float(times[idx]), float(cd_series[idx]), float(cl_series[idx]), float(reward_series[idx])])
    return out_path


def plot_force_timeseries(
    times: np.ndarray,
    cd_series: np.ndarray,
    cl_series: np.ndarray,
    reward_series: np.ndarray,
    out_path: Path,
    title: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(8.8, 7.0), sharex=True)
    ax[0].plot(times, cd_series, color="#1f77b4", lw=1.8)
    ax[0].set_ylabel("Cd")
    ax[0].grid(alpha=0.25)
    ax[1].plot(times, cl_series, color="#d62728", lw=1.8)
    ax[1].set_ylabel("Cl")
    ax[1].grid(alpha=0.25)
    ax[2].plot(times, reward_series, color="#2ca02c", lw=1.8)
    ax[2].set_ylabel("Cl / |Cd|")
    ax[2].set_xlabel("Time")
    ax[2].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_optimization_history(history: List[Dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = np.asarray([row["iter"] for row in history], dtype=np.int64)
    angles = np.asarray([row["angle_deg"] for row in history], dtype=np.float64)
    rewards = np.asarray([row["reward"] for row in history], dtype=np.float64)
    cds = np.asarray([row["mean_cd_second_half"] for row in history], dtype=np.float64)
    cls = np.asarray([row["mean_cl_second_half"] for row in history], dtype=np.float64)
    grads = np.asarray([row["grad"] for row in history], dtype=np.float64)
    steps = np.asarray([row["step_deg"] for row in history], dtype=np.float64)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))

    ax[0, 0].plot(iterations, rewards, color="#1f4e79", lw=2.0, marker="o")
    ax[0, 0].plot(iterations, np.maximum.accumulate(rewards), color="#d62728", lw=1.5, ls="--", label="best so far")
    ax[0, 0].set_title("Reward history")
    ax[0, 0].set_ylabel("mean(Cl / |Cd|)")
    ax[0, 0].grid(alpha=0.25)
    ax[0, 0].legend(loc="best")

    ax[0, 1].plot(iterations, cds, color="#1f77b4", lw=1.8, marker="o", label="Cd")
    ax[0, 1].plot(iterations, cls, color="#d62728", lw=1.8, marker="o", label="Cl")
    ax[0, 1].set_title("Force coefficients")
    ax[0, 1].set_ylabel("second-half mean")
    ax[0, 1].grid(alpha=0.25)
    ax[0, 1].legend(loc="best")

    ax[1, 0].plot(iterations, angles, color="#2c3e50", lw=1.8, marker="o", label="angle")
    ax_step = ax[1, 0].twinx()
    ax_step.bar(iterations, steps, color="#f39c12", alpha=0.25, width=0.55, label="step")
    ax[1, 0].set_title("Angle trajectory")
    ax[1, 0].set_xlabel("Iteration")
    ax[1, 0].set_ylabel("angle [deg]")
    ax_step.set_ylabel("step [deg]")
    ax[1, 0].grid(alpha=0.25)

    grad_colors = ["#d62728" if row["grad_source"] == "fd" else "#1f4e79" for row in history]
    ax[1, 1].axhline(0.0, color="#888888", lw=0.9)
    ax[1, 1].scatter(iterations, grads, c=grad_colors, s=50)
    ax[1, 1].plot(iterations, grads, color="#555555", lw=1.0, alpha=0.6)
    ax[1, 1].set_title("Reward gradient")
    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].set_ylabel("d reward / d angle [rad^-1]")
    ax[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_field(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    field_xy: np.ndarray,
    contour_xy: np.ndarray,
    title: str,
    cmap: str,
    symmetric: bool,
) -> None:
    field_xy = np.asarray(field_xy)
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(field_xy), 99.0))
        vmax = max(vmax, 1.0e-8)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        mesh = ax.pcolormesh(x, y, field_xy.T, shading="auto", cmap=cmap, norm=norm)
    else:
        vmin = float(np.nanpercentile(field_xy, 1.0))
        vmax = float(np.nanpercentile(field_xy, 99.0))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or abs(vmax - vmin) < 1.0e-12:
            vmin = float(np.nanmin(field_xy))
            vmax = float(np.nanmax(field_xy) + 1.0e-12)
        mesh = ax.pcolormesh(x, y, field_xy.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(contour_xy[:, 0], contour_xy[:, 1], color="white", lw=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, shrink=0.9)


def plot_final_simulation_report(
    times: np.ndarray,
    cd_series: np.ndarray,
    cl_series: np.ndarray,
    reward_series: np.ndarray,
    pressure_xy: np.ndarray,
    vorticity_xy: np.ndarray,
    contour_xy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    angle_deg: float,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.4))

    ax[0, 0].plot(times, cd_series, color="#1f77b4", lw=1.8, label="Cd")
    ax[0, 0].plot(times, cl_series, color="#d62728", lw=1.8, label="Cl")
    ax[0, 0].set_xlabel("Time")
    ax[0, 0].set_ylabel("Coefficient")
    ax[0, 0].set_title(f"Force coefficients at {angle_deg:.3f} deg")
    ax[0, 0].legend(loc="best")
    ax[0, 0].grid(alpha=0.25)

    ax[0, 1].plot(times, reward_series, color="#2ca02c", lw=1.8)
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("Cl / |Cd|")
    ax[0, 1].set_title("Instantaneous reward")
    ax[0, 1].grid(alpha=0.25)

    _plot_field(ax[1, 0], x, y, pressure_xy, contour_xy, "Final pressure", "viridis", symmetric=False)
    _plot_field(ax[1, 1], x, y, vorticity_xy, contour_xy, "Final vorticity", "coolwarm", symmetric=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_environment_report(
    out_dir: Path,
    args: argparse.Namespace,
    reference_values: Dict[str, float],
) -> Path:
    env = {
        "started_at_unix": time.time(),
        "cwd": str(Path.cwd()),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_process_count": jax.process_count(),
        "jax_local_device_count": jax.local_device_count(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "solver_precision": args.solver_precision,
        "reference_values": reference_values,
        "command": " ".join(["python", Path(__file__).name] + list(getattr(args, "_argv", []))),
    }
    out_path = out_dir / "environment.json"
    out_path.write_text(json.dumps(env, indent=2))
    return out_path


def prepare_simulation_objects(
    case_dict: Dict[str, Any],
    numerical_dict: Dict[str, Any],
) -> Tuple[InputManager, InitializationManager, SimulationManager, jax.Array, jax.Array, jax.Array, float, float]:
    input_manager = InputManager(case_dict, numerical_dict)
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
    return input_manager, initialization_manager, sim_manager, primes_init, x_cells, y_cells, dt, t0


def run_reference_simulation(
    angle_rad: float,
    contour_local: jax.Array,
    x_cells: jax.Array,
    y_cells: jax.Array,
    base_case: Dict[str, Any],
    numerical_dict: Dict[str, Any],
    center_x: float,
    center_y: float,
    softmin_temperature: float,
    inside_edge_sharpness: float,
    inside_score_sharpness: float,
    reference_values: Dict[str, float],
    output_dir: Path,
    case_name: str,
    sim_time: float,
    save_dt: float,
    generate_xdmf: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = output_dir / "geometry"
    figures_dir = output_dir / "figures"
    results_root = output_dir / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    angle_array = jnp.asarray(angle_rad, dtype=x_cells.dtype)
    contour_jax, levelset_jax = build_naca_levelset(
        angle_rad=angle_array,
        contour_local=contour_local,
        x_cells=x_cells,
        y_cells=y_cells,
        center_x=center_x,
        center_y=center_y,
        softmin_temperature=softmin_temperature,
        inside_edge_sharpness=inside_edge_sharpness,
        inside_score_sharpness=inside_score_sharpness,
    )
    contour_np = np.asarray(contour_jax)
    levelset_np = np.asarray(levelset_jax)

    levelset_path = write_levelset_h5(levelset_np, geometry_dir / "levelset.h5")
    preview_path = save_airfoil_preview(contour_np, math.degrees(angle_rad), geometry_dir / "airfoil.png", (center_x, center_y))

    case_dict = copy.deepcopy(base_case)
    case_dict["general"]["case_name"] = case_name
    case_dict["general"]["save_path"] = str(results_root.resolve())
    case_dict["general"]["end_time"] = float(sim_time)
    case_dict["general"]["save_dt"] = float(save_dt)
    case_dict["initial_condition"]["levelset"] = str(levelset_path.resolve())

    input_manager = InputManager(case_dict, numerical_dict)
    initialization_manager = InitializationManager(input_manager)
    simulation_manager = SimulationManager(input_manager)
    jxf_buffers = initialization_manager.initialization()
    simulation_manager.simulate(jxf_buffers)

    output_path = Path(simulation_manager.output_writer.save_path_case)
    domain_path = output_path / "domain"
    if generate_xdmf and not (domain_path / "data_time_series.xdmf").exists():
        create_xdmf_from_h5(str(domain_path))

    data = load_data(
        str(output_path),
        quantities=["pressure", "velocity", "levelset", "volume_fraction"],
        verbose=False,
    )

    x_np = np.asarray(data.cell_centers[0]).reshape(-1)
    y_np = np.asarray(data.cell_centers[1]).reshape(-1)
    pressure_series = data.data["pressure"]
    velocity_series = data.data["velocity"]
    levelset_series = data.data["levelset"]
    volume_fraction_series = data.data["volume_fraction"]
    times = np.asarray(data.times, dtype=np.float64)

    rows = []
    for idx, sim_time_value in enumerate(times):
        pressure = np.squeeze(pressure_series[idx])
        velocity_u = np.squeeze(velocity_series[idx, 0])
        velocity_v = np.squeeze(velocity_series[idx, 1])
        levelset = np.squeeze(levelset_series[idx])
        volume_fraction = np.squeeze(volume_fraction_series[idx])
        _, _, c_d, c_l = compute_force_coefficients_np(
            pressure=pressure,
            velocity_u=velocity_u,
            velocity_v=velocity_v,
            levelset=levelset,
            volume_fraction=volume_fraction,
            x=x_np,
            y=y_np,
            dynamic_viscosity=reference_values["dynamic_viscosity"],
            rho_ref=reference_values["rho_ref"],
            u_ref=reference_values["u_ref"],
            s_ref=reference_values["s_ref"],
        )
        rows.append((float(sim_time_value), c_d, c_l, c_l / max(abs(c_d), 1.0e-12)))

    arr = np.asarray(rows, dtype=np.float64)
    half_idx = len(arr) // 2
    cd_series = arr[:, 1]
    cl_series = arr[:, 2]
    reward_series = arr[:, 3]

    metrics_csv = write_force_timeseries_csv(times, cd_series, cl_series, reward_series, output_dir / "drag_lift_timeseries.csv")
    metrics_plot = plot_force_timeseries(
        times=times,
        cd_series=cd_series,
        cl_series=cl_series,
        reward_series=reward_series,
        out_path=figures_dir / "drag_lift_timeseries.png",
        title=f"Forward simulation diagnostics at {math.degrees(angle_rad):.3f} deg",
    )

    pressure_final = np.squeeze(pressure_series[-1])
    velocity_u_final = np.squeeze(velocity_series[-1, 0])
    velocity_v_final = np.squeeze(velocity_series[-1, 1])
    vorticity_final = np.gradient(velocity_v_final, x_np, axis=0, edge_order=2) - np.gradient(
        velocity_u_final, y_np, axis=1, edge_order=2
    )

    report_png = plot_final_simulation_report(
        times=times,
        cd_series=cd_series,
        cl_series=cl_series,
        reward_series=reward_series,
        pressure_xy=pressure_final,
        vorticity_xy=vorticity_final,
        contour_xy=contour_np,
        x=x_np,
        y=y_np,
        angle_deg=math.degrees(angle_rad),
        out_path=figures_dir / "final_simulation_report.png",
    )

    summary = {
        "case_name": output_path.name,
        "angle_deg": math.degrees(angle_rad),
        "time_start": float(times[0]),
        "time_end": float(times[-1]),
        "mean_cd_second_half": float(np.mean(cd_series[half_idx:])),
        "mean_cl_second_half": float(np.mean(cl_series[half_idx:])),
        "mean_reward_second_half": float(np.mean(reward_series[half_idx:])),
        "simulation_output_path": str(output_path.resolve()),
        "output_log_path": str((output_path / "output.log").resolve()),
        "levelset_h5": str(levelset_path.resolve()),
        "airfoil_preview_png": str(preview_path.resolve()),
        "metrics_csv": str(metrics_csv.resolve()),
        "metrics_plot_png": str(metrics_plot.resolve()),
        "final_report_png": str(report_png.resolve()),
        "xdmf_path": str((domain_path / "data_time_series.xdmf").resolve()) if generate_xdmf else None,
    }
    summary_path = output_dir / "simulation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_json"] = str(summary_path.resolve())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize or simulate the angle of attack of a NACA airfoil.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--case-setup", default="case_setup_naca_re100.json")
        target.add_argument("--numerical-setup", default="numerical_setup.json")
        target.add_argument("--output-dir", default="runs/naca_angle")
        target.add_argument("--naca-code", default="0012")
        target.add_argument("--chord", type=float, default=1.0)
        target.add_argument("--pivot-fraction", type=float, default=0.25)
        target.add_argument("--center-x", type=float, default=0.0)
        target.add_argument("--center-y", type=float, default=0.0)
        target.add_argument("--geometry-samples", type=int, default=160)
        target.add_argument("--sim-time", type=float, default=10.0)
        target.add_argument("--save-dt", type=float, default=0.25)
        target.add_argument("--solver-precision", choices=["from-setup", "single", "double"], default="from-setup")
        target.add_argument("--generate-xdmf", dest="generate_xdmf", action="store_true")
        target.add_argument("--no-generate-xdmf", dest="generate_xdmf", action="store_false")
        target.set_defaults(generate_xdmf=True)

    optimize = subparsers.add_parser("optimize", help="Run differentiable angle optimization and an optional final simulation.")
    add_common_arguments(optimize)
    optimize.add_argument("--initial-angle-deg", type=float, default=2.0)
    optimize.add_argument("--min-angle-deg", type=float, default=-5.0)
    optimize.add_argument("--max-angle-deg", type=float, default=15.0)
    optimize.add_argument("--max-iters", type=int, default=12)
    optimize.add_argument("--learning-rate-deg", type=float, default=0.8)
    optimize.add_argument("--adam-beta1", type=float, default=0.9)
    optimize.add_argument("--adam-beta2", type=float, default=0.999)
    optimize.add_argument("--adam-eps", type=float, default=1.0e-8)
    optimize.add_argument("--max-step-deg", type=float, default=2.0)
    optimize.add_argument("--step-tol-deg", type=float, default=0.03)
    optimize.add_argument("--reward-tol", type=float, default=1.0e-4)
    optimize.add_argument("--grad-tol", type=float, default=1.0e-4)
    optimize.add_argument("--convergence-patience", type=int, default=3)
    optimize.add_argument("--inner-steps", type=int, default=128)
    optimize.add_argument("--checkpoint-integration-step", action="store_true")
    optimize.add_argument("--checkpoint-inner-step", dest="checkpoint_inner_step", action="store_true")
    optimize.add_argument("--no-checkpoint-inner-step", dest="checkpoint_inner_step", action="store_false")
    optimize.set_defaults(checkpoint_inner_step=True)
    optimize.add_argument("--jit", dest="jit", action="store_true")
    optimize.add_argument("--no-jit", dest="jit", action="store_false")
    optimize.set_defaults(jit=True)
    optimize.add_argument("--fd-eps-deg", type=float, default=0.1)
    optimize.add_argument("--enable-fd-fallback", dest="enable_fd_fallback", action="store_true")
    optimize.add_argument("--disable-fd-fallback", dest="enable_fd_fallback", action="store_false")
    optimize.set_defaults(enable_fd_fallback=True)
    optimize.add_argument("--save-every", type=int, default=1)
    optimize.add_argument("--skip-final-simulation", action="store_true")

    simulate = subparsers.add_parser("simulate", help="Run one forward simulation at a fixed angle.")
    add_common_arguments(simulate)
    simulate.add_argument("--angle-deg", type=float, required=True)

    return parser


def run_optimize(args: argparse.Namespace) -> None:
    if args.max_iters < 1:
        raise ValueError("--max-iters must be >= 1")
    if args.geometry_samples < 16:
        raise ValueError("--geometry-samples must be >= 16")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if args.max_angle_deg <= args.min_angle_deg:
        raise ValueError("--max-angle-deg must be larger than --min-angle-deg")

    out_dir = Path(args.output_dir)
    figures_dir = out_dir / "figures"
    iter_dir = out_dir / "iterations"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    iter_dir.mkdir(parents=True, exist_ok=True)

    base_case = load_json(args.case_setup)
    numerical_dict = maybe_apply_solver_precision_override(load_json(args.numerical_setup), args.solver_precision)
    numerical_ff_dict = prepare_feedforward_compatible_numerical_setup(numerical_dict)
    reference_values = extract_reference_values(base_case, args.chord)

    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    (
        _input_manager,
        _initialization_manager,
        sim_manager,
        primes_init,
        x_cells,
        y_cells,
        dt,
        t0,
    ) = prepare_simulation_objects(base_case, numerical_ff_dict)
    save_environment_report(out_dir, args, reference_values)

    solver_dtype = get_solver_dtype(sim_manager)
    contour_local = build_naca_contour_local(
        digit_code=args.naca_code,
        chord=args.chord,
        samples_per_side=args.geometry_samples,
        pivot_fraction=args.pivot_fraction,
        dtype=solver_dtype,
    )
    smallest_cell_size = float(
        min(np.min(np.abs(np.gradient(np.asarray(x_cells)))), np.min(np.abs(np.gradient(np.asarray(y_cells)))))
    )
    softmin_temperature = 0.75 * smallest_cell_size
    inside_edge_sharpness = 1.5 / smallest_cell_size
    inside_score_sharpness = 18.0

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

    angle_min_rad = math.radians(args.min_angle_deg)
    angle_max_rad = math.radians(args.max_angle_deg)
    objective_fn = build_objective(
        sim_manager=sim_manager,
        primes_init=primes_init,
        x_cells=x_cells,
        y_cells=y_cells,
        dt=dt,
        t0=t0,
        feed_forward_setup=ff_setup,
        contour_local=contour_local,
        center_x=args.center_x,
        center_y=args.center_y,
        softmin_temperature=softmin_temperature,
        inside_edge_sharpness=inside_edge_sharpness,
        inside_score_sharpness=inside_score_sharpness,
        angle_min_rad=angle_min_rad,
        angle_max_rad=angle_max_rad,
        reference_values=reference_values,
    )
    value_and_grad = jax.value_and_grad(objective_fn, has_aux=True)
    if args.jit:
        value_and_grad = jax.jit(value_and_grad)

    current_angle_rad = min(max(math.radians(args.initial_angle_deg), angle_min_rad), angle_max_rad)
    learning_rate_rad = math.radians(args.learning_rate_deg)
    max_step_rad = math.radians(args.max_step_deg)
    step_tol_rad = math.radians(args.step_tol_deg)
    fd_eps_rad = math.radians(args.fd_eps_deg)

    adam_m = 0.0
    adam_v = 0.0
    history: List[Dict[str, Any]] = []
    stop_reason = "max_iters_reached"
    converged = False
    stable_iters = 0
    best_reward = -math.inf
    best_state: Dict[str, Any] | None = None
    previous_reward = None

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Reference Reynolds number: {reference_values['reynolds_number']:.3f}")
    print(f"Feed-forward dt={dt:.6e}, total_steps={total_steps}, outer_steps={outer_steps}")

    for iteration in range(args.max_iters):
        iter_start = time.time()
        (reward_raw, aux_raw), grad_raw = value_and_grad(jnp.asarray(current_angle_rad))
        reward_scalar = float(np.asarray(reward_raw))
        grad_scalar = float(np.asarray(grad_raw))

        grad_source = "ad"
        fd_eval_count = 0
        if not math.isfinite(grad_scalar) and args.enable_fd_fallback:
            grad_scalar, fd_info = finite_difference_grad(
                objective_fn=objective_fn,
                angle_rad=current_angle_rad,
                angle_min_rad=angle_min_rad,
                angle_max_rad=angle_max_rad,
                fd_eps_rad=fd_eps_rad,
            )
            fd_eval_count = int(fd_info["fd_eval_count"])
            grad_source = "fd"

        if not math.isfinite(grad_scalar):
            grad_scalar = 0.0
            grad_source = "zeroed"

        reward_value = reward_scalar if math.isfinite(reward_scalar) else -1.0e6

        adam_m = args.adam_beta1 * adam_m + (1.0 - args.adam_beta1) * grad_scalar
        adam_v = args.adam_beta2 * adam_v + (1.0 - args.adam_beta2) * (grad_scalar**2)
        bias_correction_1 = 1.0 - args.adam_beta1 ** (iteration + 1)
        bias_correction_2 = 1.0 - args.adam_beta2 ** (iteration + 1)
        adam_m_hat = adam_m / bias_correction_1
        adam_v_hat = adam_v / bias_correction_2
        step_rad = learning_rate_rad * adam_m_hat / (math.sqrt(adam_v_hat) + args.adam_eps)
        step_rad = float(np.clip(step_rad, -max_step_rad, max_step_rad))

        next_angle_rad = float(np.clip(current_angle_rad + step_rad, angle_min_rad, angle_max_rad))
        if abs(next_angle_rad - current_angle_rad) < 1.0e-14:
            step_rad = next_angle_rad - current_angle_rad

        reward_valid_fraction = float(
            np.asarray(aux_raw["reward_valid_count_second_half"]) / max(float(np.asarray(aux_raw["reward_total_count_second_half"])), 1.0)
        )
        row = {
            "iter": iteration,
            "angle_deg": math.degrees(current_angle_rad),
            "reward": reward_value,
            "mean_cd_second_half": float(np.nan_to_num(np.asarray(aux_raw["mean_cd_second_half"]), nan=1.0e6)),
            "mean_cl_second_half": float(np.nan_to_num(np.asarray(aux_raw["mean_cl_second_half"]), nan=0.0)),
            "grad": grad_scalar,
            "step_deg": math.degrees(step_rad),
            "reward_is_finite": math.isfinite(reward_scalar),
            "grad_is_finite": math.isfinite(float(np.asarray(grad_raw))),
            "grad_source": grad_source,
            "fd_eval_count": fd_eval_count,
            "reward_valid_fraction_second_half": reward_valid_fraction,
            "primitives_nonfinite_count": int(np.asarray(aux_raw["primitives_nonfinite_count"])),
            "cd_nonfinite_count": int(np.asarray(aux_raw["cd_nonfinite_count"])),
            "cl_nonfinite_count": int(np.asarray(aux_raw["cl_nonfinite_count"])),
            "max_abs_primitives": float(np.asarray(aux_raw["max_abs_primitives"])),
            "wall_seconds": time.time() - iter_start,
            "sim_end_time": float(np.asarray(aux_raw["end_time"])),
        }
        history.append(row)
        write_history_csv(history, out_dir / "history.csv")

        should_save_geometry = iteration % args.save_every == 0 or iteration == args.max_iters - 1
        if should_save_geometry:
            contour_np = np.asarray(rotate_translate(contour_local, current_angle_rad, args.center_x, args.center_y))
            save_airfoil_preview(
                contour_xy=contour_np,
                angle_deg=row["angle_deg"],
                out_path=iter_dir / f"iter_{iteration:03d}_airfoil.png",
                center=(args.center_x, args.center_y),
            )

        if reward_value > best_reward:
            best_reward = reward_value
            best_state = {
                "iter": iteration,
                "angle_rad": current_angle_rad,
                "angle_deg": row["angle_deg"],
                "reward": reward_value,
                "mean_cd_second_half": row["mean_cd_second_half"],
                "mean_cl_second_half": row["mean_cl_second_half"],
                "times": np.asarray(aux_raw["times"], dtype=np.float64),
                "cd_series": np.asarray(aux_raw["cd_series"], dtype=np.float64),
                "cl_series": np.asarray(aux_raw["cl_series"], dtype=np.float64),
                "reward_series": np.asarray(aux_raw["reward_series"], dtype=np.float64),
            }
            write_force_timeseries_csv(
                times=best_state["times"],
                cd_series=best_state["cd_series"],
                cl_series=best_state["cl_series"],
                reward_series=best_state["reward_series"],
                out_path=out_dir / "best_iter_timeseries.csv",
            )
            plot_force_timeseries(
                times=best_state["times"],
                cd_series=best_state["cd_series"],
                cl_series=best_state["cl_series"],
                reward_series=best_state["reward_series"],
                out_path=figures_dir / "best_iter_timeseries.png",
                title=f"Best optimization trajectory at iter {iteration:03d}",
            )

        reward_change = abs(reward_value - previous_reward) if previous_reward is not None else math.inf
        if abs(step_rad) < step_tol_rad and reward_change < args.reward_tol and abs(grad_scalar) < args.grad_tol:
            stable_iters += 1
        else:
            stable_iters = 0

        print(
            f"[iter {iteration:03d}] angle={row['angle_deg']:.4f} deg "
            f"reward={row['reward']:.6f} Cd={row['mean_cd_second_half']:.6f} "
            f"Cl={row['mean_cl_second_half']:.6f} grad={row['grad']:.6e} "
            f"step={row['step_deg']:.4f} deg source={row['grad_source']}"
        )

        previous_reward = reward_value
        current_angle_rad = next_angle_rad

        if stable_iters >= args.convergence_patience:
            stop_reason = "converged"
            converged = True
            break

    if best_state is None:  # pragma: no cover - defensive
        raise RuntimeError("Optimization completed without a valid best state.")

    history_json = out_dir / "history.json"
    history_json.write_text(json.dumps(history, indent=2))
    history_plot = plot_optimization_history(history, figures_dir / "optimization_history.png")
    overlay_plot = save_airfoil_overlay(
        contour_local=contour_local,
        initial_angle_rad=math.radians(args.initial_angle_deg),
        best_angle_rad=best_state["angle_rad"],
        center=(args.center_x, args.center_y),
        out_path=figures_dir / "airfoil_overlay.png",
    )

    final_sim_summary = None
    if not args.skip_final_simulation:
        final_sim_summary = run_reference_simulation(
            angle_rad=best_state["angle_rad"],
            contour_local=contour_local,
            x_cells=x_cells,
            y_cells=y_cells,
            base_case=base_case,
            numerical_dict=numerical_dict,
            center_x=args.center_x,
            center_y=args.center_y,
            softmin_temperature=softmin_temperature,
            inside_edge_sharpness=inside_edge_sharpness,
            inside_score_sharpness=inside_score_sharpness,
            reference_values=reference_values,
            output_dir=out_dir / "final_simulation",
            case_name="naca_aoa_re100_optimized",
            sim_time=args.sim_time,
            save_dt=args.save_dt,
            generate_xdmf=args.generate_xdmf,
        )

    final_summary = {
        "converged": converged,
        "stop_reason": stop_reason,
        "iterations_completed": len(history),
        "best_iteration": best_state["iter"],
        "initial_angle_deg": args.initial_angle_deg,
        "best_angle_deg": best_state["angle_deg"],
        "best_reward": best_state["reward"],
        "best_mean_cd_second_half": best_state["mean_cd_second_half"],
        "best_mean_cl_second_half": best_state["mean_cl_second_half"],
        "history_csv": str((out_dir / "history.csv").resolve()),
        "history_json": str(history_json.resolve()),
        "optimization_history_png": str(history_plot.resolve()),
        "airfoil_overlay_png": str(overlay_plot.resolve()),
        "best_iter_timeseries_csv": str((out_dir / "best_iter_timeseries.csv").resolve()),
        "best_iter_timeseries_png": str((figures_dir / "best_iter_timeseries.png").resolve()),
        "final_simulation": final_sim_summary,
    }
    final_summary_path = out_dir / "final_summary.json"
    final_summary_path.write_text(json.dumps(final_summary, indent=2))
    print(f"Saved final summary to: {final_summary_path.resolve()}")


def run_simulate(args: argparse.Namespace) -> None:
    if args.geometry_samples < 16:
        raise ValueError("--geometry-samples must be >= 16")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_case = load_json(args.case_setup)
    numerical_dict = maybe_apply_solver_precision_override(load_json(args.numerical_setup), args.solver_precision)
    reference_values = extract_reference_values(base_case, args.chord)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    (
        _input_manager,
        _initialization_manager,
        sim_manager,
        _primes_init,
        x_cells,
        y_cells,
        _dt,
        _t0,
    ) = prepare_simulation_objects(base_case, numerical_dict)
    save_environment_report(out_dir, args, reference_values)

    contour_local = build_naca_contour_local(
        digit_code=args.naca_code,
        chord=args.chord,
        samples_per_side=args.geometry_samples,
        pivot_fraction=args.pivot_fraction,
        dtype=get_solver_dtype(sim_manager),
    )
    smallest_cell_size = float(
        min(np.min(np.abs(np.gradient(np.asarray(x_cells)))), np.min(np.abs(np.gradient(np.asarray(y_cells)))))
    )
    summary = run_reference_simulation(
        angle_rad=math.radians(args.angle_deg),
        contour_local=contour_local,
        x_cells=x_cells,
        y_cells=y_cells,
        base_case=base_case,
        numerical_dict=numerical_dict,
        center_x=args.center_x,
        center_y=args.center_y,
        softmin_temperature=0.75 * smallest_cell_size,
        inside_edge_sharpness=1.5 / smallest_cell_size,
        inside_score_sharpness=18.0,
        reference_values=reference_values,
        output_dir=out_dir / "single_simulation",
        case_name="naca_aoa_re100_single_angle",
        sim_time=args.sim_time,
        save_dt=args.save_dt,
        generate_xdmf=args.generate_xdmf,
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args._argv = list(sys.argv[1:])

    if args.command == "optimize":
        run_optimize(args)
    elif args.command == "simulate":
        run_simulate(args)
    else:  # pragma: no cover - argparse guards this
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
