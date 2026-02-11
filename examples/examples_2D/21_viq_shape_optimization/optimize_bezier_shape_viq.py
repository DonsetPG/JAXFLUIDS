from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np

from jaxfluids import InitializationManager, InputManager, SimulationManager
from jaxfluids_postprocess import load_data

from bezier_shape import (
    bezier_closed_from_four_points,
    clamp_points_to_ring,
    render_shape_preview,
    signed_distance_from_contour,
    write_levelset_h5,
)


def smoothed_delta(phi: np.ndarray, eps: float) -> np.ndarray:
    abs_phi = np.abs(phi)
    delta = np.zeros_like(phi)
    band = abs_phi <= eps
    delta[band] = 0.5 / eps * (1.0 + np.cos(np.pi * phi[band] / eps))
    return delta


def compute_force_coefficients(
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
) -> tuple[float, float, float, float]:
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


def build_levelset_from_points(
    control_points: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    contour = bezier_closed_from_four_points(control_points, samples_per_segment=220, tension=1.0)
    levelset = signed_distance_from_contour(contour, x, y)
    return contour, levelset


def evaluate_candidate(
    tag: str,
    control_points: np.ndarray,
    base_case: Dict,
    base_numerical: Dict,
    x: np.ndarray,
    y: np.ndarray,
    work_dir: Path,
) -> Dict:
    eval_dir = work_dir / tag
    eval_dir.mkdir(parents=True, exist_ok=True)

    contour, levelset = build_levelset_from_points(control_points, x, y)
    levelset_path = write_levelset_h5(levelset, eval_dir / "levelset.h5")
    render_shape_preview(contour, control_points, eval_dir / "shape.png")

    case_dict = copy.deepcopy(base_case)
    numerical_dict = copy.deepcopy(base_numerical)

    case_dict["general"]["case_name"] = f"viq_opt_{tag}"
    case_dict["general"]["save_path"] = str((work_dir / "results").resolve())
    case_dict["general"]["end_time"] = 2.0
    case_dict["general"]["save_dt"] = 0.25
    case_dict["initial_condition"]["levelset"] = str(levelset_path.resolve())

    # Use light output for optimization loops.
    numerical_dict["output"]["is_xdmf"] = False
    numerical_dict["output"]["logging"]["frequency"] = 1000

    input_manager = InputManager(case_dict, numerical_dict)
    initialization_manager = InitializationManager(input_manager)
    simulation_manager = SimulationManager(input_manager)
    jxf_buffers = initialization_manager.initialization()
    simulation_manager.simulate(jxf_buffers)

    output_path = Path(simulation_manager.output_writer.save_path_case)
    data = load_data(
        str(output_path),
        quantities=["pressure", "velocity", "levelset", "volume_fraction"],
        verbose=False,
    )

    x_cells, y_cells, _ = data.cell_centers
    x_cells = np.asarray(x_cells).reshape(-1)
    y_cells = np.asarray(y_cells).reshape(-1)

    pressure_series = data.data["pressure"]
    velocity_series = data.data["velocity"]
    levelset_series = data.data["levelset"]
    volume_fraction_series = data.data["volume_fraction"]

    rows = []
    for idx, sim_time in enumerate(data.times):
        pressure = np.squeeze(pressure_series[idx])
        velocity_u = np.squeeze(velocity_series[idx, 0])
        velocity_v = np.squeeze(velocity_series[idx, 1])
        levelset_i = np.squeeze(levelset_series[idx])
        volume_fraction_i = np.squeeze(volume_fraction_series[idx])

        _, _, c_d, c_l = compute_force_coefficients(
            pressure=pressure,
            velocity_u=velocity_u,
            velocity_v=velocity_v,
            levelset=levelset_i,
            volume_fraction=volume_fraction_i,
            x=x_cells,
            y=y_cells,
            dynamic_viscosity=0.01,
            rho_ref=1.0,
            u_ref=1.0,
            s_ref=2.0,
        )
        rows.append((float(sim_time), c_d, c_l, c_l / max(abs(c_d), 1.0e-12)))

    csv_path = eval_dir / "coeff_timeseries.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "Cd", "Cl", "Cl_over_abs_Cd"])
        writer.writerows(rows)

    arr = np.asarray(rows, dtype=np.float64)
    half_idx = len(arr) // 2
    reward = float(np.mean(arr[half_idx:, 3]))

    result = {
        "tag": tag,
        "reward_mean_second_half": reward,
        "mean_cd_second_half": float(np.mean(arr[half_idx:, 1])),
        "mean_cl_second_half": float(np.mean(arr[half_idx:, 2])),
        "output_path": str(output_path.resolve()),
        "shape_preview_png": str((eval_dir / "shape.png").resolve()),
        "shape_levelset_h5": str(levelset_path.resolve()),
        "coeff_timeseries_csv": str(csv_path.resolve()),
        "control_points": control_points.tolist(),
    }
    (eval_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    case_setup_file = Path("case_setup_viq_re200.json")
    numerical_setup_file = Path("numerical_setup.json")
    work_dir = Path("optimization_4pt_bezier")
    work_dir.mkdir(parents=True, exist_ok=True)

    base_case = json.loads(case_setup_file.read_text())
    base_numerical = json.loads(numerical_setup_file.read_text())

    grid_input_manager = InputManager(str(case_setup_file), str(numerical_setup_file))
    x_raw, y_raw, _ = grid_input_manager.domain_information.get_global_cell_centers_unsplit()
    x = np.asarray(x_raw).reshape(-1)
    y = np.asarray(y_raw).reshape(-1)

    # Four-point parameterization to optimize.
    theta = np.array(
        [
            [1.05, 0.00],
            [0.05, 1.00],
            [-1.00, 0.10],
            [0.00, -0.95],
        ],
        dtype=np.float64,
    )
    theta = clamp_points_to_ring(theta, r_min=0.7, r_max=1.3)

    rng = np.random.default_rng(17)
    delta = rng.choice(np.array([-1.0, 1.0]), size=theta.shape)
    eps = 0.03
    lr = 0.08

    theta_plus = clamp_points_to_ring(theta + eps * delta, r_min=0.7, r_max=1.3)
    theta_minus = clamp_points_to_ring(theta - eps * delta, r_min=0.7, r_max=1.3)

    base_eval = evaluate_candidate("base", theta, base_case, base_numerical, x, y, work_dir)
    plus_eval = evaluate_candidate("plus", theta_plus, base_case, base_numerical, x, y, work_dir)
    minus_eval = evaluate_candidate("minus", theta_minus, base_case, base_numerical, x, y, work_dir)

    directional_derivative = (plus_eval["reward_mean_second_half"] - minus_eval["reward_mean_second_half"]) / (2.0 * eps)
    smoothness_indicator = abs(
        0.5 * (plus_eval["reward_mean_second_half"] + minus_eval["reward_mean_second_half"])
        - base_eval["reward_mean_second_half"]
    )
    grad_estimate = directional_derivative * delta

    theta_new = clamp_points_to_ring(theta + lr * grad_estimate, r_min=0.7, r_max=1.3)
    new_eval = evaluate_candidate("after_step_1", theta_new, base_case, base_numerical, x, y, work_dir)

    summary = {
        "objective": "mean_second_half(Cl/|Cd|)",
        "differentiability_check": {
            "epsilon": eps,
            "directional_derivative_central_diff": float(directional_derivative),
            "smoothness_indicator": float(smoothness_indicator),
            "delta_direction": delta.tolist(),
        },
        "gradient_ascent_step": {
            "learning_rate": lr,
            "reward_before": base_eval["reward_mean_second_half"],
            "reward_after": new_eval["reward_mean_second_half"],
            "improvement": float(new_eval["reward_mean_second_half"] - base_eval["reward_mean_second_half"]),
            "theta_before": theta.tolist(),
            "theta_after": theta_new.tolist(),
        },
        "evaluations": {
            "base": base_eval,
            "plus": plus_eval,
            "minus": minus_eval,
            "after_step_1": new_eval,
        },
    }

    summary_path = work_dir / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary: {summary_path.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
