import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxfluids import InitializationManager, InputManager, SimulationManager
from jaxfluids_postprocess import create_xdmf_from_h5, load_data


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
):
    dphi_dx = np.gradient(levelset, x, axis=0, edge_order=2)
    dphi_dy = np.gradient(levelset, y, axis=1, edge_order=2)
    normal_norm = np.sqrt(dphi_dx ** 2 + dphi_dy ** 2) + 1.0e-14
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

    # Keep only the fluid side in the narrow interface band.
    fluid_side = np.where(volume_fraction > 0.0, 1.0, 0.0)
    surface_weight = delta * normal_norm * fluid_side

    force_x = np.sum(traction_x * surface_weight * cell_area)
    force_y = np.sum(traction_y * surface_weight * cell_area)

    dynamic_pressure_ref = 0.5 * rho_ref * u_ref ** 2
    coeff_scale = dynamic_pressure_ref * s_ref
    c_d = force_x / coeff_scale
    c_l = force_y / coeff_scale
    return float(force_x), float(force_y), float(c_d), float(c_l)


def main():
    case_setup_file = "case_setup_viq_re200.json"
    numerical_setup_file = "numerical_setup.json"
    skip_simulation = os.environ.get("SKIP_SIMULATION", "0") == "1"
    make_xdmf_only = os.environ.get("MAKE_XDMF_ONLY", "0") == "1"
    if skip_simulation:
        with open(case_setup_file, "r", encoding="utf-8") as f:
            case_setup = json.load(f)
        output_path = Path(case_setup["general"]["save_path"]) / case_setup["general"]["case_name"]
    else:
        input_manager = InputManager(case_setup_file, numerical_setup_file)
        initialization_manager = InitializationManager(input_manager)
        simulation_manager = SimulationManager(input_manager)

        jxf_buffers = initialization_manager.initialization()
        simulation_manager.simulate(jxf_buffers)
        output_path = Path(simulation_manager.output_writer.save_path_case)

    domain_path = output_path / "domain"
    if make_xdmf_only:
        create_xdmf_from_h5(str(domain_path))
        print(f"Generated XDMF files in: {domain_path}")
        return

    if not (domain_path / "data_time_series.xdmf").exists():
        create_xdmf_from_h5(str(domain_path))

    data = load_data(
        str(output_path),
        quantities=["pressure", "velocity", "levelset", "volume_fraction"],
        verbose=True,
    )

    x, y, _ = data.cell_centers
    times = data.times
    pressure_series = data.data["pressure"]
    velocity_series = data.data["velocity"]
    levelset_series = data.data["levelset"]
    volume_fraction_series = data.data["volume_fraction"]

    rho_ref = 1.0
    u_ref = 1.0
    s_ref = 2.0
    mu = 0.01

    rows = []
    for idx, sim_time in enumerate(times):
        pressure = np.squeeze(pressure_series[idx])
        velocity_u = np.squeeze(velocity_series[idx, 0])
        velocity_v = np.squeeze(velocity_series[idx, 1])
        levelset = np.squeeze(levelset_series[idx])
        volume_fraction = np.squeeze(volume_fraction_series[idx])

        force_x, force_y, c_d, c_l = compute_force_coefficients(
            pressure=pressure,
            velocity_u=velocity_u,
            velocity_v=velocity_v,
            levelset=levelset,
            volume_fraction=volume_fraction,
            x=x,
            y=y,
            dynamic_viscosity=mu,
            rho_ref=rho_ref,
            u_ref=u_ref,
            s_ref=s_ref,
        )
        rows.append((float(sim_time), force_x, force_y, c_d, c_l))

    metrics_path = output_path / "drag_lift_timeseries.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "drag_force", "lift_force", "Cd", "Cl"])
        writer.writerows(rows)

    arr = np.asarray(rows, dtype=np.float64)
    half_idx = len(arr) // 2
    summary = {
        "case_name": output_path.name,
        "time_start": float(arr[0, 0]),
        "time_end": float(arr[-1, 0]),
        "mean_drag_force_second_half": float(np.mean(arr[half_idx:, 1])),
        "mean_lift_force_second_half": float(np.mean(arr[half_idx:, 2])),
        "mean_Cd_second_half": float(np.mean(arr[half_idx:, 3])),
        "mean_Cl_second_half": float(np.mean(arr[half_idx:, 4])),
        "mean_Cl_over_abs_Cd_second_half": float(
            np.mean(arr[half_idx:, 4] / np.maximum(np.abs(arr[half_idx:, 3]), 1.0e-12))
        ),
    }

    summary_path = output_path / "drag_lift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    fig_path = output_path / "drag_lift_coefficients.png"
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(arr[:, 0], arr[:, 3], lw=1.5, color="tab:blue", label="Cd")
    ax[0].plot(arr[:, 0], arr[:, 4], lw=1.5, color="tab:orange", label="Cl")
    ax[0].set_ylabel("Coefficient")
    ax[0].legend(loc="best")
    ax[0].grid(alpha=0.3)

    ax[1].plot(arr[:, 0], arr[:, 1], lw=1.5, color="tab:green", label="Drag force")
    ax[1].plot(arr[:, 0], arr[:, 2], lw=1.5, color="tab:red", label="Lift force")
    ax[1].set_ylabel("Force")
    ax[1].set_xlabel("Time")
    ax[1].legend(loc="best")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"Simulation output path: {output_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {fig_path}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
