import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxfluids import InitializationManager, InputManager, SimulationManager
from jaxfluids_postprocess import create_xdmf_from_h5, load_data


PAPER_PARAMETERS = {
    "diameter_D": 1.0,
    "radius_R": 0.5,
    "domain_length_Lx": 22.0,
    "domain_height_H": 4.1,
    "domain_x_range": [-2.0, 20.0],
    "domain_y_range": [-2.0, 2.1],
    "cylinder_center_y_shift_vs_domain_centerline": 0.05,
    "mean_inflow_U_bar": 1.0,
    "density_rho": 1.0,
    "reynolds_number": 100.0,
    "jet_centers_deg": [90.0, 270.0],
    "jet_width_deg": 10.0,
    "jet_q_star_limit_abs": 0.06,
}


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

    fluid_side = np.where(volume_fraction > 0.0, 1.0, 0.0)
    surface_weight = delta * normal_norm * fluid_side

    force_x = np.sum(traction_x * surface_weight * cell_area)
    force_y = np.sum(traction_y * surface_weight * cell_area)

    dynamic_pressure_ref = 0.5 * rho_ref * u_ref ** 2
    coeff_scale = dynamic_pressure_ref * s_ref
    c_d = force_x / coeff_scale
    c_l = force_y / coeff_scale
    return float(force_x), float(force_y), float(c_d), float(c_l)


def build_inflow_u_expression() -> str:
    h = PAPER_PARAMETERS["domain_height_H"]
    y_centerline = PAPER_PARAMETERS["cylinder_center_y_shift_vs_domain_centerline"]
    return (
        "lambda y,t: "
        f"6.0 * ({0.5 * h:.16g} - (y - {y_centerline:.16g})) "
        f"* ({0.5 * h:.16g} + (y - {y_centerline:.16g})) / ({h:.16g}**2)"
    )


def compute_q_ref() -> float:
    d = PAPER_PARAMETERS["diameter_D"]
    y = np.linspace(-0.5 * d, 0.5 * d, 4001)
    inflow = 6.0 * (0.5 * PAPER_PARAMETERS["domain_height_H"] - (y - 0.0)) * (
        0.5 * PAPER_PARAMETERS["domain_height_H"] + (y - 0.0)
    ) / (PAPER_PARAMETERS["domain_height_H"] ** 2)
    return float(np.trapezoid(inflow, y))


def build_jet_forcing_expression(component: str, q1_star: float, forcing_gain: float, shell_sigma: float, ramp_tau: float) -> str:
    assert component in ("u", "v")
    r = PAPER_PARAMETERS["radius_R"]
    omega = np.deg2rad(PAPER_PARAMETERS["jet_width_deg"])
    theta_1 = 0.5 * np.pi
    theta_2 = -0.5 * np.pi
    half_width = 0.5 * omega
    q_ref = compute_q_ref()
    q1 = q1_star * q_ref
    q2 = -q1
    amplitude_prefactor = np.pi / (2.0 * omega * (r ** 2))

    radial_component = "x / (jnp.sqrt(x**2 + y**2) + 1.0e-12)"
    if component == "v":
        radial_component = "y / (jnp.sqrt(x**2 + y**2) + 1.0e-12)"

    return (
        "lambda x,y,t: "
        f"{forcing_gain:.16g}"
        f" * (1.0 - jnp.exp(-t / {ramp_tau:.16g}))"
        f" * jnp.where(jnp.sqrt(x**2 + y**2) >= {r:.16g}, 1.0, 0.0)"
        f" * jnp.exp(-((jnp.sqrt(x**2 + y**2) - {r:.16g}) / {shell_sigma:.16g})**2)"
        " * ("
        f"({q1:.16g}) * ({amplitude_prefactor:.16g})"
        f" * jnp.where(jnp.abs(jnp.arctan2(y, x) - {theta_1:.16g}) <= {half_width:.16g},"
        f" jnp.cos(jnp.pi * (jnp.arctan2(y, x) - {theta_1:.16g}) / {omega:.16g}), 0.0)"
        " + "
        f"({q2:.16g}) * ({amplitude_prefactor:.16g})"
        f" * jnp.where(jnp.abs(jnp.arctan2(y, x) - {theta_2:.16g}) <= {half_width:.16g},"
        f" jnp.cos(jnp.pi * (jnp.arctan2(y, x) - {theta_2:.16g}) / {omega:.16g}), 0.0)"
        ")"
        f" * ({radial_component})"
    )


def build_case_setup(case_name: str, end_time: float, save_dt: float, q1_star: float) -> dict:
    rho = PAPER_PARAMETERS["density_rho"]
    u_ref = PAPER_PARAMETERS["mean_inflow_U_bar"]
    d = PAPER_PARAMETERS["diameter_D"]
    re = PAPER_PARAMETERS["reynolds_number"]
    mu = rho * u_ref * d / re
    inflow_u_expr = build_inflow_u_expression()

    jet_forcing_gain = 8.0
    jet_shell_sigma = 0.03
    jet_ramp_tau = 0.1

    forcings_dict = {}
    if abs(q1_star) > 0.0:
        forcings_dict["custom_forcing"] = {
            "rho": 0.0,
            "u": build_jet_forcing_expression(
                component="u",
                q1_star=q1_star,
                forcing_gain=jet_forcing_gain,
                shell_sigma=jet_shell_sigma,
                ramp_tau=jet_ramp_tau,
            ),
            "v": build_jet_forcing_expression(
                component="v",
                q1_star=q1_star,
                forcing_gain=jet_forcing_gain,
                shell_sigma=jet_shell_sigma,
                ramp_tau=jet_ramp_tau,
            ),
            "w": 0.0,
            "p": 0.0,
        }

    case_setup = {
        "general": {
            "case_name": case_name,
            "end_time": float(end_time),
            "save_path": "./results",
            "save_dt": float(save_dt),
        },
        "domain": {
            "x": {
                "cells": 440,
                "range": PAPER_PARAMETERS["domain_x_range"],
            },
            "y": {
                "cells": 82,
                "range": PAPER_PARAMETERS["domain_y_range"],
            },
            "z": {"cells": 1, "range": [0.0, 1.0]},
            "decomposition": {"split_x": 1, "split_y": 1, "split_z": 1},
        },
        "boundary_conditions": {
            "primitives": {
                "east": {"type": "ZEROGRADIENT"},
                "west": {
                    "type": "DIRICHLET",
                    "primitives_callable": {
                        "rho": 1.0,
                        "u": inflow_u_expr,
                        "v": 0.0,
                        "w": 0.0,
                        "p": 25.0,
                    },
                },
                "north": {
                    "type": "WALL",
                    "wall_velocity_callable": {"u": 0.0, "v": 0.0, "w": 0.0},
                },
                "south": {
                    "type": "WALL",
                    "wall_velocity_callable": {"u": 0.0, "v": 0.0, "w": 0.0},
                },
                "top": {"type": "INACTIVE"},
                "bottom": {"type": "INACTIVE"},
            },
            "levelset": {
                "east": {"type": "ZEROGRADIENT"},
                "west": {"type": "ZEROGRADIENT"},
                "north": {"type": "ZEROGRADIENT"},
                "south": {"type": "ZEROGRADIENT"},
                "top": {"type": "INACTIVE"},
                "bottom": {"type": "INACTIVE"},
            },
        },
        "initial_condition": {
            "primitives": {
                "rho": 1.0,
                "u": "lambda x,y: 6.0 * (2.05 - (y - 0.05)) * (2.05 + (y - 0.05)) / (4.1**2)",
                "v": 0.0,
                "w": 0.0,
                "p": 25.0,
            },
            "levelset": "lambda x,y: jnp.sqrt(x**2 + y**2) - 0.5",
        },
        "material_properties": {
            "equation_of_state": {
                "model": "IdealGas",
                "specific_heat_ratio": 1.4,
                "specific_gas_constant": 1.0,
            },
            "transport": {
                "dynamic_viscosity": {"model": "CUSTOM", "value": mu},
                "bulk_viscosity": 0.0,
                "thermal_conductivity": {"model": "CUSTOM", "value": 0.0},
            },
        },
        "forcings": forcings_dict,
        "nondimensionalization_parameters": {
            "density_reference": 1.0,
            "length_reference": 1.0,
            "velocity_reference": 1.0,
            "temperature_reference": 1.0,
        },
        "output": {
            "primitives": ["density", "velocity", "pressure"],
            "levelset": ["levelset", "volume_fraction"],
            "miscellaneous": ["vorticity"],
        },
    }
    return case_setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-star", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=8.0)
    parser.add_argument("--save-dt", type=float, default=0.2)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--make-xdmf-only", action="store_true")
    args = parser.parse_args()

    q_limit = PAPER_PARAMETERS["jet_q_star_limit_abs"]
    if abs(args.q1_star) > q_limit:
        raise ValueError(f"|q1_star| must be <= {q_limit}. Got {args.q1_star}.")

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    q_tag = f"{args.q1_star:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    case_name = f"rabault_jet_re100_nx440_ny82_q{q_tag}"
    case_setup = build_case_setup(
        case_name=case_name,
        end_time=args.end_time,
        save_dt=args.save_dt,
        q1_star=args.q1_star,
    )

    case_setup_file = script_dir / "_generated_case_setup.json"
    case_setup_file.write_text(json.dumps(case_setup, indent=2))
    numerical_setup = json.loads((script_dir / "numerical_setup.json").read_text())
    numerical_setup["active_forcings"]["is_custom_forcing"] = abs(args.q1_star) > 0.0
    numerical_setup["active_forcings"]["is_temperature_forcing"] = False
    numerical_setup_file = script_dir / "_generated_numerical_setup.json"
    numerical_setup_file.write_text(json.dumps(numerical_setup, indent=2))

    if args.skip_simulation:
        output_path = (script_dir / case_setup["general"]["save_path"] / case_setup["general"]["case_name"]).resolve()
    else:
        input_manager = InputManager(str(case_setup_file), str(numerical_setup_file))
        initialization_manager = InitializationManager(input_manager)
        simulation_manager = SimulationManager(input_manager)
        jxf_buffers = initialization_manager.initialization()
        simulation_manager.simulate(jxf_buffers)
        output_path = Path(simulation_manager.output_writer.save_path_case)

    domain_path = output_path / "domain"
    if args.make_xdmf_only:
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

    rho_ref = PAPER_PARAMETERS["density_rho"]
    u_ref = PAPER_PARAMETERS["mean_inflow_U_bar"]
    s_ref = PAPER_PARAMETERS["diameter_D"]
    mu = case_setup["material_properties"]["transport"]["dynamic_viscosity"]["value"]

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
        "q1_star": float(args.q1_star),
        "q2_star": float(-args.q1_star),
    }

    summary_path = output_path / "drag_lift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    parameters = dict(PAPER_PARAMETERS)
    parameters["q1_star"] = float(args.q1_star)
    parameters["q2_star"] = float(-args.q1_star)
    parameters["q_ref"] = float(compute_q_ref())
    parameters_path = output_path / "rabault_parameters.json"
    parameters_path.write_text(json.dumps(parameters, indent=2))

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
    print(f"Saved: {parameters_path}")
    print(f"Saved: {fig_path}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
