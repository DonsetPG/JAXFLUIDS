#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def maybe_override_precision(numerical_setup: Dict[str, Any], precision_mode: str) -> Dict[str, Any]:
    out = copy.deepcopy(numerical_setup)
    if precision_mode == "single":
        out.setdefault("precision", {})["is_double_precision_compute"] = False
        out.setdefault("precision", {})["is_double_precision_output"] = False
    elif precision_mode == "double":
        out.setdefault("precision", {})["is_double_precision_compute"] = True
        out.setdefault("precision", {})["is_double_precision_output"] = True
    elif precision_mode != "from-setup":
        raise ValueError(f"Unknown precision mode: {precision_mode}")
    return out


def run_backend_worker(args: argparse.Namespace) -> int:
    # Imports are intentionally delayed so JAX backend selection from env is respected.
    import jax
    from jaxfluids import InitializationManager, InputManager, SimulationManager

    case_setup = load_json(Path(args.case_setup))
    numerical_setup = load_json(Path(args.numerical_setup))
    numerical_setup = maybe_override_precision(numerical_setup, args.precision)

    # Reduce benchmark noise from logging and output overhead.
    numerical_setup.setdefault("output", {}).setdefault("logging", {})["frequency"] = int(1e9)
    numerical_setup.setdefault("output", {})["is_xdmf"] = False

    output_root = Path(args.output_dir) / "raw_runs"
    output_root.mkdir(parents=True, exist_ok=True)

    def one_run(end_time: float, run_name: str) -> float:
        case_local = copy.deepcopy(case_setup)
        case_local["general"]["end_time"] = float(end_time)
        # Save once after end_time to avoid periodic write overhead.
        case_local["general"]["save_dt"] = float(end_time + 1.0)
        case_local["general"]["save_path"] = str(output_root)
        case_local["general"]["case_name"] = run_name

        input_manager = InputManager(case_local, numerical_setup)
        initialization_manager = InitializationManager(input_manager)
        simulation_manager = SimulationManager(input_manager)
        jxf_buffers = initialization_manager.initialization()

        t0 = time.perf_counter()
        simulation_manager.simulate(jxf_buffers)
        return time.perf_counter() - t0

    try:
        warmup_time = one_run(
            end_time=args.warmup_sim_time,
            run_name=f"bench_{args.backend_run}_warmup_t{args.warmup_sim_time:.3f}".replace(".", "p"),
        )

        measure_times: List[float] = []
        for rep in range(args.repeats):
            measure_times.append(
                one_run(
                    end_time=args.sim_time,
                    run_name=f"bench_{args.backend_run}_t{args.sim_time:.3f}_rep{rep:02d}".replace(".", "p"),
                )
            )

        result = {
            "success": True,
            "backend_requested": args.backend_run,
            "backend_actual": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "precision_mode": args.precision,
            "warmup_seconds": warmup_time,
            "measure_seconds": measure_times,
            "measure_mean_seconds": float(statistics.mean(measure_times)),
            "measure_std_seconds": float(statistics.pstdev(measure_times)) if len(measure_times) > 1 else 0.0,
            "sim_time_seconds": args.sim_time,
            "repeats": args.repeats,
        }
    except Exception as exc:  # pragma: no cover - benchmark error path
        result = {
            "success": False,
            "backend_requested": args.backend_run,
            "precision_mode": args.precision,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    print(json.dumps(result, indent=2))
    # Single-line sentinel for robust parent parsing despite noisy logs.
    print("RESULT_JSON:" + json.dumps(result))
    return 0 if result.get("success", False) else 2


def truncate_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def parse_worker_result(stdout: str, backend: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            payload = line[len("RESULT_JSON:") :].strip()
            try:
                return json.loads(payload)
            except Exception:
                break
    return {
        "success": False,
        "backend_requested": backend,
        "error": "Could not parse RESULT_JSON payload from worker subprocess.",
    }


def run_backend_subprocess(
    script_path: Path,
    base_args: argparse.Namespace,
    backend: str,
) -> Dict[str, Any]:
    cmd = [
        base_args.python_exec,
        str(script_path),
        "--backend-run",
        backend,
        "--case-setup",
        str(base_args.case_setup),
        "--numerical-setup",
        str(base_args.numerical_setup),
        "--output-dir",
        str(base_args.output_dir),
        "--sim-time",
        str(base_args.sim_time),
        "--warmup-sim-time",
        str(base_args.warmup_sim_time),
        "--repeats",
        str(base_args.repeats),
        "--precision",
        base_args.precision,
    ]
    env = os.environ.copy()
    if backend == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif backend == "gpu":
        env["JAX_PLATFORMS"] = "cuda,cpu"
        env.setdefault("CUDA_VISIBLE_DEVICES", base_args.cuda_visible_devices)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)

    parsed = parse_worker_result(proc.stdout, backend)

    parsed["worker_return_code"] = proc.returncode
    parsed["worker_stdout_tail"] = truncate_text(proc.stdout)
    parsed["worker_stderr_tail"] = truncate_text(proc.stderr)
    return parsed


def make_summary(cpu: Dict[str, Any], gpu: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "cpu": cpu,
        "gpu": gpu,
    }
    cpu_backend_ok = cpu.get("backend_actual") == "cpu"
    gpu_backend_ok = gpu.get("backend_actual") in ("gpu", "cuda")
    if cpu.get("success") and gpu.get("success") and cpu_backend_ok and gpu_backend_ok:
        cpu_t = float(cpu["measure_mean_seconds"])
        gpu_t = float(gpu["measure_mean_seconds"])
        summary["speedup_gpu_vs_cpu"] = cpu_t / gpu_t if gpu_t > 0 else None
        summary["cpu_mean_seconds"] = cpu_t
        summary["gpu_mean_seconds"] = gpu_t
        summary["gpu_faster"] = bool(gpu_t < cpu_t)
        summary["comparison_ok"] = True
    else:
        summary["comparison_ok"] = False
        summary["comparison_issue"] = {
            "cpu_success": cpu.get("success", False),
            "gpu_success": gpu.get("success", False),
            "cpu_backend_actual": cpu.get("backend_actual"),
            "gpu_backend_actual": gpu.get("backend_actual"),
            "cpu_backend_ok": cpu_backend_ok,
            "gpu_backend_ok": gpu_backend_ok,
        }
    return summary


def resolve_input_path(path_str: str, script_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    candidate = script_dir / p
    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VIQ case at 2s (or custom) on CPU vs GPU with isolated JAX backends."
    )
    parser.add_argument(
        "--case-setup",
        default="case_setup_viq_re200.json",
        help="Path to base case setup JSON.",
    )
    parser.add_argument(
        "--numerical-setup",
        default="numerical_setup.json",
        help="Path to numerical setup JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="bench_cpu_vs_gpu_viq",
        help="Directory to write benchmark outputs.",
    )
    parser.add_argument("--sim-time", type=float, default=2.0, help="Measured simulation end time.")
    parser.add_argument("--warmup-sim-time", type=float, default=0.2, help="Warmup simulation end time.")
    parser.add_argument("--repeats", type=int, default=2, help="Number of measured repeats per backend.")
    parser.add_argument(
        "--precision",
        choices=["from-setup", "single", "double"],
        default="single",
        help="Precision override applied to both CPU and GPU runs.",
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python interpreter for worker subprocesses.",
    )
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--backend-run",
        choices=["cpu", "gpu"],
        default=None,
        help="Internal worker mode. Do not set manually for normal use.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.sim_time <= 0.0:
        raise ValueError("--sim-time must be > 0")
    if args.warmup_sim_time <= 0.0:
        raise ValueError("--warmup-sim-time must be > 0")

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    args.case_setup = str(resolve_input_path(args.case_setup, script_dir))
    args.numerical_setup = str(resolve_input_path(args.numerical_setup, script_dir))
    args.output_dir = str(Path(args.output_dir).resolve())

    if args.backend_run is not None:
        return run_backend_worker(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Launching CPU benchmark worker...")
    cpu = run_backend_subprocess(script_path, args, "cpu")
    print("Launching GPU benchmark worker...")
    gpu = run_backend_subprocess(script_path, args, "gpu")

    summary = make_summary(cpu, gpu)
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nBenchmark summary:")
    if summary.get("comparison_ok"):
        print(
            f"CPU mean: {summary['cpu_mean_seconds']:.3f}s | "
            f"GPU mean: {summary['gpu_mean_seconds']:.3f}s | "
            f"speedup GPU vs CPU: {summary['speedup_gpu_vs_cpu']:.3f}x"
        )
    else:
        print("Comparison failed. Inspect benchmark_summary.json for backend errors.")
    print(f"Saved: {summary_path.resolve()}")
    return 0 if summary.get("comparison_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
