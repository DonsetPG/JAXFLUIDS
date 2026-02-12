#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${PYTHON:=python}"
: "${SIM_TIME:=2.0}"
: "${WARMUP_SIM_TIME:=0.2}"
: "${REPEATS:=2}"
: "${PRECISION:=single}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${OUTPUT_DIR:=bench_cpu_vs_gpu_viq_$(date +%Y%m%d_%H%M%S)}"

# Reasonable defaults for stable GPU benchmarking.
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
: "${XLA_PYTHON_CLIENT_ALLOCATOR:=platform}"
: "${JAX_ENABLE_X64:=0}"

export CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_PREALLOCATE
export XLA_PYTHON_CLIENT_ALLOCATOR
export JAX_ENABLE_X64

echo "Benchmark CPU vs GPU (VIQ)"
echo "  workdir: ${SCRIPT_DIR}"
echo "  output:  ${OUTPUT_DIR}"
echo "  sim_time=${SIM_TIME}s warmup=${WARMUP_SIM_TIME}s repeats=${REPEATS} precision=${PRECISION}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

CMD=(
  "${PYTHON}" "benchmark_cpu_gpu_viq.py"
  "--case-setup" "case_setup_viq_re200.json"
  "--numerical-setup" "numerical_setup.json"
  "--output-dir" "${OUTPUT_DIR}"
  "--sim-time" "${SIM_TIME}"
  "--warmup-sim-time" "${WARMUP_SIM_TIME}"
  "--repeats" "${REPEATS}"
  "--precision" "${PRECISION}"
  "--cuda-visible-devices" "${CUDA_VISIBLE_DEVICES}"
)

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

echo "Command: ${CMD[*]}"
exec "${CMD[@]}"
