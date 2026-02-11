#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${PYTHON:=python}"
: "${SIM_TIME:=100.0}"
: "${NUM_ITERS:=8}"
: "${INNER_STEPS:=256}"
: "${LEARNING_RATE:=0.01}"
: "${MAX_STEP_NORM:=0.05}"
: "${RING_R_MIN:=0.7}"
: "${RING_R_MAX:=1.3}"
: "${BEZIER_SAMPLES:=180}"
: "${SEED:=0}"
: "${USE_WANDB:=0}"
: "${WANDB_PROJECT:=jaxfluids-viq-shape-opt}"
: "${WANDB_ENTITY:=}"
: "${WANDB_RUN_NAME:=}"
: "${WANDB_MODE:=online}"
: "${WANDB_TAGS:=viq,shape,opt,differentiable}"

: "${JAX_PLATFORMS:=cuda,cpu}"
: "${JAX_ENABLE_X64:=1}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
: "${XLA_PYTHON_CLIENT_ALLOCATOR:=platform}"
: "${CUDA_VISIBLE_DEVICES:=0}"

export JAX_PLATFORMS
export JAX_ENABLE_X64
export XLA_PYTHON_CLIENT_PREALLOCATE
export XLA_PYTHON_CLIENT_ALLOCATOR
export CUDA_VISIBLE_DEVICES

STAMP="$(date +%Y%m%d_%H%M%S)"
: "${OUT_DIR:=full_diff_opt_t100_${STAMP}}"

echo "Launching VIQ full differentiable optimization"
echo "  workdir: ${SCRIPT_DIR}"
echo "  output:  ${OUT_DIR}"
echo "  sim_time=${SIM_TIME}s num_iters=${NUM_ITERS} inner_steps=${INNER_STEPS}"
echo "  JAX_PLATFORMS=${JAX_PLATFORMS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  W&B: USE_WANDB=${USE_WANDB} PROJECT=${WANDB_PROJECT} MODE=${WANDB_MODE}"

"${PYTHON}" - <<'PY'
import jax
print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())
PY

CMD=(
  "${PYTHON}" "run_full_diff_optimization.py"
  "--case-setup" "case_setup_viq_re200.json"
  "--numerical-setup" "numerical_setup.json"
  "--output-dir" "${OUT_DIR}"
  "--sim-time" "${SIM_TIME}"
  "--num-iters" "${NUM_ITERS}"
  "--inner-steps" "${INNER_STEPS}"
  "--learning-rate" "${LEARNING_RATE}"
  "--max-step-norm" "${MAX_STEP_NORM}"
  "--ring-r-min" "${RING_R_MIN}"
  "--ring-r-max" "${RING_R_MAX}"
  "--bezier-samples" "${BEZIER_SAMPLES}"
  "--seed" "${SEED}"
  "--jit"
)

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

if [ "${USE_WANDB}" = "1" ]; then
  CMD+=(
    "--use-wandb"
    "--wandb-project" "${WANDB_PROJECT}"
    "--wandb-mode" "${WANDB_MODE}"
    "--wandb-tags" "${WANDB_TAGS}"
  )
  if [ -n "${WANDB_ENTITY}" ]; then
    CMD+=("--wandb-entity" "${WANDB_ENTITY}")
  fi
  if [ -n "${WANDB_RUN_NAME}" ]; then
    CMD+=("--wandb-run-name" "${WANDB_RUN_NAME}")
  fi
fi

echo "Command: ${CMD[*]}"
exec "${CMD[@]}"
