#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${PYTHON:=python}"
: "${SIM_TIME:=100.0}"
: "${NUM_ITERS:=8}"
: "${PARALLEL_SIMS:=2}"
: "${INIT_NOISE:=0.05}"
: "${INNER_STEPS:=1024}"
: "${LEARNING_RATE:=0.01}"
: "${MAX_STEP_NORM:=0.05}"
: "${RING_R_MIN:=0.7}"
: "${RING_R_MAX:=1.3}"
: "${BEZIER_SAMPLES:=180}"
: "${SEED:=0}"
: "${SOLVER_PRECISION:=single}"
: "${THETA_DTYPE:=float32}"
: "${CHECKPOINT_INTEGRATION_STEP:=0}"
: "${CHECKPOINT_INNER_STEP:=1}"
: "${SAVE_EVERY:=4}"
: "${SKIP_SHAPE_ARTIFACTS:=1}"

: "${USE_WANDB:=0}"
: "${WANDB_PROJECT:=jaxfluids-viq-shape-opt}"
: "${WANDB_ENTITY:=}"
: "${WANDB_RUN_NAME:=}"
: "${WANDB_MODE:=online}"
: "${WANDB_TAGS:=viq,shape,opt,differentiable,tpu}"

: "${JAX_PLATFORMS:=tpu,cpu}"
: "${JAX_ENABLE_X64:=0}"

export JAX_PLATFORMS
export JAX_ENABLE_X64

STAMP="$(date +%Y%m%d_%H%M%S)"
: "${OUT_DIR:=full_diff_opt_tpu_fast_t100_${STAMP}}"

echo "Launching VIQ full differentiable optimization (TPU fast profile)"
echo "  workdir: ${SCRIPT_DIR}"
echo "  output:  ${OUT_DIR}"
echo "  sim_time=${SIM_TIME}s num_iters=${NUM_ITERS} parallel_sims=${PARALLEL_SIMS} inner_steps=${INNER_STEPS}"
echo "  solver_precision=${SOLVER_PRECISION} theta_dtype=${THETA_DTYPE}"
echo "  checkpoints: inner=${CHECKPOINT_INNER_STEP} integration=${CHECKPOINT_INTEGRATION_STEP}"
echo "  save_every=${SAVE_EVERY} skip_shape_artifacts=${SKIP_SHAPE_ARTIFACTS}"
echo "  JAX_PLATFORMS=${JAX_PLATFORMS}"
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
  "--parallel-sims" "${PARALLEL_SIMS}"
  "--init-noise" "${INIT_NOISE}"
  "--inner-steps" "${INNER_STEPS}"
  "--learning-rate" "${LEARNING_RATE}"
  "--max-step-norm" "${MAX_STEP_NORM}"
  "--ring-r-min" "${RING_R_MIN}"
  "--ring-r-max" "${RING_R_MAX}"
  "--bezier-samples" "${BEZIER_SAMPLES}"
  "--seed" "${SEED}"
  "--solver-precision" "${SOLVER_PRECISION}"
  "--theta-dtype" "${THETA_DTYPE}"
  "--save-every" "${SAVE_EVERY}"
  "--jit"
)

if [ "${CHECKPOINT_INNER_STEP}" = "1" ]; then
  CMD+=("--checkpoint-inner-step")
else
  CMD+=("--no-checkpoint-inner-step")
fi

if [ "${CHECKPOINT_INTEGRATION_STEP}" = "1" ]; then
  CMD+=("--checkpoint-integration-step")
fi

if [ "${SKIP_SHAPE_ARTIFACTS}" = "1" ]; then
  CMD+=("--skip-shape-artifacts")
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

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

echo "Command: ${CMD[*]}"
exec "${CMD[@]}"
