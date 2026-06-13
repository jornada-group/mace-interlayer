#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

: "${EXAMPLE_ROOT:=${WORKSPACE_ROOT}/mace-interlayer-example}"
: "${LOAD_TRAIN_ENV:=${WORKSPACE_ROOT}/load_train_env.sh}"
: "${RELAX_FMAX:=1e-6}"
: "${RELAX_STEPS:=50}"
: "${RELAX_DEVICE:=cuda}"
: "${RELAX_DEFAULT_DTYPE:=float64}"
: "${RELAX_RATTLE_STD:=0.0}"
: "${RELAX_RATTLE_SEED:=123}"
: "${RELAX_CPUS_PER_TASK:=32}"

export SLURM_CPU_BIND=cores

srun -N 1 -n 1 -c "${RELAX_CPUS_PER_TASK}" --gpus-per-task=1 \
  bash -lc "cd '${WORKSPACE_ROOT}' && source '${LOAD_TRAIN_ENV}' && cd '${REPO_ROOT}' && python examples/mose2_wse2_relax_smoke/relax_mose2_wse2.py --example-root '${EXAMPLE_ROOT}' --device '${RELAX_DEVICE}' --default-dtype '${RELAX_DEFAULT_DTYPE}' --fmax '${RELAX_FMAX}' --steps '${RELAX_STEPS}' --rattle-std '${RELAX_RATTLE_STD}' --rattle-seed '${RELAX_RATTLE_SEED}'"
