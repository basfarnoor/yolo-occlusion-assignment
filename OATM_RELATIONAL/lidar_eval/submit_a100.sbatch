#!/usr/bin/env bash
#SBATCH --job-name=oatm-lidar-eval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=lidar_eval/results/slurm-%x-%j.out
#SBATCH --error=lidar_eval/results/slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIRECTORY="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
cd "${PROJECT_DIRECTORY}"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing OATM_RELATIONAL/.venv. Create it on the login node with: uv sync --locked" >&2
    exit 2
fi

export UV_PROJECT_ENVIRONMENT="${PROJECT_DIRECTORY}/.venv"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project=${PROJECT_DIRECTORY}"
uv --version
uv run --frozen python --version
nvidia-smi

# Read-only preparation: creates CAM_FRONT frame metadata and privileged
# projected annotations. The latter are not opened by the tracking stage.
uv run --frozen python scripts/prepare_nuscenes.py

RUN_ID="nuscenes-mini-${SLURM_JOB_ID:-manual}"
RUN_DIRECTORY="lidar_eval/results/${RUN_ID}"
EVALUATION_ARGUMENTS=(
    --config lidar_eval/config.yaml
    --run-id "${RUN_ID}"
    --output-dir "${RUN_DIRECTORY}"
    --device cuda:0
)

# Default: validate and reuse the frozen detector cache. Submit with
# `sbatch --export=ALL,REFRESH_DETECTOR=1 ...` to rerun YOLO on the A100.
if [[ "${REFRESH_DETECTOR:-0}" == "1" ]]; then
    EVALUATION_ARGUMENTS+=(--refresh-detector)
fi

uv run --frozen python -m lidar_eval.run "${EVALUATION_ARGUMENTS[@]}"

echo "completed=${RUN_DIRECTORY}"
