#!/bin/bash -l

#SBATCH --job-name=yolo_microgel
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%N.%j.out
#SBATCH --error=logs/%x.%N.%j.err

set -euo pipefail

PROJECT="${YOLO_PROJECT:-/mnt/fastscratch/users/$USER/yolo_project}"
YOLO_ENV="${YOLO_ENV:-/mnt/fastscratch/users/$USER/conda_envs/yolo}"

EPOCHS="${YOLO_EPOCHS:-1}"
IMGSZ="${YOLO_IMGSZ:-640}"
BATCH="${YOLO_BATCH:-2}"
WORKERS="${SLURM_CPUS_PER_TASK:-4}"

echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Job name:     ${SLURM_JOB_NAME}"
echo "Partition:    ${SLURM_JOB_PARTITION}"
echo "Node list:    ${SLURM_JOB_NODELIST}"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "========================================"

module purge
module load miniforge3/25.3.0-python3.12.10-dynamic

eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$WORKERS"
export MPLBACKEND=Agg

echo "Python executable: $(which python)"
python --version

echo "PyTorch and allocated GPU:"
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("Built CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Compiled architectures:", torch.cuda.get_arch_list())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: Slurm allocated the job, but PyTorch cannot access a GPU")

print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
PY

echo "NVIDIA status:"
nvidia-smi

cd "$PROJECT"

if [[ -n "${YOLO_RESUME:-}" ]]; then
    echo "Resuming from: $YOLO_RESUME"
    python -u train.py --resume "$YOLO_RESUME"
else
    echo "Starting a new training run"
    echo "Epochs:  $EPOCHS"
    echo "Image:   $IMGSZ"
    echo "Batch:   $BATCH"
    echo "Workers: $WORKERS"

    python -u train.py \
        --epochs "$EPOCHS" \
        --imgsz "$IMGSZ" \
        --batch "$BATCH" \
        --workers "$WORKERS"
fi

echo "========================================"
echo "Finish time: $(date)"
echo "Training command completed"
echo "========================================"
