#!/bin/bash -l
#SBATCH --chdir=/mnt/fastscratch/users/sgzjia25/yolo_train
#SBATCH --export=ALL
#SBATCH --job-name=yolo-threshold-all
#SBATCH --output=log.%N.%j.out
#SBATCH --error=log.%N.%j.err
#SBATCH --partition=cpu-l40s-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH

ulimit -s unlimited

module purge
module load cuda/12.8.0-gcc14.2.0

source /mnt/fastscratch/users/sgzjia25/yolo_env/bin/activate

PYTHON=/mnt/fastscratch/users/sgzjia25/yolo_env/bin/python
WORKDIR=/mnt/fastscratch/users/sgzjia25/yolo_train
SCRIPT="$WORKDIR/find_best_yolo_thresholds_all_models.py"
MODEL_ROOT="$WORKDIR"
DATA_YAML="microgel_dataset_clean/data_new.yaml"
RESULTS_DIR="$WORKDIR/threshold_search_results"
SUMMARY_CSV="$RESULTS_DIR/best_thresholds_summary.csv"

module list

echo "========================================================="
echo "SLURM job started: $(date)"
date_start=$(date +%s)
hostname
echo "Current directory: $(pwd)"

echo "Testing Python and GPU environment:"
"$PYTHON" -c "
import torch
import matplotlib
import pyparsing
from ultralytics import YOLO

print('Environment OK')
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo "Models that will be searched:"
find "$MODEL_ROOT" -type f -path '*/weights/best.pt' -print | sort

echo "Running sequential threshold search for every discovered model"

srun "$PYTHON" "$SCRIPT" \
  --model-root "$MODEL_ROOT" \
  --data "$DATA_YAML" \
  --split val \
  --imgsz 1920 \
  --conf 0.05:0.80:0.05 \
  --iou 0.30:0.90:0.05 \
  --match-iou 0.50 \
  --map-conf 0.001 \
  --max-det 1000 \
  --device 0 \
  --batch 1 \
  --selection-metric map50_95 \
  --duplicate-weight 0.25 \
  --output-dir "$RESULTS_DIR" \
  --summary-out "$SUMMARY_CSV" \
  --resume \
  --continue-on-error

# To run only selected models, add one repeated option per model, for example:
#   --model microgel_labeled_tune-2 \
#   --model microgel_tune_reflection_bg_1920-10

date_end=$(date +%s)
elapsed=$((date_end - date_start))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo "========================================================="
echo "SLURM job finished: $(date)"
echo "Total run time: ${hours} Hours ${minutes} Minutes ${seconds} Seconds"
echo "Best-parameter summary: $SUMMARY_CSV"
echo "========================================================="
