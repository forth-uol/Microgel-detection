#!/bin/bash -l

#SBATCH -D /mnt/fastscratch/users/sgzjia25/microgel-pix2pix-turbo
#SBATCH --export=ALL
#SBATCH -J microgel-pix2pix-turbo
#SBATCH -o log.%N.%j.out
#SBATCH -p gpu-a-lowsmall
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00

set -euo pipefail

module purge
module load cuda/12.8.0-gcc14.2.0

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

source /mnt/fastscratch/users/sgzjia25/img2img_turbo_env/bin/activate

WORK_ROOT=/mnt/fastscratch/users/sgzjia25/microgel-pix2pix-turbo
TRAIN_SCRIPT=/mnt/fastscratch/users/sgzjia25/img2img-turbo/src/train_pix2pix_turbo.py
DATASET_ROOT="$WORK_ROOT/data"
OUTPUT_ROOT="$WORK_ROOT/output/microgel_pix2pix_turbo"

# Check that the training script exists.
test -f "$TRAIN_SCRIPT"

# Check that all dataset directories exist.
test -d "$DATASET_ROOT/train_A"
test -d "$DATASET_ROOT/train_B"
test -d "$DATASET_ROOT/test_A"
test -d "$DATASET_ROOT/test_B"

# Check that prompt files exist.
test -f "$DATASET_ROOT/train_prompts.json"
test -f "$DATASET_ROOT/test_prompts.json"

# Cache directories.
export HF_HOME="$WORK_ROOT/.cache/huggingface"
export TORCH_HOME="$WORK_ROOT/.cache/torch"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"

# Save Weights & Biases data locally.
export WANDB_DIR="$WORK_ROOT/wandb"
export WANDB_MODE=offline

mkdir -p \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$WANDB_DIR" \
  "$OUTPUT_ROOT"

# ---------------------------------------------------------------------------
# Training configuration
#
# Dataset: 945 paired training samples
# Batch size: 2
# Steps per epoch: ceil(945 / 2) = 473
#
# Requested optimizer steps: 8000
# Equivalent epochs: 8000 / 473 = approximately 16.91
# Epoch limit: 17
#
# Training stops exactly at max_train_steps=8000.
#
# Loss weights:
#   L2      = 2.0  protects paired structure and particle alignment
#   LPIPS   = 5.0  encourages perceptual texture and edge similarity
#   GAN     = 0.5  encourages realistic generated appearance
#   CLIP    = 0.0  disables text supervision
# ---------------------------------------------------------------------------

echo "Starting pix2pix-turbo training"
echo "Training script: $TRAIN_SCRIPT"
echo "Dataset:         $DATASET_ROOT"
echo "Output:          $OUTPUT_ROOT"
echo "Epoch limit:     17"
echo "Training steps:  8000"
echo "Batch size:      2"
echo "Checkpoints:     every 500 steps"
echo "Evaluation:      every 250 steps"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path "stabilityai/sd-turbo" \
  --dataset_folder "$DATASET_ROOT" \
  --output_dir "$OUTPUT_ROOT" \
  --resolution 512 \
  --train_image_prep no_resize \
  --test_image_prep no_resize \
  --train_batch_size 2 \
  --num_training_epochs 17 \
  --max_train_steps 8000 \
  --learning_rate 5e-6 \
  --checkpointing_steps 500 \
  --eval_freq 250 \
  --num_samples_eval 18 \
  --viz_freq 100 \
  --lambda_l2 2.0 \
  --lambda_lpips 5.0 \
  --lambda_gan 0.5 \
  --lambda_clipsim 0.0 \
  --mixed_precision no \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --allow_tf32 \
  --set_grads_to_none \
  --dataloader_num_workers 4 \
  --seed 42 \
  --report_to wandb \
  --tracker_project_name microgel_pix2pix_turbo

echo "Training completed successfully"
echo "Completed optimizer steps: 8000"
echo "Results saved to: $OUTPUT_ROOT"
echo "W&B offline data saved to: $WANDB_DIR"