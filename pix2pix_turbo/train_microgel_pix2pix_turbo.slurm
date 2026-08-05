#!/bin/bash -l

#SBATCH -D /mnt/fastscratch/users/sgzjia25/microgel-pix2pix-turbo
#SBATCH --export=ALL
#SBATCH -J microgel-pix2pix-continue
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


IMG2IMG_TURBO_SRC=/mnt/fastscratch/users/sgzjia25/img2img-turbo/src
export PYTHONPATH="$IMG2IMG_TURBO_SRC"


WORK_ROOT=/mnt/fastscratch/users/sgzjia25/microgel-pix2pix-turbo

TRAIN_SCRIPT="$WORK_ROOT/train_pix2pix_turbo_continue.py"
DATASET_ROOT="$WORK_ROOT/data"
OUTPUT_ROOT="$WORK_ROOT/output/microgel_pix2pix_turbo"

RESUME_CHECKPOINT="$OUTPUT_ROOT/checkpoints/model_8001.pkl"
FINAL_CHECKPOINT="$OUTPUT_ROOT/checkpoints/model_13000.pkl"

# model_8001 + 4999 = model_13000
ADDITIONAL_STEPS=4999
CONTINUATION_EPOCHS=11


if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Continuation training script does not exist:"
    echo "$TRAIN_SCRIPT"
    exit 1
fi

if [[ ! -f "$IMG2IMG_TURBO_SRC/pix2pix_turbo.py" ]]; then
    echo "ERROR: Missing Python module:"
    echo "$IMG2IMG_TURBO_SRC/pix2pix_turbo.py"
    exit 1
fi

if [[ ! -f "$IMG2IMG_TURBO_SRC/model.py" ]]; then
    echo "ERROR: Missing Python module:"
    echo "$IMG2IMG_TURBO_SRC/model.py"
    exit 1
fi

if [[ ! -f "$IMG2IMG_TURBO_SRC/my_utils/training_utils.py" ]]; then
    echo "ERROR: Missing Python module:"
    echo "$IMG2IMG_TURBO_SRC/my_utils/training_utils.py"
    exit 1
fi

if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "ERROR: Resume checkpoint does not exist:"
    echo "$RESUME_CHECKPOINT"
    exit 1
fi

for directory in \
    "$DATASET_ROOT/train_A" \
    "$DATASET_ROOT/train_B" \
    "$DATASET_ROOT/test_A" \
    "$DATASET_ROOT/test_B"
do
    if [[ ! -d "$directory" ]]; then
        echo "ERROR: Dataset directory does not exist:"
        echo "$directory"
        exit 1
    fi
done

for prompt_file in \
    "$DATASET_ROOT/train_prompts.json" \
    "$DATASET_ROOT/test_prompts.json"
do
    if [[ ! -f "$prompt_file" ]]; then
        echo "ERROR: Prompt file does not exist:"
        echo "$prompt_file"
        exit 1
    fi
done

export HF_HOME="$WORK_ROOT/.cache/huggingface"
export TORCH_HOME="$WORK_ROOT/.cache/torch"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"

export WANDB_DIR="$WORK_ROOT/wandb"
export WANDB_MODE=offline

mkdir -p \
    "$HF_HOME" \
    "$TORCH_HOME" \
    "$WANDB_DIR" \
    "$OUTPUT_ROOT" \
    "$OUTPUT_ROOT/checkpoints" \
    "$OUTPUT_ROOT/eval"

# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Starting pix2pix-turbo continuation training"
echo "============================================================"
echo "Training script:       $TRAIN_SCRIPT"
echo "Python module path:    $IMG2IMG_TURBO_SRC"
echo "Resume checkpoint:     $RESUME_CHECKPOINT"
echo "Dataset:               $DATASET_ROOT"
echo "Output:                $OUTPUT_ROOT"
echo "Checkpoint step:       8001"
echo "Additional steps:      $ADDITIONAL_STEPS"
echo "Expected final step:   13000"
echo "Continuation epochs:   $CONTINUATION_EPOCHS"
echo "Batch size:            2"
echo "Learning rate:         5e-6"
echo "L2 weight:             2.0"
echo "LPIPS weight:          5.0"
echo "GAN weight:            0.5"
echo "CLIP weight:           0.0"
echo "============================================================"

# ---------------------------------------------------------------------------
# Continue training
# ---------------------------------------------------------------------------

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    "$TRAIN_SCRIPT" \
    --pretrained_model_name_or_path "stabilityai/sd-turbo" \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" \
    --dataset_folder "$DATASET_ROOT" \
    --output_dir "$OUTPUT_ROOT" \
    --resolution 512 \
    --train_image_prep no_resize \
    --test_image_prep no_resize \
    --train_batch_size 2 \
    --num_training_epochs "$CONTINUATION_EPOCHS" \
    --max_train_steps "$ADDITIONAL_STEPS" \
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

# ---------------------------------------------------------------------------
# Verify final checkpoint
# ---------------------------------------------------------------------------

echo "============================================================"

if [[ -f "$FINAL_CHECKPOINT" ]]; then
    echo "Training completed successfully"
    echo "Additional steps: 4999"
    echo "Final global step: 13000"
    echo "Final checkpoint: $FINAL_CHECKPOINT"
    echo "W&B offline data: $WANDB_DIR"
else
    echo "ERROR: Expected final checkpoint was not found:"
    echo "$FINAL_CHECKPOINT"
    echo
    echo "Available recent checkpoints:"

    find "$OUTPUT_ROOT/checkpoints" \
        -maxdepth 1 \
        -type f \
        -name "model_*.pkl" \
        -print \
        | sort -V \
        | tail -10

    exit 1
fi

echo "============================================================"