#!/bin/bash
#SBATCH -J ft_goal_frame_delay
#SBATCH -p i64m1tga800u
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH -o /hpc2hdd/home/tzhuang778/daojie/openvla-oft/logs/finetune_goal_frame_delay_%j.out
#SBATCH -e /hpc2hdd/home/tzhuang778/daojie/openvla-oft/logs/finetune_goal_frame_delay_%j.err
#SBATCH -D ./

# ==============================================================================
# Training: Frame Delay + VisionActionHead for async deployment
#
# Method: VLA runs on both delayed and current frames each step.
#   - delayed frame → stale VLA latents (simulates cloud LLM processing old frame)
#   - current frame → fresh VLA latents
#   - VisionActionHead fuses either latent with real-time vision → both predict gt actions
#   - Loss = L1(action_fresh, gt) + L1(action_stale, gt)
#
# This teaches VisionActionHead to correct stale latents using real-time vision.
# ==============================================================================

# Environment setup
module load cuda/12.8
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla-oft

export HF_HOME="/hpc2hdd/home/tzhuang778/daojie/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create logs directory if it doesn't exist
mkdir -p logs

# Training configuration
VLA_PATH="openvla/openvla-7b"
DATA_ROOT_DIR="/hpc2hdd/home/tzhuang778/daojie/modified_libero_rlds"
DATASET_NAME="libero_goal_no_noops"
RUN_ROOT_DIR="/hpc2hdd/home/tzhuang778/daojie/openvla-oft/runs"

# VisionActionHead Configuration
USE_VISION_ACTION_HEAD=true
ACTION_HEAD_VISION_ENCODER="siglip-base"
FREEZE_ACTION_HEAD_VISION=true
ACTION_HEAD_NUM_VIEWS=2

# Frame Delay Configuration
# window_size controls how many past frames the dataset provides per sample.
# Set window_size = max_delay + 1 (e.g. 11 for max delay of 10 env steps).
USE_FRAME_DELAY=true
WINDOW_SIZE=21

# Training hyperparameters
BATCH_SIZE=8
LEARNING_RATE=0.0005
LORA_RANK=16
MAX_STEPS=200000
NUM_STEPS_BEFORE_DECAY=100000
SAVE_FREQ=10000
NUM_IMAGES=2
USE_PROPRIO=true

echo "============================================================"
echo "Frame Delay + VisionActionHead Training"
echo "============================================================"
echo "Dataset: ${DATASET_NAME}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "LoRA Rank: ${LORA_RANK}"
echo "VisionActionHead: ${USE_VISION_ACTION_HEAD}"
echo "Vision Encoder: ${ACTION_HEAD_VISION_ENCODER}"
echo "Num Views: ${ACTION_HEAD_NUM_VIEWS}"
echo "Frame Delay: ${USE_FRAME_DELAY}"
echo "Window Size: ${WINDOW_SIZE}"
echo "============================================================"

echo "Started at: $(date)"

MASTER_PORT=$((20000 + RANDOM % 10000))
echo "Using master port: ${MASTER_PORT}"

torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} vla-scripts/finetune.py \
    --vla_path "${VLA_PATH}" \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --lora_rank ${LORA_RANK} \
    --max_steps ${MAX_STEPS} \
    --num_steps_before_decay ${NUM_STEPS_BEFORE_DECAY} \
    --save_freq ${SAVE_FREQ} \
    --num_images_in_input ${NUM_IMAGES} \
    --use_proprio ${USE_PROPRIO} \
    --use_l1_regression true \
    --image_aug true \
    --use_lora true \
    --lora_dropout 0.0 \
    --save_latest_checkpoint_only false \
    --wandb_entity "pengdaojie-the-hong-kong-university-of-science-and-techn" \
    --wandb_project "openvla-frame-delay" \
    --run_id_note "frame_delay_w${WINDOW_SIZE}_visionAH" \
    --use_vision_action_head ${USE_VISION_ACTION_HEAD} \
    --action_head_vision_encoder ${ACTION_HEAD_VISION_ENCODER} \
    --freeze_action_head_vision ${FREEZE_ACTION_HEAD_VISION} \
    --action_head_num_views ${ACTION_HEAD_NUM_VIEWS} \
    --use_frame_delay ${USE_FRAME_DELAY} \
    --window_size ${WINDOW_SIZE}

echo "=============================================="
echo "Training completed at $(date)!"
echo "=============================================="
