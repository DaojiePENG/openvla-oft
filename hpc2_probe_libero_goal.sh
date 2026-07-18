#!/bin/bash
#SBATCH -J probe_libero_goal
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -o /hpc2hdd/home/tzhuang778/daojie/openvla-oft/logs/probe_libero_goal_%j.out
#SBATCH -e /hpc2hdd/home/tzhuang778/daojie/openvla-oft/logs/probe_libero_goal_%j.err
#SBATCH -D ./

# ==============================================================================
# Representational Probe Analysis: Single Suite (libero_goal)
# Linear probes on VLA backbone hidden states to measure information encoding.
# ==============================================================================

# Environment setup
module load cuda/12.8
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla-oft

export HF_HOME="/hpc2hdd/home/tzhuang778/daojie/huggingface"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results

echo "============================================================"
echo "Representational Probe Analysis — libero_goal (Job ID: ${SLURM_JOB_ID})"
echo "Started at: $(date)"
echo "============================================================"
echo "GPU Info:"
nvidia-smi | grep -A 5 "GPU 0"
echo "============================================================"

##########################################################
# Configuration
##########################################################

BASE_DIR="/hpc2hdd/home/tzhuang778/daojie/openvla-oft"

# Checkpoints
CKPT_STANDARD="${BASE_DIR}/data/openvla-7b-oft-finetuned-libero-goal"
CKPT_OURS="${BASE_DIR}/runs/openvla-7b+libero_goal_no_noops+b8+lr-0.0005+lora-r16+dropout-0.0--image_aug--frame_delay_w21_visionAH--200000_chkpt"

# Verify checkpoints exist
if [ ! -d "$CKPT_STANDARD" ]; then
    echo "ERROR: Standard checkpoint not found: ${CKPT_STANDARD}"
    exit 1
fi
if [ ! -d "$CKPT_OURS" ]; then
    echo "ERROR: CloudEdgeVLA checkpoint not found: ${CKPT_OURS}"
    exit 1
fi

echo "Standard VLA checkpoint : ${CKPT_STANDARD}"
echo "CloudEdgeVLA checkpoint : ${CKPT_OURS}"
echo ""

##########################################################
# Run probe analysis
##########################################################

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting probe analysis..."

python scripts/visualize_representational_probe.py \
    --checkpoint_standard "${CKPT_STANDARD}" \
    --checkpoint_ours     "${CKPT_OURS}" \
    --task_suite_name     libero_goal \
    --lora_rank           16 \
    --num_episodes        5 \
    --num_frames          60 \
    --output_path         results/fig5_probe_goal.pdf \
    --seed                42

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Probe analysis completed successfully!"
    echo "  Output: results/fig5_probe_goal.pdf"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Probe analysis FAILED (exit code: ${EXIT_CODE})"
fi
echo "  Log: logs/probe_libero_goal_${SLURM_JOB_ID}.out"
echo "============================================================"

exit $EXIT_CODE
