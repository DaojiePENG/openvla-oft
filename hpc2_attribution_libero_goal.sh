#!/bin/bash
#SBATCH -J attr_libero_goal
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -o /hpc2hdd/home/tzhuang778/daojie/openvla-oft/logs/attr_libero_goal_%j.out
#SBATCH -e /hpc2hdd/home/tzhuang778/daojie/openvla-oft/logs/attr_libero_goal_%j.err
#SBATCH -D ./

# ==============================================================================
# Action-Head Attribution Analysis: Single Suite (libero_goal)
# Measures effective α(d) — how much the VisionActionHead relies on cloud
# planning features vs. edge vision features under varying delay.
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
echo "Action-Head Attribution — libero_goal (Job ID: ${SLURM_JOB_ID})"
echo "Started at: $(date)"
echo "============================================================"
echo "GPU Info:"
nvidia-smi | grep -A 5 "GPU 0"
echo "============================================================"

##########################################################
# Configuration
##########################################################

BASE_DIR="/hpc2hdd/home/tzhuang778/daojie/openvla-oft"

# CloudEdgeVLA checkpoint (with VisionActionHead)
CKPT="${BASE_DIR}/runs/openvla-7b+libero_goal_no_noops+b8+lr-0.0005+lora-r16+dropout-0.0--image_aug--frame_delay_w21_visionAH--200000_chkpt"

# Verify checkpoint exists
if [ ! -d "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi

echo "Checkpoint: ${CKPT}"
echo ""

##########################################################
# Run attribution analysis
##########################################################

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting attribution analysis..."

# --- Gradient method (original) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running gradient attribution..."
python scripts/visualize_action_head_attribution.py \
    --pretrained_checkpoint "${CKPT}" \
    --task_suite_name       libero_goal \
    --method                gradient \
    --lora_rank             16 \
    --num_episodes          3 \
    --num_frames            40 \
    --output_path           results/fig6_attribution_goal_gradient.pdf

GRAD_EXIT=$?

echo ""
# --- Ablation method (scale-free, for cross-validation) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ablation attribution..."
python scripts/visualize_action_head_attribution.py \
    --pretrained_checkpoint "${CKPT}" \
    --task_suite_name       libero_goal \
    --method                ablation \
    --lora_rank             16 \
    --num_episodes          3 \
    --num_frames            40 \
    --output_path           results/fig6_attribution_goal_ablation.pdf

ABLT_EXIT=$?

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Results summary:"
echo "  Gradient : exit=${GRAD_EXIT} → results/fig6_attribution_goal_gradient.pdf"
echo "  Ablation : exit=${ABLT_EXIT} → results/fig6_attribution_goal_ablation.pdf"
echo "  Log: logs/attr_libero_goal_${SLURM_JOB_ID}.out"
echo "============================================================"

exit $((GRAD_EXIT + ABLT_EXIT))
