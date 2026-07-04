#!/bin/bash
#SBATCH -J eval_frame_delay
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH -o /hpc2hdd/home/tzhuang778/daojie/openvla-oft/logs/eval_frame_delay_%j.out
#SBATCH -e /hpc2hdd/home/tzhuang778/daojie/openvla-oft/logs/eval_frame_delay_%j.err
#SBATCH -D ./

# ==============================================================================
# Evaluation: Frame Delay + VisionActionHead on LIBERO
# Tests multiple checkpoints × delay configs, staggered launches on 1 GPU
# ==============================================================================

# Environment setup
module load cuda/12.8
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla-oft

export HF_HOME="/hpc2hdd/home/tzhuang778/daojie/huggingface"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

echo "============================================================"
echo "Frame Delay + VisionActionHead Evaluation (Job ID: ${SLURM_JOB_ID})"
echo "Started at: $(date)"
echo "============================================================"
echo "GPU Info:"
nvidia-smi | grep -A 5 "GPU 0"
echo "============================================================"

##########################################################
# Configuration
##########################################################

EVAL_SCRIPT="experiments/robot/libero/run_libero_eval.py"

# Checkpoint root directory
CKPT_ROOT="/hpc2hdd/home/tzhuang778/daojie/openvla-oft/runs"

# Common model parameters (MUST match training config)
COMMON_ARGS="--use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --lora_rank 16 \
    --center_crop True \
    --num_trials_per_task 50 \
    --use_vision_action_head True \
    --action_head_vision_encoder siglip-base \
    --freeze_action_head_vision True \
    --action_head_num_views 2"

# Staggered start delay (seconds) between launches
DELAY=90

##########################################################
# Task definitions: CHECKPOINT | LOG_PREFIX | EXTRA_ARGS
#
# Each task is one checkpoint × one delay config.
# Add/remove tasks as needed.
#
# Delay configs:
#   --use_frame_delay_eval false                          (baseline, no delay)
#   --use_frame_delay_eval true --max_delay_steps_eval 5  (uniform delay ≤5)
#   --use_frame_delay_eval true --max_delay_steps_eval 10 (uniform delay ≤10)
##########################################################

declare -a TASKS=(
    # ============================================
    # Baseline (no delay) - seed 7
    # ============================================
    "${CKPT_ROOT}/YOUR_CHECKPOINT--100000_chkpt\
    :baseline_d0_seed07\
    :--task_suite_name libero_goal --seed 7 \
    --use_frame_delay_eval false \
    --run_id_note baseline_d0_seed07"

    # ============================================
    # Uniform delay, max 5 steps - seed 7
    # ============================================
    "${CKPT_ROOT}/YOUR_CHECKPOINT--100000_chkpt\
    :frame_delay_d5_seed07\
    :--task_suite_name libero_goal --seed 7 \
    --use_frame_delay_eval true --max_delay_steps_eval 5 \
    --run_id_note frame_delay_d5_seed07"

    # ============================================
    # Uniform delay, max 10 steps - seed 7
    # ============================================
    "${CKPT_ROOT}/YOUR_CHECKPOINT--100000_chkpt\
    :frame_delay_d10_seed07\
    :--task_suite_name libero_goal --seed 7 \
    --use_frame_delay_eval true --max_delay_steps_eval 10 \
    --run_id_note frame_delay_d10_seed07"

    # ============================================
    # Uniform delay, max 15 steps - seed 7
    # ============================================
    "${CKPT_ROOT}/YOUR_CHECKPOINT--100000_chkpt\
    :frame_delay_d15_seed07\
    :--task_suite_name libero_goal --seed 7 \
    --use_frame_delay_eval true --max_delay_steps_eval 15 \
    --run_id_note frame_delay_d15_seed07"
)

##########################################################
# Launch tasks with staggered starts
##########################################################
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching ${#TASKS[@]} evaluation tasks..."

declare -a PIDS=()

for INDEX in "${!TASKS[@]}"; do
    TASK="${TASKS[$INDEX]}"
    CHECKPOINT=$(echo "$TASK" | cut -d ':' -f 1)
    LOG_PREFIX=$(echo "$TASK" | cut -d ':' -f 2)
    EXTRA_ARGS=$(echo "$TASK" | cut -d ':' -f 3)
    TASK_LOG="logs/T${SLURM_JOB_ID}_${LOG_PREFIX}_eval.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $((INDEX+1))/${#TASKS[@]}: ${LOG_PREFIX}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Checkpoint: ${CHECKPOINT}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Log: ${TASK_LOG}"

    # Verify checkpoint exists
    if [ ! -d "$CHECKPOINT" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Checkpoint not found: ${CHECKPOINT}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Skipping task ${LOG_PREFIX}..."
        continue
    fi

    # Launch task in background
    python "$EVAL_SCRIPT" \
        --pretrained_checkpoint "$CHECKPOINT" \
        $COMMON_ARGS \
        $EXTRA_ARGS \
        > "$TASK_LOG" 2>&1 &

    TASK_PID=$!
    PIDS+=("$TASK_PID")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   PID: ${TASK_PID}"

    # Staggered start (skip delay after last task)
    if [ $INDEX -ne $(( ${#TASKS[@]} - 1 )) ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Waiting ${DELAY}s before next launch..."
        sleep $DELAY
    fi
done

##########################################################
# Wait for all tasks and collect results
##########################################################
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All tasks launched. Waiting for completion..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

FAILED=0
SUCCEEDED=0

for INDEX in "${!PIDS[@]}"; do
    PID="${PIDS[$INDEX]}"
    LOG_PREFIX=$(echo "${TASKS[$INDEX]}" | cut -d ':' -f 2)
    TASK_LOG="logs/T${SLURM_JOB_ID}_${LOG_PREFIX}_eval.log"

    wait $PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${LOG_PREFIX} (exit code: ${EXIT_CODE}) -> ${TASK_LOG}"
        FAILED=$((FAILED + 1))
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK: ${LOG_PREFIX} -> ${TASK_LOG}"
        SUCCEEDED=$((SUCCEEDED + 1))
    fi
done

##########################################################
# Summary
##########################################################
echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation Complete (Job ID: ${SLURM_JOB_ID})"
echo "  Succeeded: ${SUCCEEDED}"
echo "  Failed:    ${FAILED}"
echo "  Main log:  logs/eval_frame_delay_${SLURM_JOB_ID}.out"
echo "  Task logs: logs/T${SLURM_JOB_ID}_*_eval.log"
echo "============================================================"

exit $FAILED
