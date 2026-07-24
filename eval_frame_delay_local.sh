#!/bin/bash
# Local evaluation for frame-delay VisionActionHead checkpoints on LIBERO Goal.

set -euo pipefail

GPU_IDS="0,1"
NUM_TRIALS=50
STAGGER_DELAY=30
TASKS_PER_GPU=4
DELAYS_CSV="0,5,10,15,20"
CHECKPOINT="${EVAL_CHECKPOINT:-}"
SEED=7
MAX_TRAIN_DELAY=20

usage() {
    echo "Usage: $0 --checkpoint PATH [--gpus 0,1] [--num_trials 50] [--delays 0,5,10,15,20]"
    echo "          [--tasks_per_gpu 4] [--stagger 30] [--seed 7]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --gpus) GPU_IDS="$2"; shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --delays) DELAYS_CSV="$2"; shift 2 ;;
        --tasks_per_gpu) TASKS_PER_GPU="$2"; shift 2 ;;
        --stagger) STAGGER_DELAY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required. Do not evaluate an old frame-delay checkpoint by accident."
    usage
    exit 1
fi
if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

shopt -s nullglob
ACTION_HEAD_FILES=("$CHECKPOINT"/action_head--*_checkpoint.pt)
PROPRIO_FILES=("$CHECKPOINT"/proprio_projector--*_checkpoint.pt)
if [[ ${#ACTION_HEAD_FILES[@]} -ne 1 || ${#PROPRIO_FILES[@]} -ne 1 ]]; then
    echo "ERROR: Expected exactly one action head and one proprio projector checkpoint in $CHECKPOINT"
    exit 1
fi
if [[ ! -f "$CHECKPOINT/model.safetensors.index.json" || ! -f "$CHECKPOINT/dataset_statistics.json" ]]; then
    echo "ERROR: Checkpoint is missing merged model weights or dataset statistics: $CHECKPOINT"
    exit 1
fi

source /home/sheng/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft-eval

export HF_HOME="/home/sheng/workspace/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_SCRIPT="experiments/robot/libero/run_libero_eval.py"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
IFS=',' read -r -a DELAY_ARRAY <<< "$DELAYS_CSV"
NUM_GPUS=${#GPU_ARRAY[@]}

if [[ $NUM_GPUS -eq 0 || $TASKS_PER_GPU -lt 1 ]]; then
    echo "ERROR: At least one GPU and one task per GPU are required."
    exit 1
fi
if [[ ${#DELAY_ARRAY[@]} -eq 0 ]]; then
    echo "ERROR: At least one evaluation delay is required."
    exit 1
fi

for DELAY in "${DELAY_ARRAY[@]}"; do
    if [[ ! "$DELAY" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid delay '$DELAY'; delays must be non-negative integers."
        exit 1
    fi
    if (( DELAY > MAX_TRAIN_DELAY )); then
        echo "WARNING: delay=$DELAY exceeds the training range 0-$MAX_TRAIN_DELAY environment steps."
    fi
done

COMMON_ARGS=(
    --use_l1_regression True
    --use_diffusion False
    --use_film False
    --num_images_in_input 2
    --use_proprio True
    --lora_rank 16
    --center_crop True
    --num_trials_per_task "$NUM_TRIALS"
    --num_open_loop_steps 8
    --use_vision_action_head True
    --action_head_vision_encoder siglip-base
    --freeze_action_head_vision True
    --action_head_num_views 2
    --task_suite_name libero_goal
    --seed "$SEED"
)

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="logs/eval_frame_delay_local_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
START_SECONDS=$SECONDS

declare -a GPU_SLOTS=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_SLOTS+=(0)
done

declare -A PID_GPU_IDX=()
declare -A PID_LABEL=()
declare -A TASK_RESULTS=()
RUNNING=0
SUCCEEDED=0
FAILED=0
SELECTED_GPU_IDX=0

stop_children() {
    trap - INT TERM
    echo "Stopping ${RUNNING} evaluation process(es)..."
    for PID in "${!PID_GPU_IDX[@]}"; do
        kill "$PID" 2>/dev/null || true
    done
    wait || true
    exit 130
}
trap stop_children INT TERM

reap_finished() {
    local PID GPU_IDX LABEL EXIT_CODE
    for PID in "${!PID_GPU_IDX[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            continue
        fi

        GPU_IDX=${PID_GPU_IDX[$PID]}
        LABEL=${PID_LABEL[$PID]}
        if wait "$PID"; then
            EXIT_CODE=0
        else
            EXIT_CODE=$?
        fi

        GPU_SLOTS[$GPU_IDX]=$((GPU_SLOTS[$GPU_IDX] - 1))
        RUNNING=$((RUNNING - 1))
        unset 'PID_GPU_IDX[$PID]'
        unset 'PID_LABEL[$PID]'

        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "[OK]     $LABEL"
            SUCCEEDED=$((SUCCEEDED + 1))
            TASK_RESULTS[$LABEL]="OK"
        else
            echo "[FAILED] $LABEL (exit code: $EXIT_CODE)"
            FAILED=$((FAILED + 1))
            TASK_RESULTS[$LABEL]="FAILED"
        fi
    done
}

select_available_gpu() {
    while true; do
        reap_finished
        for ((i=0; i<NUM_GPUS; i++)); do
            if (( GPU_SLOTS[i] < TASKS_PER_GPU )); then
                SELECTED_GPU_IDX=$i
                return
            fi
        done
        sleep 2
    done
}

echo "============================================================"
echo "Frame Delay Evaluation: LIBERO Goal"
echo "Checkpoint: $(basename "$CHECKPOINT")"
echo "GPUs: $GPU_IDS | tasks/GPU: $TASKS_PER_GPU"
echo "Delays (environment steps): $DELAYS_CSV"
echo "Trials per task: $NUM_TRIALS | seed: $SEED"
echo "Logs: $LOG_DIR"
echo "============================================================"

TOTAL_TASKS=${#DELAY_ARRAY[@]}
for TASK_IDX in "${!DELAY_ARRAY[@]}"; do
    DELAY=${DELAY_ARRAY[$TASK_IDX]}
    if [[ "$DELAY" -eq 0 ]]; then
        LABEL="baseline_d0_seed${SEED}"
        DELAY_ARGS=(--use_frame_delay_eval false)
    else
        LABEL="frame_delay_d${DELAY}_seed${SEED}"
        DELAY_ARGS=(--use_frame_delay_eval true --max_delay_steps_eval "$DELAY")
    fi

    select_available_gpu
    GPU_IDX=$SELECTED_GPU_IDX
    GPU_ID=${GPU_ARRAY[$GPU_IDX]}
    TASK_LOG="$LOG_DIR/${LABEL}.log"
    TASK_ERR="$LOG_DIR/${LABEL}.err"

    echo "[$(date '+%H:%M:%S')] Task $((TASK_IDX + 1))/$TOTAL_TASKS: delay=$DELAY -> GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVAL_SCRIPT" \
        --pretrained_checkpoint "$CHECKPOINT" \
        "${COMMON_ARGS[@]}" \
        "${DELAY_ARGS[@]}" \
        --run_id_note "$LABEL" \
        > "$TASK_LOG" 2> "$TASK_ERR" &

    PID=$!
    PID_GPU_IDX[$PID]=$GPU_IDX
    PID_LABEL[$PID]=$LABEL
    GPU_SLOTS[$GPU_IDX]=$((GPU_SLOTS[$GPU_IDX] + 1))
    RUNNING=$((RUNNING + 1))

    if (( TASK_IDX + 1 < TOTAL_TASKS && STAGGER_DELAY > 0 )); then
        sleep "$STAGGER_DELAY"
    fi
done

while (( RUNNING > 0 )); do
    reap_finished
    if (( RUNNING > 0 )); then
        sleep 2
    fi
done

echo "============================================================"
echo "Evaluation Complete"
echo "Succeeded: $SUCCEEDED/$TOTAL_TASKS"
echo "Failed:    $FAILED/$TOTAL_TASKS"
echo "Duration:  $((SECONDS - START_SECONDS)) seconds"
echo "Logs:      $LOG_DIR"
echo "============================================================"

for DELAY in "${DELAY_ARRAY[@]}"; do
    if [[ "$DELAY" -eq 0 ]]; then
        LABEL="baseline_d0_seed${SEED}"
    else
        LABEL="frame_delay_d${DELAY}_seed${SEED}"
    fi
    printf '%5s | %s\n' "$DELAY" "${TASK_RESULTS[$LABEL]:-N/A}"
done

exit "$FAILED"
