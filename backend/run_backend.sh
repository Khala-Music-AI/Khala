#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Backend launcher
#
# Usage:
#   bash run_backend.sh
#   bash run_backend.sh --gpus 0
#   bash run_backend.sh --gpus 0,1 --runtime-mode keep_loaded
#   bash run_backend.sh stop
#
# This script assumes the runtime environment is already prepared.
# It starts one worker per GPU listed in GPU_IDS, waits for workers to become
# healthy, then starts backend_api.py.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MEGATRON_ROOT="$PROJECT_ROOT/models/Megatron"
DECODER_ROOT="$PROJECT_ROOT/models/Decoder"
LOG_DIR="$SCRIPT_DIR/logs"

# ============================================================================
# User-facing configuration
# Default behavior is single-GPU safe mode.
# Advanced users can override GPU selection and runtime mode via CLI flags.
# ============================================================================

GPU_IDS=(0)
WORKER_RUNTIME_MODE=one_shot # keep_loaded | one_shot

API_PORT=8889
WORKER_BASE_PORT=8001
BASE_MASTER_PORT=8791
BASE_SEED=1283

# Shared Megatron inference arguments passed to every worker.
MEGATRON_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --tokenizer-type NullTokenizer
    --norm-epsilon 1e-6
    --num-tokens-to-generate 23552
    --inference-max-seq-length 25600
    --stream
    --enable-cuda-graph
    --flash-decode
    --bf16
)

NUM_WORKERS=0


usage() {
    cat <<'EOF'
Usage:
  bash run_backend.sh
  bash run_backend.sh --gpus 0
  bash run_backend.sh --gpus 0,1 --runtime-mode keep_loaded
  bash run_backend.sh stop

Options:
  --gpus <ids>            Comma-separated physical GPU ids, e.g. 0 or 0,1 or 6,7
  --runtime-mode <mode>   Worker runtime mode: one_shot or keep_loaded
  --help                  Show this help message

Default behavior:
  --gpus 0
  --runtime-mode one_shot
EOF
}


parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpus)
                [[ $# -ge 2 ]] || { echo "ERROR: --gpus requires a value."; exit 1; }
                IFS=',' read -r -a GPU_IDS <<< "$2"
                shift 2
                ;;
            --runtime-mode)
                [[ $# -ge 2 ]] || { echo "ERROR: --runtime-mode requires a value."; exit 1; }
                WORKER_RUNTIME_MODE="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "ERROR: Unknown argument: $1"
                echo ""
                usage
                exit 1
                ;;
        esac
    done
}


stop_services() {
    echo "=== Stopping all services ==="
    for pid_file in "$LOG_DIR"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  Killed $(basename "$pid_file" .pid) (PID $pid)"
        fi
        rm -f "$pid_file"
    done
    echo "Done."
}


cleanup_stale_pids() {
    mkdir -p "$LOG_DIR"
    for pid_file in "$LOG_DIR"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        pid="$(cat "$pid_file")"
        kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
    done
}


validate_configuration() {
    local cleaned_gpu_ids=()
    for gpu in "${GPU_IDS[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        [[ -n "$gpu" ]] || continue
        [[ "$gpu" =~ ^[0-9]+$ ]] || {
            echo "ERROR: Invalid GPU id '$gpu'. Use comma-separated integers like --gpus 0,1."
            exit 1
        }
        cleaned_gpu_ids+=("$gpu")
    done

    GPU_IDS=("${cleaned_gpu_ids[@]}")
    NUM_WORKERS=${#GPU_IDS[@]}

    if [[ "$NUM_WORKERS" -eq 0 ]]; then
        echo "ERROR: GPU_IDS is empty, so no workers would be started."
        exit 1
    fi

    if [[ "$WORKER_RUNTIME_MODE" != "one_shot" && "$WORKER_RUNTIME_MODE" != "keep_loaded" ]]; then
        echo "ERROR: Unsupported runtime mode '$WORKER_RUNTIME_MODE'. Use one_shot or keep_loaded."
        exit 1
    fi
}


start_workers() {
    echo ""
    echo "=== Starting $NUM_WORKERS worker(s) ==="

    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local_port=$((WORKER_BASE_PORT + i))
        local_master_port=$((BASE_MASTER_PORT + i))
        local_seed=$((BASE_SEED + i * 1000))
        local_gpu="${GPU_IDS[$i]}"
        local_log_file="$LOG_DIR/worker_${i}.log"

        echo "  [Worker $i] gpu=$local_gpu port=$local_port seed=$local_seed master_port=$local_master_port"

        CUDA_VISIBLE_DEVICES="$local_gpu" \
        MASTER_ADDR=127.0.0.1 \
        MASTER_PORT="$local_master_port" \
        PYTHONUNBUFFERED=1 \
        nohup python backend_worker.py \
            --worker-port "$local_port" \
            --runtime-mode "$WORKER_RUNTIME_MODE" \
            --seed "$local_seed" \
            "${MEGATRON_ARGS[@]}" \
            > "$local_log_file" 2>&1 &

        echo $! > "$LOG_DIR/worker_${i}.pid"
    done
}


wait_for_workers() {
    local max_wait=300
    local elapsed=0
    local ready=0

    echo ""
    echo "=== Waiting for workers to become healthy (up to 5 min) ==="

    while [[ $elapsed -lt $max_wait ]]; do
        sleep 5
        elapsed=$((elapsed + 5))
        ready=0

        for i in $(seq 0 $((NUM_WORKERS - 1))); do
            local_port=$((WORKER_BASE_PORT + i))
            status="$(
                curl -s --max-time 2 "http://127.0.0.1:$local_port/health" 2>/dev/null \
                | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null \
                || true
            )"
            [[ "$status" == "idle" || "$status" == "busy" ]] && ready=$((ready + 1))
        done

        echo "  [${elapsed}s] $ready / $NUM_WORKERS workers ready"
        [[ $ready -ge $NUM_WORKERS ]] && break
    done

    if [[ $ready -lt $NUM_WORKERS ]]; then
        echo "WARNING: Only $ready / $NUM_WORKERS workers are ready after ${max_wait}s."
        echo "Check $LOG_DIR/worker_*.log for details."
    fi
}


start_api() {
    echo ""
    echo "=== Starting API server on port $API_PORT ==="

    PYTHONUNBUFFERED=1 \
    nohup python backend_api.py \
        --port "$API_PORT" \
        --num-workers "$NUM_WORKERS" \
        --worker-base-port "$WORKER_BASE_PORT" \
        > "$LOG_DIR/api.log" 2>&1 &

    echo $! > "$LOG_DIR/api.pid"
    sleep 2
}


print_summary() {
    echo ""
    echo "============================================"
    echo "  Runtime mode: $WORKER_RUNTIME_MODE"
    echo "  Workers:  $NUM_WORKERS"
    echo "  GPUs:     ${GPU_IDS[*]}"
    echo "  Worker ports: $WORKER_BASE_PORT-$((WORKER_BASE_PORT + NUM_WORKERS - 1))"
    echo "  API:      http://0.0.0.0:$API_PORT"
    echo "  Project:  $PROJECT_ROOT"
    echo "============================================"
    echo "  Logs:     tail -f $LOG_DIR/worker_*.log"
    echo "            tail -f $LOG_DIR/api.log"
    echo "  Stop:     bash run_backend.sh stop"
    echo "============================================"
    echo ""
}


tail_logs() {
    echo "=== Tailing all logs (Ctrl-C detaches, services keep running) ==="
    exec tail -f "$LOG_DIR"/*.log
}


main() {
    cd "$SCRIPT_DIR"
    export PYTHONPATH="$PROJECT_ROOT:$MEGATRON_ROOT:$DECODER_ROOT:${PYTHONPATH:-}"

    parse_args "$@"
    validate_configuration
    cleanup_stale_pids
    start_workers
    wait_for_workers
    start_api
    print_summary
    tail_logs
}


if [[ "${1:-}" == "stop" ]]; then
    stop_services
    exit 0
fi

main "$@"
