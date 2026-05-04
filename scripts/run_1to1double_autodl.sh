#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

RESULT_ROOT="results/DoubleLoop1to1/GAN_cifar10/wgan-gp-in"
OPTIMIZERS=(rmsprop)
LR_VALUES=(0.0001 0.001 0.01 0.1)
CRITIC_ITERS=1
GPU_IDS=(0 1 2 3)

is_completed_run() {
    local optimizer="$1"
    local lr_x="$2"
    local lr_y="$3"
    local optimizer_dir="${RESULT_ROOT}/${optimizer}"
    local run_dir
    local config_file

    [[ -d "$optimizer_dir" ]] || return 1

    for run_dir in "$optimizer_dir"/*; do
        [[ -d "$run_dir" ]] || continue
        [[ -f "$run_dir/best_metrics.csv" ]] || continue
        [[ -f "$run_dir/training_log.csv" ]] || continue

        config_file="$run_dir/config_snapshot.yaml"
        [[ -f "$config_file" ]] || continue

        if rg -F -x -q "  lr_x: ${lr_x}" "$config_file" \
            && rg -F -x -q "  lr_y: ${lr_y}" "$config_file" \
            && rg -F -x -q "  critic_iters: ${CRITIC_ITERS}" "$config_file"; then
            return 0
        fi
    done

    return 1
}

run_worker() {
    local worker_slot="$1"
    local gpu_id="$2"
    local num_workers="${#GPU_IDS[@]}"
    local global_index=0
    local assigned_runs=0
    local skipped_runs=0
    local started_runs=0
    local optimizer
    local lr_x
    local lr_y

    for optimizer in "${OPTIMIZERS[@]}"; do
        for lr_x in "${LR_VALUES[@]}"; do
            for lr_y in "${LR_VALUES[@]}"; do
                if (( global_index % num_workers != worker_slot )); then
                    global_index=$((global_index + 1))
                    continue
                fi

                assigned_runs=$((assigned_runs + 1))

                if is_completed_run "$optimizer" "$lr_x" "$lr_y"; then
                    skipped_runs=$((skipped_runs + 1))
                    echo "[worker ${worker_slot} gpu ${gpu_id}] Skip completed run: optimizer=${optimizer}, lr_x=${lr_x}, lr_y=${lr_y}"
                    global_index=$((global_index + 1))
                    continue
                fi

                started_runs=$((started_runs + 1))
                echo "[worker ${worker_slot} gpu ${gpu_id}] Run: optimizer=${optimizer}, lr_x=${lr_x}, lr_y=${lr_y}"

                CUDA_VISIBLE_DEVICES="${gpu_id}" python main.py \
                    optimizers="${optimizer}" \
                    optimizers.critic_iters="${CRITIC_ITERS}" \
                    optimizers.lr_x="${lr_x}" \
                    optimizers.lr_y="${lr_y}"

                global_index=$((global_index + 1))
            done
        done
    done

    echo "[worker ${worker_slot} gpu ${gpu_id}] Finished. assigned=${assigned_runs}, skipped=${skipped_runs}, started=${started_runs}"
}

echo "Starting 4-GPU run across: ${GPU_IDS[*]}"

pids=()
for worker_slot in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$worker_slot]}"
    run_worker "$worker_slot" "$gpu_id" &
    pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        exit_code=1
    fi
done

exit "$exit_code"
