#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RESULT_ROOT="results/DoubleLoop1to1/GAN_cifar10/wgan-gp-in"
OPTIMIZERS=(rmsprop)
LR_VALUES=(0.0001 0.001 0.01 0.1)
CRITIC_ITERS=1

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

total_runs=0
skipped_runs=0
started_runs=0

for optimizer in "${OPTIMIZERS[@]}"; do
    for lr_x in "${LR_VALUES[@]}"; do
        for lr_y in "${LR_VALUES[@]}"; do
            total_runs=$((total_runs + 1))

            if is_completed_run "$optimizer" "$lr_x" "$lr_y"; then
                skipped_runs=$((skipped_runs + 1))
                echo "[$skipped_runs/$total_runs] Skip completed run: optimizer=${optimizer}, lr_x=${lr_x}, lr_y=${lr_y}"
                continue
            fi

            started_runs=$((started_runs + 1))
            echo "[$started_runs started] Run: optimizer=${optimizer}, lr_x=${lr_x}, lr_y=${lr_y}"

            python main.py \
                optimizers="${optimizer}" \
                optimizers.critic_iters="${CRITIC_ITERS}" \
                optimizers.lr_x="${lr_x}" \
                optimizers.lr_y="${lr_y}"
        done
    done
done

echo "Finished. total=${total_runs}, skipped=${skipped_runs}, started=${started_runs}"
