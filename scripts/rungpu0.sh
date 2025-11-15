#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
# python main.py -m optimizers.lr_x=0.0001,0.001,0.01,0.1,1 \
#     optimizers.lr_y=0.0001,0.001,0.01,0.1,1

# python main.py -m optimizers=tiada \
#     optimizers.lr_x=0.0001,0.001,0.01,0.1,1 \
#     optimizers.lr_y=0.0001,0.001,0.01,0.1,1

set -e

# (lr_x lr_y) pairs
pairs=(
  "0.05 0.05"
  "0.05 0.07"
  "0.5  0.0005"
  "0.5  0.05"
  "0.5  0.5"
  "0.5  0.07"
  "0.07 0.005"
  "0.07 0.05"
  "0.07 0.07"
)

for pair in "${pairs[@]}"; do
  # 把 "a b" 展开成 $1, $2
  set -- $pair
  lr_x="$1"
  lr_y="$2"

  echo "=============================="
  echo "Running: optimizers.lr_x=${lr_x}, optimizers.lr_y=${lr_y}"
  echo "=============================="

  python main.py optimizers.lr_x="${lr_x}" optimizers.lr_y="${lr_y}"
done
