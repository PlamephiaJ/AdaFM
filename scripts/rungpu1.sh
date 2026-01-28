#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python main.py -m \
  optimizers=adafm \
  setup.seed=42,97