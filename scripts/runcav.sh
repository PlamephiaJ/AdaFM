#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python main.py -m optimizers=adafm \
    datasets.use_ratio=0.3 \
    setup.seed=100
