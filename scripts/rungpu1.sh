#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python main.py -m \
  optimizers=tiada \
  optimizers.lr_x=1e-4,1e-3 \
  optimizers.lr_y=1,1e-1,1e-3,1e-4