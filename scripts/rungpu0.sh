#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python main.py -m \
  optimizers=tiada \
  optimizers.lr_x=1,1e-1,1e-3,1e-4 \
  optimizers.lr_y=1,1e-1,1e-3,1e-4