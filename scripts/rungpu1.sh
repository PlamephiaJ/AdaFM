#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python main.py -m optimizers=adafm \
     optimizers.lr_x=0.0001,0.001,0.01,0.1,1 \
     optimizers.lr_y=0.0001,0.001,0.01,0.1,1