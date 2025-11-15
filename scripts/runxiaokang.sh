#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python main.py -m optimizers=pesg \
    optimizers.lr_x=0.0001,0.001,0.01,0.1,1 \
    optimizers.lr_y=0.0001,0.001,0.01,0.1,1