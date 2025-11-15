#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python main.py -m optimizers=pesg \
    optimizers.lr=0.0001 \
    models.evaluation.use_fid=false
    #,0.001,0.01,0.1,1 \