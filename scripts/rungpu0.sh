#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python main.py -m optimizers=adafm,msgda,pesg,tiada
