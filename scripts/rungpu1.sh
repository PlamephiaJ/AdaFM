#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python main.py -m optimizers=adagrad,adam,rmsprop
