export CUDA_VISIBLE_DEVICES=0
python main.py -m optimizers.lr_x=0.0002,0.002,0.02,0.2,2 \
    optimizers.lr_y=0.0002,0.002,0.02,0.2,2 \