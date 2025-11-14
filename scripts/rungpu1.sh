export CUDA_VISIBLE_DEVICES=1
python main.py -m \
    optimizers.lr_x=0.0005,0.005,0.05,0.5,0.07 \
    optimizers.lr_y=0.0005,0.005,0.05,0.5,0.07