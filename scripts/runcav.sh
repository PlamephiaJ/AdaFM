export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=pesg \
    optimizers.lr_x=0.1,0.001,0.05 \
    optimizers.lr_y=0.1,0.001,0.05