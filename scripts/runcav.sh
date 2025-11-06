export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=msgda \
    optimizers.lr_x=0.0001 \
    optimizers.lr_y=0.0001