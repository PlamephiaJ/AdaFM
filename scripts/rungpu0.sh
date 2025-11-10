export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=adafm \
    datasets=cifar10 \
    optimizers.lr_x=0.005,0.008 \
    optimizers.lr_y=0.005,0.008,0.01,0.012,0.015