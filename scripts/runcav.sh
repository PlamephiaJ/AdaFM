export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    datasets=cifar10 \
    optimizers=adafm \
    optimizers.delta=0,1e-3,0.1,0.2