export CUDA_VISIBLE_DEVICES=0

python main.py -m \
    datasets=cifar100 \
    optimizers=adafm,tiada