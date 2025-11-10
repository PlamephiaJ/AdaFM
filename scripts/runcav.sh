export CUDA_VISIBLE_DEVICES=0

python main.py -m \
    datasets=cifar10 \
    optimizers=adafm,msgda,pesg,tiada