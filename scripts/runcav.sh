export CUDA_VISIBLE_DEVICES=0

python main.py -m \
    datasets=cifar10 \
    optimizers=adafm \
    '+multi_adafm_run_id=range(1,21)'