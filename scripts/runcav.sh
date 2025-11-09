export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=adafm \
    optimizers.delta=1e-3,0.1,0.2