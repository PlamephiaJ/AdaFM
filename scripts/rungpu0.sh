export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=pesg \
    optimizers.lr=0.2,0.03,0.05,0.001,0.003,0.01