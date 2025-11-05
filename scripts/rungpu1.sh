export CUDA_VISIBLE_DEVICES=1
python main.py -m \
    optimizers=pesg \
    optimizers.lr=0.1,0.2,0.03,0.05