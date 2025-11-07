export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=adam \
    models/backbone=wgan-gp-in