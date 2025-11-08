export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=adagrad \
    models/backbone=wgan-gp-in