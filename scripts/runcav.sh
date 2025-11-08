export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=adafm,adagrad,rmsprop,sgd \
    models/backbone=wgan-gp-in