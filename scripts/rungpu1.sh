export CUDA_VISIBLE_DEVICES=1
python main.py -m \
    optimizers=adafm,pesg,msgda,tiada \
    models/backbone=wgan-gp-light