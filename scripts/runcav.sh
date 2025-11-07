export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=adafm,pesg,msgda,tiada \
    models/backbone=wgan-gp-attnd