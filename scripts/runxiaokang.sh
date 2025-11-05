export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=msgda \
    optimizers.lr_discriminator=0.1,0.001,0.00001 \
    optimizers.lr_generator=0.1,0.001,0.00001