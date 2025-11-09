export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    datasets=cifar10 \
    optimizers=adafm,msgda,tiada,pesg \
    models.lambda_term=0.1,1