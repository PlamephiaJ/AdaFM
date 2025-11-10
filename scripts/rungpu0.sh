export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    datasets=cifar10 \
    optimizers=tiada,adafm,msgda,pesg \
    models.lambda_term=0.1,1.0,10.0