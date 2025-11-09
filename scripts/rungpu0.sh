export CUDA_VISIBLE_DEVICES=0
python main.py -m \
    optimizers=adafm,msgda,tiada,pesg \ 
    models.lambda_term=0.1