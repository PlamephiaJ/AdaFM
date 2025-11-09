export CUDA_VISIBLE_DEVICES=1
python main.py -m \
    optimizers=adam,adagrad,rmsprop \ 
    models.lambda_term=1