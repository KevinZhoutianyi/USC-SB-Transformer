#!/bin/bash

source ~/.bashrc
conda activate sbias

python main.py --model_name 'roberta-scratch' --roberta_act 'relu'\
    --batch_size 32 \
    --dropout 0.1 \
    --embedding_dim 256 \
    --epochs 100 \
    --gamma 0.9 \
    --hidden_dim 256 \
    --lr 1e-04 \
    --weight_decay 1e-04 \
    --max_length 128 \
    --num_head 8 \
    --num_labels 2 \
    --num_layers 4 \
    --replace_size 3 \
    --report_num_points 5000 \
    --train_num_points 392700 \
    --valid_num_points 9795 \
    --dataset 'qqp' \
    --sensitivity_method 'embedding' \
    --embedding_noise_variance 4 \
    --exp_name 'variance-4' \