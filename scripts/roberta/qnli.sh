#!/bin/bash

source ~/.bashrc
conda activate sbias

python main.py --model_name 'roberta-scratch'\
    --batch_size 64 \
    --dropout 0.2 \
    --embedding_dim 300 \
    --epochs 30 \
    --gamma 1 \
    --hidden_dim 128 \
    --lr 1e-05 \
    --max_length 32 \
    --num_head 4 \
    --num_labels 2 \
    --num_layers 1 \
    --replace_size 3 \
    --report_num_points 50000 \
    --train_num_points 392700 \
    --valid_num_points 9795 \
    --dataset 'qnli' \