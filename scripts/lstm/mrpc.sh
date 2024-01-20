#!/bin/bash

source ~/.bashrc
conda activate sbias

python main.py --model_name 'lstm'\
    --batch_size 32 \
    --dropout 0.1 \
    --embedding_dim 300 \
    --epochs 100 \
    --gamma 0.9 \
    --hidden_dim 256 \
    --lr 2e-05 \
    --weight_decay 0.01 \
    --max_length 128 \
    --num_head 8 \
    --num_labels 2 \
    --num_layers 4 \
    --replace_size 3 \
    --report_num_points 50000 \
    --train_num_points 392700 \
    --valid_num_points 9795 \
    --dataset 'mrpc' \
    --sensitivity_method 'embedding' \