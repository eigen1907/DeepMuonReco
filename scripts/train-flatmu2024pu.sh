#!/bin/bash
../train.py --device cuda:5 \
           --train_file_path ../data/flatmu2024pu/train.root \
           --val_file_path ../data/flatmu2024pu/val.root \
           --num_epochs 500 \
           --early_stopping_patience 30 \
           --batch_size 64 \
           --eval_batch_size 64 \
           --dim_model 128 \
           --dim_feedforward 256 \
           --num_heads 8 \
           --num_layers 4 \
           --activation gelu \
           --dropout 0.1 \
           --weight_decay 1e-1 \
           --lr 1e-4

#################################################################
# Best Model (25 KPS Spring)
#################################################################
#./train.py --device cuda:4 \
#           --train_file_path ./data/train.root \
#           --val_file_path ./data/val.root \
#           --num_epochs 100 \
#           --early_stopping_patience 10 \
#           --batch_size 128 \
#           --eval_batch_size 128 \
#           --dim_model 128 \
#           --dim_feedforward 256 \
#           --num_heads 8 \
#           --num_layers 4 \
#           --activation gelu \
#           --dropout 0.1 \
#           --weight_decay 1e-1 \
#           --lr 1e-4 \
