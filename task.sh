#!/bin/bash
python main.py --copy_source_model --noise_dataset generated_cifar_finetune --num_iter 50 -k 100 --initial_size 100 --lr 0.00003 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar
python main.py --copy_source_model --noise_dataset generated_cifar_finetune --num_iter 50 -k 100 --initial_size 100 --lr 0.001 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar


# 用更小的模型试一下
python main.py --train_source_model --source_model cnn_2_2 --num_epoch 200 --train_dropou 0.5 --train_l2 0.001
python main.py --copy_source_model --noise_dataset generated_cifar --num_iter 50 -k 100 --initial_size 100 --lr 0.001 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar --source_model cnn_2_2 --copy_model cnn_2_2
python main.py --copy_source_model --noise_dataset imagenet --num_iter 50 -k 100 --initial_size 100 --lr 0.001 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar --source_model cnn_2_2 --copy_model cnn_2_2
python main.py --copy_source_model --noise_dataset generated_cifar --num_iter 50 -k 100 --initial_size 100 --lr 0.001 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar --source_model cnn_2_2 --copy_model cnn_1_2
