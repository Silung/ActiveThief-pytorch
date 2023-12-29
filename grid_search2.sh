#!/bin/bash
lrs=(1e-5 3e-5)
python main.py --path_prefix f1 --train_source_model --train_dropout 0.1 --train_l2 0
for lr in "${lrs[@]}"
do
    echo "try lr=$lr"
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0.1 --train_l2 0 --mea_dropout 0.1 --mea_l2 0.001
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0.1 --train_l2 0 --mea_dropout 0 --mea_l2 0.001
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0.1 --train_l2 0 --mea_dropout 0.1 --mea_l2 0
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0.1 --train_l2 0 --mea_dropout 0 --mea_l2 0
done

python main.py --path_prefix f1 --train_source_model --train_dropout 0 --train_l2 0
for lr in "${lrs[@]}"
do
    echo "try lr=$lr"
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0 --train_l2 0 --mea_dropout 0.1 --mea_l2 0.001
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0 --train_l2 0 --mea_dropout 0 --mea_l2 0.001
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0 --train_l2 0 --mea_dropout 0.1 --mea_l2 0
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0 --train_l2 0 --mea_dropout 0 --mea_l2 0
done

