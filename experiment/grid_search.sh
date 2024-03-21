#!/bin/bash
lrs=(1e-5 3e-5)
python main.py --path_prefix f1 --train_source_model --train_dropout 0.1 --train_l2 0.001
for lr in "${lrs[@]}"
do
    echo "try lr=$lr"
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0.1 --train_l2 0.001 --mea_dropout 0.1 --mea_l2 0.001
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0.1 --train_l2 0.001 --mea_dropout 0 --mea_l2 0.001
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0.1 --train_l2 0.001 --mea_dropout 0.1 --mea_l2 0
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0.1 --train_l2 0.001 --mea_dropout 0 --mea_l2 0
done

python main.py --path_prefix f1 --train_source_model --train_dropout 0 --train_l2 0.001
for lr in "${lrs[@]}"
do
    echo "try lr=$lr"
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0 --train_l2 0.001 --mea_dropout 0.1 --mea_l2 0.001
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0 --train_l2 0.001 --mea_dropout 0 --mea_l2 0.001
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0 --train_l2 0.001 --mea_dropout 0.1 --mea_l2 0
    python main.py --path_prefix f1 --copy_source_model --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --train_dropout 0 --train_l2 0.001 --mea_dropout 0 --mea_l2 0
done





# # 定义lr的取值范围
# lrs=(1e-5 3e-5 5e-5 7e-5 1e-4)

# # 循环遍历lr的取值范围
# for lr in "${lrs[@]}"
# do
#     # 执行命令
#     echo "try lr=$lr"
#     python main.py --copy_source_model --true_dataset mnist --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --dropout 0.1
# done

# # 循环遍历lr的取值范围
# for lr in "${lrs[@]}"
# do
#     # 执行命令
#     echo "try lr=$lr"
#     python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --dropout 0.2
# done

# for lr in "${lrs[@]}"
# do
#     # 执行命令
#     echo "try lr=$lr"
#     python main.py --copy_source_model --true_dataset cifar --noise_dataset generated_cifar --num_iter 10 -k 2000 --initial_size 2000 --lr $lr --dropout 0.2
# done

