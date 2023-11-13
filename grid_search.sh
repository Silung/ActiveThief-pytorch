#!/bin/bash

# 定义lr的取值范围
lrs=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)

# 循环遍历lr的取值范围
for lr in "${lrs[@]}"
do
    # 执行命令
    echo "try lr=$lr"
    python main.py --copy_source_model --true_dataset mnist --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr $lr
done
