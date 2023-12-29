# 测试代码是否正确
# 不同k和iter在相同Query设置下性能是否一致
# 分成test1.sh和test2.sh并行
python main.py --copy_source_model --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 100 --lr 0.00003 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar --copy_model cnn_3_2
python main.py --copy_source_model --noise_dataset generated_cifar --num_iter 20 -k 1000 --initial_size 100 --lr 0.00003 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar --copy_model cnn_3_2
