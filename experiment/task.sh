echo normal mnist resnet
python main.py --copy_source_model --copy_model resnet --true_dataset mnist --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0
echo normal cifar resnet
python main.py --copy_source_model --copy_model resnet --true_dataset cifar --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0
echo normal gtsrb resnet
python main.py --copy_source_model --copy_model resnet --true_dataset gtsrb --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0


# 验证al中的k-center方法
echo al mnist resnet
python main.py --copy_source_model --copy_model resnet --true_dataset mnist --noise_dataset imagenet --num_iter 8 -k 5000 --initial_size 5000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0 --sampling_method kcenter
echo al cifar resnet
python main.py --copy_source_model --copy_model resnet --true_dataset cifar --noise_dataset imagenet --num_iter 8 -k 5000 --initial_size 5000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0 --sampling_method kcenter
echo al gtsrb resnet
python main.py --copy_source_model --copy_model resnet --true_dataset gtsrb --noise_dataset imagenet --num_iter 8 -k 5000 --initial_size 5000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0 --sampling_method kcenter


#pretrain
echo pretrain mnist resnet
python main.py --copy_source_model --copy_model resnet_pretrained --true_dataset mnist --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0
echo pretrain cifar resnet
python main.py --copy_source_model --copy_model resnet_pretrained --true_dataset cifar --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0
echo pretrain gtsrb resnet
python main.py --copy_source_model --copy_model resnet_pretrained --true_dataset gtsrb --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0
