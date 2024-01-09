# 第一次实验 pretrained restnet18
python main.py --copy_source_model --noise_dataset imagenet --num_iter 50 -k 100 --initial_size 100 --lr 0.00003 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar --source_model cnn_3_2 --copy_model resnet_pretrained

# 没跑完实验 随机初始化restnet18
python main.py --copy_source_model --noise_dataset imagenet --num_iter 50 -k 100 --initial_size 100 --lr 0.00003 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar --source_model cnn_3_2 --copy_model resnet

# 实验非常慢，以后的实验参数改成 num_iter=20,k=init=1000

# pretrained cnn32
# q=5k, agr=0.4989
# q=10k, agr=0.5849
# q=15k, agr=0.5996
# q=20k， agr=0.6324
python main.py --copy_source_model --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --lr 0.00003 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar --source_model cnn_3_2 --copy_model cnn_3_2 --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

# normal cnn32
# q=5k, agr=0.4835
# q=10k, agr=0.5455
# q=15k, agr=0.565
# q=20k, agr=0.5864
python main.py --copy_source_model --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --lr 0.00003 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --true_dataset cifar --source_model cnn_3_2 --copy_model cnn_3_2

# baseline差10%性能，重新跑一下试试
# 结果 q=20k, agr=56%, 差的远
python main.py --train_source_model --num_epoch 1000 --true_dataset cifar --train_dropout 0.2
python main.py --copy_source_model --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --lr 0.00003 --train_dropout 0.2 --train_l2 0.001 --mea_dropout 0.2 --mea_l2 0.001 --true_dataset cifar

# 完全按active的论文设置跑一遍
# 结果 q=20k, agr=0.6009
python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --lr 0.01 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 50 --patience 20

python main.py --train_source_model --num_epoch 1000 --true_dataset cifar --train_dropout 0.5 --train_l2 0.001 --lr 3e-5 --batch_size 50
python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --lr 3e-5 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 50 --patience 200

python main.py --train_source_model --num_epoch 1000 --true_dataset cifar --train_dropout 0.5 --train_l2 0.001 --lr 3e-5 --batch_size 50
python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --lr 3e-5 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 50 --patience 200

# 20k best agr 0.7141
python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20


python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0
# 10k 20k 30k 40k
# 0.5794 0.6946 0.7329 0.7462 seed=1960
# 0.6163 0.7145 0.7392 0.7462 seed=0
# 0.6316 0.6817 0.7147 0.7453 seed=1
# 0.6077 0.6879 0.7441 0.7503 seed=2
# 0.593 0.7203 0.7383 0.7539 seed=3
# 0.63 0.7134 0.725 0.7506 seed=4
# 0.6149 0.698 0.7283 0.7476 seed=5
# 0.71 0.751 0.7608 0.7786 seed=0 pretrain
# 0.6163(不涉及al不起作用) 0.7064 0.7244 0.7527 seed=0 al=k-center

python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0 --copy_model resnet
python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0 --copy_model resnet_pretrained
# 10k 20k 30k 40k
# 0.5744 0.7455 0.7409 0.7335 seed=0 pretrain 85.91min 更快
# 0.3817 0.3533 0.3982 0.4018 seed=0 185.2min


python main.py --copy_source_model --true_dataset mnist --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0
# 10k 20k 30k 40k
# 0.9262 0.9651 0.9719 0.9686
# 0.9319 0.9653 0.9728 0.9779 pretrain

python main.py --copy_source_model --true_dataset gtsrb --noise_dataset imagenet --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0
# 10k 20k 30k 40k
# 0.807284 0.885827 0.923753 0.942755
# 0.833967 0.907601 0.935154 0.947348 pretrain

# 验证al中的k-center方法
python main.py --copy_source_model --true_dataset mnist --noise_dataset imagenet --num_iter 8 -k 5000 --initial_size 5000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0 --sampling_method kcenter
python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 8 -k 5000 --initial_size 5000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0 --sampling_method kcenter
python main.py --copy_source_model --true_dataset gtsrb --noise_dataset imagenet --num_iter 8 -k 5000 --initial_size 5000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0 --sampling_method kcenter
# 5k 10k 15k 20k 25k 30k 35k 40k
# 0.8383 0.9467 0.9626 0.9659 0.9768 0.9738 0.9783 0.9774 mnist
# 0.495 0.6186 0.6555 0.6947 0.7109 0.7262 0.7486 0.7317 cifar
# 0.571496 0.812193 0.873001 0.911639 0.907443 0.912272 0.941489 0.946239 gtsrb

# 结合AL
python main.py --copy_source_model --true_dataset cifar --noise_dataset imagenet --num_iter 8 -k 5000 --initial_size 5000 --lr 0 --train_dropout 0.5 --train_l2 0.001 --mea_dropout 0.5 --mea_l2 0.001 --batch_size 150 --patience 20 --seed 0 --sampling_method kcenter --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth
# 10k 20k 30k 40k
# 