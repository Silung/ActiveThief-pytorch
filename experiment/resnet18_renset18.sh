# 画图
cd ..

# 从头直接提取
python main.py --copy_source_model --source_model resnet18 --copy_model resnet18 --true_dataset mnist --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr 0
python main.py --copy_source_model --source_model resnet18 --copy_model resnet18 --true_dataset cifar --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr 0
python main.py --copy_source_model --source_model resnet18 --copy_model resnet18 --true_dataset gtsrb --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr 0

# al
python main.py --copy_source_model --source_model resnet18 --copy_model resnet18 --true_dataset mnist --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr 0 --sampling_method kcenter
python main.py --copy_source_model --source_model resnet18 --copy_model resnet18 --true_dataset cifar --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr 0 --sampling_method kcenter
python main.py --copy_source_model --source_model resnet18 --copy_model resnet18 --true_dataset gtsrb --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr 0 --sampling_method kcenter

# pretrian on imagenet
# python main.py --train_source_model --source_model resnet18 --num_epoch 1000 --true_dataset imagenet

# transferThief
python main.py --copy_source_model --source_model resnet18 --copy_model resnet18 --true_dataset mnist --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr 0 --pretrain saved/sm_resnet18/td_imagenet/t_drop_0.5/t_l2_0.001/true/trained_model.pth
python main.py --copy_source_model --source_model resnet18 --copy_model resnet18 --true_dataset cifar --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr 0 --pretrain saved/sm_resnet18/td_imagenet/t_drop_0.5/t_l2_0.001/true/trained_model.pth
python main.py --copy_source_model --source_model resnet18 --copy_model resnet18 --true_dataset gtsrb --noise_dataset imagenet --num_iter 10 -k 2000 --initial_size 2000 --lr 0 --pretrain saved/sm_resnet18/td_imagenet/t_drop_0.5/t_l2_0.001/true/trained_model.pth
