# train true model
python main.py --train_source_model

# onehot random
python main.py --copy_source_model --num_epoch 1000 --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --api_retval onehot --sampling_method random
# onehot uncertainty
# python main.py --copy_source_model --num_epoch 1000 --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --api_retval onehot --sampling_method uncertainty
# onehot kcenter
# python main.py --copy_source_model --num_epoch 1000 --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --api_retval onehot --sampling_method kcenter
# softmax random
# python main.py --copy_source_model --num_epoch 1000 --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --api_retval softmax --sampling_method random
# softmax uncertainty
# python main.py --copy_source_model --num_epoch 1000 --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --api_retval softmax --sampling_method uncertainty
# softmax kcenter
# python main.py --copy_source_model --num_epoch 1000 --noise_dataset imagenet --num_iter 20 -k 1000 --initial_size 1000 --api_retval softmax --sampling_method kcenter

# 使用Mnist 10%的数据做生成式数据验证
# python main.py --copy_source_model --num_epoch 1000 --noise_dataset mnist_small --num_iter 20 -k 1000 --initial_size 1000 --api_retval onehot --sampling_method random
