# Getting started

TransferThief runs on Python 3 using pytorch. The repo is forked from https://github.com/shukla-aditya-csa/activethief. We reimplement a part of ActiveThief in this repo using Python 3 and pytorch.

To download the required datasets, first run:

    python download_datasets.py

Then, to preprocess the ImageNet data:

    python preprocess_imagenet.py

Following this, the repository is ready for use. Here is a sample command:

    python main.py --train_source_model --copy_source_model

For experiments about attack performance, run:

    python main.py --train_source_model --true_dataset imagenet

    python main.py --train_source_model --copy_source_model --true_dataset mnist --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --seed 0 --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

    python main.py --train_source_model --copy_source_model --true_dataset cifar --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --seed 0 --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

    python main.py --train_source_model --copy_source_model --true_dataset gtsrb --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --seed 0 --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

For experiments about query efficiency, run:
    # Query efficiency
    python main.py --copy_source_model --true_dataset mnist --num_iter 20 -k 1000 --initial_size 1000 --lr 0 --seed 0 --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

    python main.py --copy_source_model --true_dataset cifar --num_iter 20 -k 1000 --initial_size 1000 --lr 0 --seed 0 --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

    python main.py --copy_source_model --true_dataset gtsrb --num_iter 20 -k 1000 --initial_size 1000 --lr 0 --seed 0 --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

For experiments about soft label, run:

    python main.py --train_source_model --copy_source_model --true_dataset mnist --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --seed 0 --api_retval softmax --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

    python main.py --train_source_model --copy_source_model --true_dataset cifar --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --seed 0 --api_retval softmax --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

    python main.py --train_source_model --copy_source_model --true_dataset gtsrb --num_iter 4 -k 10000 --initial_size 10000 --lr 0 --seed 0 --api_retval softmax --pretrain saved/cnn_3_2/imagenet/true/trained_model.pth

# License

ActiveThief-pytorch is available under an MIT License. Parts of the codebase is based on code from other repositories, also under an MIT license.  
Please see the LICENSE file, and the inline license included in each code file for more details.





