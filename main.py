import os
import sys
import time
import random
import argparse
import numpy as np
import torch

from mea import *
from train import *
from true_model_test_on_noise_dataset import *
from utils.class_loader import *
from utils.optuna_search import *


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Active Thief')

    parser.add_argument('--source_model', type=str, default='cnn_3_2')
    parser.add_argument('--copy_model', type=str, default='cnn_3_2')
    parser.add_argument('--true_dataset', type=str, default='mnist')
    parser.add_argument('--noise_dataset', type=str, default='imagenet')
    parser.add_argument('--num_to_keep', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--train_dropout', type=float, default=0.5)
    parser.add_argument('--train_l2', type=float, default=0.001)
    parser.add_argument('--mea_dropout', type=float, default=0.5)
    parser.add_argument('--mea_l2', type=float, default=0.001)
    parser.add_argument('--num_train_batch', type=int, default=1)

    parser.add_argument('--iterative', action='store_true')
    parser.add_argument('--initial_size', type=int, default=1000)
    parser.add_argument('--num_iter', type=int, default=20)
    parser.add_argument('-k', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--train_source_model', action='store_true')
    parser.add_argument('--copy_source_model', action='store_true')
    parser.add_argument('--true_model_test', action='store_true')
    parser.add_argument('--true_model_test_on_noise_dataset', action='store_true')

    parser.add_argument('--sampling_method', type=str, choices=['random', 'uncertainty', 'kcenter', 'deepfool', 'certainty'], default='random')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'adagrad'], default='adam')
    parser.add_argument('--api_retval', choices=['onehot', 'softmax'], type=str, default='onehot')
    parser.add_argument('--pretrain', type=str, default=None)
    
    parser.add_argument('--path_prefix', type=str, default='')
    parser.add_argument('--num_fig', type=int, default=10)
    parser.add_argument('--optuna_search', action='store_true')
    parser.add_argument('--ssl', action='store_true')
    parser.add_argument('--normalize_channels', action='store_true')
    

    args = parser.parse_args()

    fix_random_seed_as(args.seed)
            
    assert args.source_model is not None
    assert args.copy_model is not None
    assert args.true_dataset is not None
    assert args.noise_dataset is not None


    noise_dataset = args.noise_dataset

    if args.num_to_keep is not None:
        noise_dataset = noise_dataset + '-' + str(args.num_to_keep)

    if args.iterative:
        assert args.initial_seed is not None 
        assert args.val_size is not None
        assert args.num_iter is not None
        assert args.k is not None
        noise_dataset = "{}-{}-{}+{}+{}-{}" .format(noise_dataset, args.sampling_method, args.initial_seed, args.val_size , args.num_iter * args.k, args.optimizer)

    if args.true_model_test:
        true_model_test(args)

    if args.train_source_model:
        # Train our ground truth model.
        t = time.time()
        train(args)
        print("Training source model completed {} min".format(round((time.time() - t)/60, 2)))
        true_model_test(args)

    if args.copy_source_model:
        t = time.time()
        mea(args)
        print("Copying source model completed {} min".format(round((time.time() - t)/60, 2)))

    if args.true_model_test_on_noise_dataset:
        t = time.time()
        true_model_test_on_noise_dataset(args)
        print("True model test on noise dataset completed {} min".format(round((time.time() - t)/60, 2)))
        
    if args.optuna_search:
        optuna_search(args)
    
    if args.ssl:
        ssl(args)


main()

