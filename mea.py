import os
import shutil
import numpy as np
from tqdm import tqdm, trange
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

from utils.class_loader import *
from dataset.uniform_dataset import UniformDataset
from dataset.imagenet_dataset import ImagenetDataset

from al.random import RandomSelectionStrategy
# from al.adversarial import AdversarialSelectionStrategy
from al.uncertainty import UncertaintySelectionStrategy
from al.kcenter import KCenterGreedyApproach

from utils.early_stop import EarlyStopping

import matplotlib.pyplot as plt

def show(tensor):
    # 将张量的形状转换为（28, 28）
    tensor = tensor.view(28, 28)

    # 显示灰度图像
    plt.imshow(tensor, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def mark_dataset(args, model, dataloader):
    if isinstance(dataloader, tqdm):
        dataset = dataloader.iterable.dataset
    else:
        dataset = dataloader.dataset
    dataset.set_state('unmark')
    for i in range(len(dataset)):
        dataset.mark(i)
        
    dataset.set_state('marking')
    
    model.eval()
    update_info = {}
    with torch.no_grad():
        print('Labeling dataset...')
        for trX, l, idx, _ in dataloader:
            trY = model(trX.to(args.device))
            for i, y in enumerate(trY):
                update_info[idx[i].item()] = y
        for i, y in update_info.items():
            dataset.update(i, aux_data=y)
            
    dataset.set_state('marked')

def mea(args):
    # true dataset
    dataset = load_dataset(args.true_dataset, markable=True)
    val_dataset = dataset(mode='val')
    test_dataset = dataset(mode='test')
    
    if args.true_dataset not in ['agnews', 'imdb']:
        val_dataloader = tqdm(DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False))
        test_dataloader = tqdm(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False))
        sample_shape = val_dataset.get_sample_shape()
        width, height, channels = sample_shape
        resize = (width, height)
    else:
        val_dataloader = tqdm(DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_batch))
        test_dataloader = tqdm(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_batch))
    num_classes = val_dataset.get_num_classes()
    
    # true model
    true_model_dir = os.path.join(args.path_prefix, 'saved', args.source_model, args.true_dataset, 'true')
    if not os.path.exists(true_model_dir):
        print('Train true model first!')
    source_model_type = load_model(args.source_model)
    if args.true_dataset not in ['agnews', 'imdb']:
        true_model = source_model_type(num_classes, args.true_dataset, channels)
    else:
        vocab_size = train_dataset.get_vocab_size()
        model = source_model_type(num_classes, args.true_dataset, vocab_size=vocab_size)
        true_model = source_model_type(num_classes, args.true_dataset, channels)
    true_model.load_state_dict(torch.load(os.path.join(true_model_dir, 'trained_model.pth')))
    true_model = true_model.to(args.device)

    # Log dir 
    logdir_copy = os.path.join(args.path_prefix, 'logdir', 
                               'source_model_' + args.source_model, 
                               'true_dataset_' + args.true_dataset, 
                               'copy_model_' + args.copy_model,
                               'noise_dataset_' + args.noise_dataset,
                               'api_retval_' + args.api_retval,
                               'sampling_method_' + args.sampling_method,
                               'lr_' + str(args.lr),
                               'num_iter_' + str(args.num_iter), 
                               'k_' + str(args.k))
    # logdir_papernot_copy = os.path.join(args.path_prefix, 'logdir', args.source_model, args.true_dataset, str(args.num_iter), str(args.k), args.api_retval, args.copy_model, 'papernot', args.sampling_method)
    
    print("deleting the dir {}".format(logdir_copy))
    shutil.rmtree(logdir_copy, ignore_errors=True, onerror=None)
    writer2 = SummaryWriter(logdir_copy)
    print("Copying source model using iterative approach")

    # copy dataset
    noise_dataset = load_dataset(args.noise_dataset, markable=True)
    if args.noise_dataset == 'mnist_dist':
        train_noise_dataset = noise_dataset(mode='train', resize=resize, normalize_channels=True, num_fig=args.num_fig)
        val_noise_dataset = noise_dataset(mode='val', resize=resize, normalize_channels=True, num_fig=args.num_fig)
    elif 'mnist' in args.true_dataset:
        train_noise_dataset = noise_dataset(mode='train', resize=resize, normalize_channels=True)
        val_noise_dataset = noise_dataset(mode='val', resize=resize, normalize_channels=True)
    else:
        train_noise_dataset = noise_dataset(mode='train', resize=resize)
        val_noise_dataset = noise_dataset(mode='val', resize=resize)
    
    train_noise_dataloader = tqdm(DataLoader(train_noise_dataset, batch_size=args.batch_size, shuffle=True))
    val_noise_dataloader = tqdm(DataLoader(val_noise_dataset, batch_size=args.batch_size, shuffle=True))

    # copy model
    copy_model_type = load_model(args.copy_model)
    if args.true_dataset == 'cifar':
        copy_model = copy_model_type(in_channels=channels, num_classes=num_classes, dataset_name=args.true_dataset, fc_layers=[], drop_prob=0.2).to(args.device)
    else:
        copy_model = copy_model_type(in_channels=channels, num_classes=num_classes, dataset_name=args.true_dataset, fc_layers=[]).to(args.device)
    
    # 标记val noise dataset并查询
    mark_dataset(args, true_model, val_noise_dataloader)
    # 标记true val dataset并查询
    mark_dataset(args, true_model, val_dataloader)
    # 标记true test dataset并查询
    mark_dataset(args, true_model, test_dataloader)

    # 初始标记train dataset S0 
    for i in range(args.initial_size):
        train_noise_dataset.mark(i)

    optimizer  = optim.Adam(copy_model.parameters(), lr=args.lr, weight_decay=0.001)
    loss_func = nn.CrossEntropyLoss()
    
    step = 0
    for it in range(args.num_iter):
        # 查询标记
        true_model.eval()
        train_noise_dataset.set_state('marking')
        update_info = {}
        with torch.no_grad():
            for trX, y, idx, _ in train_noise_dataloader:
                # for i in trX:
                #     show(i)
                trY = true_model(trX.to(args.device))
                for i, y in enumerate(trY):
                    update_info[idx[i].item()] = y
            for i, y in update_info.items():
                train_noise_dataset.update(i, aux_data=y)

        # 训练替代模型
        copy_model.train()
        train_noise_dataset.set_state('marked')
        query_count = len(train_noise_dataset)
        print(f'Marked Dataset Size: {query_count}')
        loss_list = []
        acc_list = []
        early_stopping = EarlyStopping(patience=100, verbose=True, trace_func=None)
        for epoch in trange(args.num_epoch):
            for trX, _, idx, p in train_noise_dataloader:
                optimizer.zero_grad()
                trY = copy_model(trX.to(args.device))
                label = torch.max(p.to(args.device), dim=-1, keepdim=False)[-1]
                if args.api_retval == 'onehot':
                    loss = loss_func(trY, label)
                elif args.api_retval == 'softmax':
                    loss = loss_func(trY, p.to(args.device).softmax(dim=-1))
                else:
                    raise NotImplementedError
                loss.backward()
                optimizer.step()
                
                pred = torch.max(trY, dim=-1, keepdim=False)[-1]
                acc = pred.eq(label).cpu().numpy().mean()
            
                loss_list.append(loss.item())
                acc_list.append(acc)
                # print(f'Iter: {it}\t Epoch: {epoch}\t Loss: {loss.item()}\t ACC: {acc}')
            
            # val on noise
            # if step % 10 == 0:
                # writer2.add_scalar('copy_val/loss_on_noise', val_noise_loss, step)
                # writer2.add_scalar('copy_val/aggrement_on_noise', val_noise_acc, step)
            
            val_noise_loss, val_noise_acc = eval(args, copy_model, val_noise_dataloader, print_result=False)
            early_stopping(val_noise_loss, copy_model)
            if early_stopping.early_stop:
                writer2.add_scalar('copy_val/loss_on_noise', val_noise_loss, query_count)
                writer2.add_scalar('copy_val/aggrement_on_noise', val_noise_acc, query_count)
                # print("Early Stop!")
                break
            
            step += 1
                
        train_loss = np.array(loss_list).mean()
        train_acc = np.array(acc_list).mean()
        print('Copy model iter: {}\t Train Loss: {:.6}\t Train Acc: {:.6}\t'.format(it, train_loss, train_acc))
        writer2.add_scalar('copy_train/loss', train_loss, query_count)
        writer2.add_scalar('copy_train/aggrement', train_acc, query_count)
        
        # val
        # print('Eval on true dataset')
        # val_loss, val_acc = eval(args, copy_model, val_dataloader)
        # writer2.add_scalar('copy_val/loss_on_true', val_loss, query_count)
        # writer2.add_scalar('copy_val/aggrement_on_true', val_acc, query_count)
        
        # test
        print('Test on true dataset')
        test_loss, test_acc = eval(args, copy_model, test_dataloader)
        writer2.add_scalar('copy_test/loss', test_loss, query_count)
        writer2.add_scalar('copy_test/aggrement', test_acc, query_count)
        
        if it == args.num_iter:
            break
            
        # 使用替代模型查询剩余未标记样本标签
        copy_model.eval()
        train_noise_dataset.set_state('unmark')
        
        Y = None
        Idx = []
        with torch.no_grad():
            for x, _, idx, _ in train_noise_dataloader:
                y = copy_model(x.to(args.device))
                if Y is None:
                    Y = y
                else:
                    Y = torch.concat([Y, y], dim=0)
                Idx += idx.tolist()
            
        # Active Learning策略
        if args.sampling_method == 'random':
            sss = RandomSelectionStrategy(args.k, Idx, Y)
        elif args.sampling_method == 'uncertainty':
            sss = UncertaintySelectionStrategy(args.k, Idx, F.softmax(Y.cpu(),dim=-1))
        elif args.sampling_method == 'kcenter':
            prob = train_noise_dataset.aux_data.values()
            true_points = torch.concat(list(prob), dim=0).reshape(len(prob), -1)
            sss = KCenterGreedyApproach(args.k, Idx, Y, true_points, args.batch_size)
        # elif args.sampling_method == 'deepfool':
        #     sss = AdversarialSelectionStrategy(args.k, Idx, Y)
        s = sss.get_subset()
        for i in s:
            train_noise_dataset.mark(i)

    print("---Copynet trainning completed---")


def eval(args, model, val_dataloader, print_result=True):
    model.eval()
    loss_list = []
    acc_list = []
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for trX, _, idx, p in val_dataloader:
            trY = model(trX.to(args.device))
            label = torch.max(p.to(args.device), dim=-1, keepdim=False)[-1]
            loss = loss_func(trY, label)
            
            pred = torch.max(trY, dim=-1, keepdim=False)[-1]
            acc = pred.eq(label).cpu().numpy().mean()
        
            loss_list.append(loss.item())
            acc_list.append(acc)
            val_dataloader.set_postfix({'Val Loss': '{0:1.4f}'.format(loss.item()), 'ACC': '{0:1.4f}'.format(acc)})
            
    val_loss = np.array(loss_list).mean()
    val_acc = np.array(acc_list).mean()
    if print_result:
        print('Val Loss: {:.6}\t Val Acc: {:.6}\t'.format(val_loss, val_acc))

    return val_loss, val_acc
