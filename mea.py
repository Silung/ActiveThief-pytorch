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

def mea(args):
    # true dataset
    dataset = load_dataset(args.true_dataset)
    train_dataset = dataset(mode='val')
    sample_shape = train_dataset.get_sample_shape()
    width, height, channels = sample_shape
    resize = (width, height)
    num_classes = train_dataset.get_num_classes()
    
    # true model
    true_model_dir = os.path.join(args.path_prefix, 'saved', args.source_model, args.true_dataset, 'true')
    if not os.path.exists(true_model_dir):
        print('Train true model first!')
    source_model_type = load_model(args.source_model)
    true_model = source_model_type(channels, num_classes, args.true_dataset)
    true_model.load_state_dict(torch.load(os.path.join(true_model_dir, 'trained_model.pth')))
    true_model = true_model.to(args.device)

    # Log dir 
    logdir_copy = os.path.join(args.path_prefix, 'logdir' , args.source_model, args.true_dataset, args.api_retval, args.copy_model, args.noise_dataset, args.sampling_method)
    logdir_papernot_copy = os.path.join(args.path_prefix, 'logdir', args.source_model, args.true_dataset, args.api_retval, args.copy_model, 'papernot', args.sampling_method)
    
    print("deleting the dir {}".format(logdir_copy))
    shutil.rmtree(logdir_copy, ignore_errors=True, onerror=None)
    writer2 = SummaryWriter(logdir_copy)
    print("Copying source model using iterative approach")

    # copy dataset
    noise_dataset = load_noise_dataset(args.noise_dataset)
    train_noise_dataset = noise_dataset(mode='train', resize=resize, normalize_channels=True)
    val_noise_dataset = noise_dataset(mode='val', resize=resize, normalize_channels=True)
    
    train_noise_dataloader = DataLoader(train_noise_dataset, batch_size=args.batch_size, shuffle=True)
    val_noise_dataloader = DataLoader(val_noise_dataset, batch_size=args.batch_size, shuffle=False)
    val_noise_dataloader = tqdm(val_noise_dataloader)

    # copy model
    copy_model_type = load_model(args.copy_model)
    copy_model = copy_model_type(in_channels=channels, num_classes=num_classes, dataset_name='cifar', fc_layers=[]).to(args.device)
    
    # 标记val dataset并查询
    val_noise_dataset.set_state('unmark')
    val_dataset_size = len(val_noise_dataset)
    for i in range(val_dataset_size):
        val_noise_dataset.mark(i)
        
    val_noise_dataset.set_state('marking')
    true_model.eval()
    update_info = {}
    with torch.no_grad():
        for trX, _, idx, _ in val_noise_dataloader:
            trY = true_model(trX.to(args.device))
            for i, y in enumerate(trY):
                update_info[idx[i].item()] = y
        for i, y in update_info.items():
            val_noise_dataset.update(i, aux_data=y)
    
    # val dataset统计信息
    val_label_counts = dict(list(enumerate([0] * num_classes)))
    
    # 初始标记train dataset S0 
    for i in range(args.initial_size):
        train_noise_dataset.mark(i)

    optimizer  = optim.Adam(copy_model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    
    for it in range(args.num_iter):
        # 查询标记
        true_model.eval()
        train_noise_dataset.set_state('marking')
        update_info = {}
        with torch.no_grad():
            for trX, _, idx, _ in train_noise_dataloader:
                trY = true_model(trX.to(args.device))
                for i, y in enumerate(trY):
                    update_info[idx[i].item()] = y
            for i, y in update_info.items():
                train_noise_dataset.update(i, aux_data=y)

        # 训练替代模型
        copy_model.train()
        train_noise_dataset.set_state('marked')
        print(f'Marked Dataset Size: {len(train_noise_dataset)}')
        loss_list = []
        acc_list = []
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
                
        train_loss = np.array(loss_list).mean()
        train_acc = np.array(acc_list).mean()
        print('Copy model iter: {}\t Train Loss: {:.6}\t Train Acc: {:.6}\t'.format(it, train_loss, train_acc))
        writer2.add_scalar('copy_train/loss', train_loss, it)
        writer2.add_scalar('copy_train/acc', train_acc, it)
        
        # val
        val_noise_dataset.set_state('marked')
        val_loss, val_acc = eval(args, copy_model, val_noise_dataloader)
        writer2.add_scalar('copy_val/loss', val_loss, it)
        writer2.add_scalar('copy_val/acc', val_acc, it)
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


def eval(args, model, val_dataloader):
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
    print('Val Loss: {:.6}\t Val Acc: {:.6}\t'.format(val_loss, val_acc))

    return val_loss, val_acc

def copy_model_test(args):
    pass