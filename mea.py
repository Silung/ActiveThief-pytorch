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
from al.uncertainty import UncertaintySelectionStrategy, CertaintySelectionStrategy
from al.kcenter import KCenterGreedyApproach

from utils.early_stop import EarlyStopping
from utils.utils import f1_score

import matplotlib.pyplot as plt
from collections import Counter

def show(tensor):
    # 将张量的形状转换为（28, 28）
    tensor = tensor.view(28, 28)

    # 显示灰度图像
    plt.imshow(tensor, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def mark_dataset(args, model, dataloader, budget=None):
    model.eval()
    dataset = dataloader.dataset
    dataset.set_state('unmark')
    if budget is None:
        budget = len(dataset)
    for i in range(budget):
        dataset.mark(i)
    
    lebel_dataset(args, model, dataloader)
    
        
def lebel_dataset(args, model, dataloader):
    dataset = dataloader.dataset
    dataset.set_state('marking')
    model.eval()
    update_info = {}
    with torch.no_grad():
        print('Labeling dataset...')
        for trX, l, idx, _ in tqdm(dataloader):
            trY = model(trX.to(args.device))
            for i, y in enumerate(trY):
                update_info[idx[i].item()] = y
        for i, y in update_info.items():
            dataset.update(i, aux_data=y)
    dataset.set_state('marked')
    
def load_true_data(args):
    dataset = load_dataset(args.true_dataset, markable=True)
    val_dataset = dataset(mode='val')
    test_dataset = dataset(mode='test')
    
    if args.true_dataset not in ['agnews', 'imdb']:
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_batch)
    return val_dataset, test_dataset, val_dataloader, test_dataloader

def load_noise_data(args):
    noise_dataset = load_dataset(args.noise_dataset, markable=True)
    if not hasattr(args, 'resize'):
        args.resize = None
    if args.noise_dataset == 'mnist_dist':
        train_noise_dataset = noise_dataset(mode='train', resize=args.resize, normalize_channels=True, num_fig=args.num_fig)
        val_noise_dataset = noise_dataset(mode='val', resize=args.resize, normalize_channels=True, num_fig=args.num_fig)
    elif 'mnist' in args.true_dataset:
        train_noise_dataset = noise_dataset(mode='train', resize=args.resize, normalize_channels=True)
        val_noise_dataset = noise_dataset(mode='val', resize=args.resize, normalize_channels=True)
    else:
        train_noise_dataset = noise_dataset(mode='train', resize=args.resize)
        val_noise_dataset = noise_dataset(mode='val', resize=args.resize)
    
    train_noise_dataloader = DataLoader(train_noise_dataset, batch_size=args.batch_size, shuffle=True)
    val_noise_dataloader = DataLoader(val_noise_dataset, batch_size=args.batch_size, shuffle=False)
    return train_noise_dataset, val_noise_dataset, train_noise_dataloader, val_noise_dataloader

def mea(args):
    # true dataset
    val_dataset, test_dataset, val_dataloader, test_dataloader = load_true_data(args)
    if args.true_dataset not in ['agnews', 'imdb']:
        sample_shape = val_dataset.get_sample_shape()
        width, height, channels = sample_shape
        args.resize = (width, height)
    
    num_classes = val_dataset.get_num_classes()
    
    # copy dataset
    train_noise_dataset, val_noise_dataset, train_noise_dataloader, val_noise_dataloader = load_noise_data(args)
    
    # true model
    true_model_dir = os.path.join(args.path_prefix, 'saved', 
                                  f'sm_{args.source_model}', 
                                  f'td_{args.true_dataset}', 
                                  f't_drop_{args.train_dropout}',
                                  f't_l2_{args.train_l2}','true')
    if not os.path.exists(true_model_dir):
        print('Train true model first!')
    source_model_type = load_model(args.source_model)
    if args.true_dataset not in ['agnews', 'imdb']:
        if args.mea_dropout is not None:
            true_model = source_model_type(num_classes, args.true_dataset, channels, drop_prob=args.mea_dropout)
        elif args.true_dataset == 'cifar':
            true_model = source_model_type(num_classes, args.true_dataset, channels, drop_prob=0.2)
        else:
            true_model = source_model_type(num_classes, args.true_dataset, channels)
    else:
        vocab_size = train_noise_dataset.get_vocab_size()
        true_model = source_model_type(num_classes, args.true_dataset, vocab_size=vocab_size)
    print(true_model)
    true_model.load_state_dict(torch.load(os.path.join(true_model_dir, 'trained_model.pth')))
    true_model = true_model.to(args.device)

    # Log dir 
    logdir_copy = os.path.join(args.path_prefix, 'logdir', 
                               'sm_' + args.source_model, 
                               'td_' + args.true_dataset,
                               f't_drop_{args.train_dropout}',
                               f't_l2_{args.train_l2}',
                               'cm_' + args.copy_model + ('' if args.pretrain is None else '_pretrain'),
                               'nd_' + args.noise_dataset,
                               'api_' + args.api_retval,
                               'sampling_' + args.sampling_method,
                               'lr_' + str(args.lr),
                               'iter_' + str(args.num_iter), 
                               'k_' + str(args.k), 
                               'm_drop_' + str(args.mea_dropout),
                               f'm_l2_{args.mea_l2}',
                               f'pat_{args.patience}')
    # logdir_papernot_copy = os.path.join(args.path_prefix, 'logdir', args.source_model, args.true_dataset, str(args.num_iter), str(args.k), args.api_retval, args.copy_model, 'papernot', args.sampling_method)
    
    print("deleting the dir {}".format(logdir_copy))
    shutil.rmtree(logdir_copy, ignore_errors=True, onerror=None)
    writer2 = SummaryWriter(logdir_copy)
    print("Copying source model using iterative approach")

    used_budget = 0
    # 标记true val dataset并查询
    mark_dataset(args, true_model, val_dataloader)
    # 标记true test dataset并查询
    mark_dataset(args, true_model, test_dataloader)
    print(f'True val dataset size: {len(val_dataset)}')
    print(f'True test dataset size: {len(test_dataset)}')

    # 初始标记 val noise dataset并查询
    mark_dataset(args, true_model, val_noise_dataloader, int(0.2*args.initial_size))
    # 初始标记train dataset S0
    mark_dataset(args, true_model, train_noise_dataloader, int(0.8*args.initial_size))
    used_budget += args.initial_size

    loss_func = nn.CrossEntropyLoss()
    
    max_entropy = entropy(torch.ones((val_dataset.get_num_classes())))
    
    step = 0
    weighted_val_score = 0
    for it in range(args.num_iter):
        # copy model
        copy_model_type = load_model(args.copy_model)
        if args.copy_model.startswith('cnn'):
            if args.mea_dropout is not None:
                copy_model = copy_model_type(in_channels=channels, num_classes=num_classes, dataset_name=args.true_dataset, fc_layers=[], drop_prob=args.mea_dropout).to(args.device)
            elif args.true_dataset == 'cifar':
                copy_model = copy_model_type(in_channels=channels, num_classes=num_classes, dataset_name=args.true_dataset, fc_layers=[], drop_prob=0.2).to(args.device)
            else:
                copy_model = copy_model_type(in_channels=channels, num_classes=num_classes, dataset_name=args.true_dataset, fc_layers=[]).to(args.device)
        elif args.copy_model.startswith('resnet'):
            copy_model = copy_model_type(num_classes=num_classes)

        if args.pretrain is not None:
            parms = torch.load(args.pretrain)
            if 'fc.weight' in parms:
                del parms['fc.weight']
            if 'fc.bias' in parms:
                del parms['fc.bias']
            # if args.true_dataset == 'mnist':
            #     del parms['conv_blocks.0.conv_blocks.0.weight']
            copy_model.load_state_dict(parms, strict=False)
        copy_model.to(args.device)
        # optimizer  = optim.Adam(copy_model.parameters(), lr=args.lr, weight_decay=args.mea_l2)
        if args.optimizer == 'adam':
            opti = optim.Adam
        elif args.optimizer == 'sgd':
            opti = optim.SGD
        if args.copy_model.startswith('cnn'):
            if args.lr > 0:
                optimizer  = opti([
                    {'params': copy_model.fc.parameters()},
                    {'params': copy_model.conv_blocks.parameters(), 'weight_decay': args.train_l2}
                    ], lr=args.lr)
            else:
                optimizer  = opti([
                    {'params': copy_model.fc.parameters()},
                    {'params': copy_model.conv_blocks.parameters(), 'weight_decay': args.train_l2}
                    ])
        elif args.copy_model.startswith('resnet'):
            optimizer  = opti([{"params":copy_model.model.layer4.parameters()},
                                     {"params":copy_model.model.fc.parameters()}], weight_decay=args.train_l2)
            

        # 训练替代模型
        copy_model.train()
        train_noise_dataset.set_state('marked')
        val_noise_dataset.set_state('marked')
        print(f'Marked Train Noise Dataset Size: {len(train_noise_dataset)}')
        print(f'Marked Val Noise Dataset Size: {len(val_noise_dataset)}')
        print(f'Used Budget: {used_budget}')
        loss_list = []
        acc_list = []
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, trace_func=None)
        # early_stopping = EarlyStopping(patience=300, verbose=True, trace_func=None, delta=1.11e-5)
        for epoch in trange(args.num_epoch):
            for trX, _, idx, p in train_noise_dataloader:
                optimizer.zero_grad()
                trY = copy_model(trX.to(args.device))
                label = torch.max(p.to(args.device), dim=-1, keepdim=False)[-1]
                if args.api_retval == 'onehot':
                    # loss = loss_func(trY, label) + 5*max_entropy/entropy(torch.mean(trY, dim=0))
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
            
            # "We set aside 20% of the query budget for validation"
            val_noise_loss, val_noise_acc, val_noise_f1 = eval(args, copy_model, val_noise_dataloader, print_result=False)
            # print(val_noise_acc)
            early_stopping(val_noise_f1, np.mean(loss_list), copy_model)
            # early_stopping(val_noise_loss, copy_model)
            
            # if epoch % 20 == 0:
            #     test_loss, test_acc, test_f1 = eval(args, copy_model, test_dataloader)
                # print(loss)
                # print(label)
                # print(pred)
                # print(val_noise_acc)
            if early_stopping.early_stop:
                writer2.add_scalar('noise_data_val/loss', val_noise_loss, used_budget)
                writer2.add_scalar('noise_data_val/aggrement', val_noise_acc, used_budget)
                writer2.add_scalar('noise_data_val/f1', val_noise_f1, used_budget)
                # print("Early Stop!")
                break
            step += 1
        
        weighted_val_score += 1/args.num_iter * val_noise_acc
        copy_model.load_state_dict(early_stopping.best_model_parms)

        train_loss = np.array(loss_list).mean()
        train_acc = np.array(acc_list).mean()
        print('Copy model iter: {}\t Train Loss: {:.6}\t Train Agr: {:.6}\t'.format(it, train_loss, train_acc))
        print('Val Loss: {:.6}\t Val Agr: {:.2}\t Val F1: {:.2}\t'.format(val_noise_loss, val_noise_acc, val_noise_f1))
        writer2.add_scalar('noise_data_train/loss', train_loss, used_budget)
        writer2.add_scalar('noise_data_train/aggrement', train_acc, used_budget)
        
        # test
        print('Test on true dataset')
        test_loss, test_acc, test_f1 = eval(args, copy_model, test_dataloader)
        writer2.add_scalar('true_data_test/loss', test_loss, used_budget)
        writer2.add_scalar('true_data_test/aggrement', test_acc, used_budget)
        writer2.add_scalar('true_data_test/f1', test_f1, used_budget)
        
        if it == args.num_iter - 1:
            break
            
        # 使用替代模型查询剩余未标记样本标签
        copy_model.eval()
        train_noise_dataset.set_state('unmark')
        val_noise_dataset.set_state('unmark')
        
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
        b = int(0.8*args.k)
        if args.sampling_method == 'random':
            sss = RandomSelectionStrategy(b, Idx, Y)
        elif args.sampling_method == 'uncertainty':
            sss = UncertaintySelectionStrategy(b, Idx, F.softmax(Y.cpu(),dim=-1))
        elif args.sampling_method == 'certainty':
            sss = CertaintySelectionStrategy(b, Idx, F.softmax(Y.cpu(),dim=-1))
        elif args.sampling_method == 'kcenter':
            prob = train_noise_dataset.aux_data.values()
            true_points = torch.concat(list(prob), dim=0).reshape(len(prob), -1)
            sss = KCenterGreedyApproach(b, Idx, Y, true_points, args.batch_size)
        # elif args.sampling_method == 'deepfool':
        #     sss = AdversarialSelectionStrategy(args.k, Idx, Y)
        s = sss.get_subset()
        for i in s:
            train_noise_dataset.mark(i)
        lebel_dataset(args, true_model, train_noise_dataloader)
        for i in range(int(0.2*used_budget), int(0.2*(used_budget + args.k))):
            val_noise_dataset.mark(i)
        lebel_dataset(args, true_model, val_noise_dataloader)
        used_budget += args.k

    print("---Copynet trainning completed---")
    return weighted_val_score

def entropy(vector):
    # 将向量归一化为概率分布
    probabilities = F.softmax(vector, dim=0)
    # 计算信息熵
    entropy_value = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy_value

def eval(args, model, val_dataloader, print_result=True):
    val_dataloader.dataset.set_state('marked')
    model.eval()
    loss_list = []
    label_list = None
    pred_list = None
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for trX, _, idx, p in val_dataloader:
            trY = model(trX.to(args.device))
            label = torch.max(p.to(args.device), dim=-1, keepdim=False)[-1]
            loss = loss_func(trY, label)
            
            pred = torch.max(trY, dim=-1, keepdim=False)[-1]
            acc = pred.eq(label).cpu().numpy().mean()
        
            loss_list.append(loss.item())
            
            if label_list is None:
                label_list = label
                pred_list = pred
            else:
                label_list = torch.cat([label_list, label], dim=0)
                pred_list = torch.cat([pred_list, pred], dim=0)
            
            # val_dataloader.set_postfix({'Val Loss': '{0:1.4f}'.format(loss.item()), 'ACC': '{0:1.4f}'.format(acc)})
            
    val_loss = np.array(loss_list).mean()
    val_acc = pred_list.eq(label_list).cpu().numpy().mean()
    # if isinstance(val_dataloader, tqdm):
    #     num_class = val_dataloader.iterable.dataset.get_num_classes()
    # else:
    #     num_class = val_dataloader.dataset.get_num_classes()
    if 'cifar' in args.true_dataset:
        num_class = 10
    elif 'mnist' in args.true_dataset:
        num_class = 10
    elif 'gtsrb' in args.true_dataset:
        num_class = 43
    elif 'imagenet' in args.true_dataset:
        num_class = 1000
    val_f1 = f1_score(label_list.cpu(), pred_list.cpu(), num_class)
    if print_result:
        print('Loss: {:.6}\t Agr: {:.6}\t F1: {:.6}\t'.format(val_loss, val_acc, val_f1))
        cnt = Counter(label_list.cpu().numpy())
        print('label_list counter:')
        print(cnt)
        cnt = Counter(pred_list.cpu().numpy())
        print('pred_list counter:')
        print(cnt)

    return val_loss, val_acc, val_f1