import os
import logging
import shutil
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

import lightly
from lightly import loss as Loss
from lightly import transforms
from lightly.models.modules import heads
import lightly.data as data

from utils.class_loader import *
from utils.utils import f1_score


def train(args):
    logdir = os.path.join(args.path_prefix, 'logdir', 
                          'sm_' + args.source_model, 
                          'td_' + args.true_dataset, 
                          't_drop_' + str(args.train_dropout),
                          't_l2_' + str(args.train_l2),'true')
    shutil.rmtree(logdir, ignore_errors=True, onerror=None)
    writer1 = SummaryWriter(logdir)

    print("Training source model...")

    dataset = load_dataset(args.true_dataset) 

    train_dataset = dataset(mode='train')
    val_dataset = dataset(mode='val')

    num_classes = train_dataset.get_num_classes()
    source_model_type = load_model(args.source_model)
    
    if args.true_dataset not in ['agnews', 'imdb']:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        sample_shape = train_dataset.get_sample_shape()
        width, height, channels = sample_shape
        if args.train_dropout is not None:
            model = source_model_type(num_classes, args.true_dataset, channels, drop_prob=args.train_dropout)
        elif args.true_dataset == 'cifar':
            model = source_model_type(num_classes, args.true_dataset, channels, drop_prob=0.2)
        else:
            model = source_model_type(num_classes, args.true_dataset, channels)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_batch)
        
        vocab_size = train_dataset.get_vocab_size()
        model = source_model_type(num_classes, args.true_dataset, vocab_size=vocab_size)
        
    print(model)
    model = model.to(args.device)

    if args.optimizer == 'adagrad':
        optimizer  = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.train_l2)
    elif args.optimizer == 'adam':
        # optimizer  = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.train_l2)
        optimizer  = optim.Adam([
                {'params': model.fc.parameters()},
                {'params': model.conv_blocks.parameters(), 'weight_decay': args.train_l2}
            ], lr=args.lr, weight_decay=0)
    elif args.optimizer == 'sgd':
        optimizer  = optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise Exception('Optimizer not be specified!')

    loss_func = nn.CrossEntropyLoss()

    best_acc = -1
    for epoch in range(args.num_epoch):
        model.train()
        acc_list = []
        loss_list = []
        for items in tqdm(train_dataloader):
            optimizer.zero_grad()
            if args.true_dataset in ['agnews', 'imdb']:
                input, label, offset = items
                output = model(input.to(args.device), offset.to(args.device))
            else:
                input, label = items
                output = model(input.to(args.device))
            loss = loss_func(output, label.to(args.device).long())
            loss.backward()
            optimizer.step()

            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            
            acc = pred.cpu().eq(label.data).numpy().mean()
            acc_list.append(acc)
            loss_list.append(loss.item())
            # train_dataloader.set_description(f'Epoch: {epoch}')
            # train_dataloader.set_postfix({'Loss': '{0:1.4f}'.format(loss.item()), 'ACC': '{0:1.4f}'.format(acc)})
            

        train_loss = np.array(loss_list).mean()
        train_acc = np.array(acc_list).mean()
        print('Epoch: {}\t Train Loss: {:.6}\t Train Acc: {:.6}\t'.format(epoch, train_loss, train_acc))
        writer1.add_scalar('train/loss', train_loss, epoch)
        writer1.add_scalar('train/acc', train_acc, epoch)

        save_dir = os.path.join(args.path_prefix, 'saved', 
                                f'sm_{args.source_model}', 
                                f'td_{args.true_dataset}', 
                                f't_drop_{args.train_dropout}',
                                f't_l2_{args.train_l2}','true')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        acc = eval(args, model, val_dataloader)
        writer1.add_scalar('val/acc', acc, epoch)
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(save_dir, 'trained_model.pth'))
            best_acc = acc
    return model

def eval(args, model, val_dataloader):
    model.eval()

    acc_list = []
    for items in tqdm(val_dataloader):
        if args.true_dataset in ['agnews', 'imdb']:
            input, label, offset = items
            output = model(input.to(args.device), offset.to(args.device))
        else:
            input, label = items
            output = model(input.to(args.device))
        pred = torch.max(output, dim=-1, keepdim=False)[-1]
        
        acc = pred.cpu().eq(label.data).numpy().mean()
        acc_list.append(acc)
        # val_dataloader.set_description('Evaluating')
        # val_dataloader.set_postfix({'ACC': '{0:1.4f}'.format(acc)})
        
    val_acc = np.array(acc_list).mean()
    print('Val Acc: {:.6}\t'.format(val_acc))
    return val_acc

def true_model_test(args):
    save_dir = os.path.join(args.path_prefix, 'saved', 
                                f'sm_{args.source_model}', 
                                f'td_{args.true_dataset}', 
                                f't_drop_{args.train_dropout}',
                                f't_l2_{args.train_l2}','true','trained_model.pth')

    dataset = load_dataset(args.true_dataset) 
    test_dataset = dataset(mode='test')
    
    num_classes = test_dataset.get_num_classes()
    source_model_type = load_model(args.source_model)
    
    if args.true_dataset not in ['agnews', 'imdb']:
        sample_shape = test_dataset.get_sample_shape()
        width, height, channels = sample_shape
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        model = source_model_type(num_classes, args.true_dataset, channels)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_batch)
        vocab_size = test_dataset.get_vocab_size()
        model = source_model_type(num_classes, args.true_dataset, vocab_size=vocab_size)
        
    model.load_state_dict(torch.load(save_dir))
    model = model.to(args.device)
    model.eval()

    label_list = None
    pred_list = None
    for items in tqdm(test_dataloader):
        if args.true_dataset in ['agnews', 'imdb']:
            input, label, offset = items
            output = model(input.to(args.device), offset.to(args.device))
        else:
            input, label = items
            output = model(input.to(args.device))
        pred = torch.max(output, dim=-1, keepdim=False)[-1]
        
        if label_list is None:
            label_list = label
            pred_list = pred
        else:
            label_list = torch.cat([label_list, label], dim=0)
            pred_list = torch.cat([pred_list, pred], dim=0)
        # test_dataloader.set_description('Testing')
        # test_dataloader.set_postfix({'ACC': '{0:1.4f}'.format(acc)})
    
    test_acc = pred_list.eq(label_list).cpu().numpy().mean()
    test_f1 = f1_score(label_list.cpu(), pred_list.cpu(), num_classes)
    print('Test Acc: {:.6}\t Test F1: {:.6}'.format(test_acc, test_f1))
    return test_acc


def ssl(args):
    # Create a PyTorch module for the SimCLR model.
    class SimCLR(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()

            if hasattr(backbone, 'model'):
                model = backbone.model
            else:
                model = backbone
                
            self.projection_head = heads.SimCLRProjectionHead(
                input_dim=model.fc.in_features,
                hidden_dim=512,
                output_dim=128,
            )
            # Ignore the classification head as we only want the features.
            model.fc = torch.nn.Identity()
            self.backbone = backbone

        def forward(self, x):
            features = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(features)
            return z
        
    
    if 'minst' in args.true_dataset:
        img_size = 28
    elif 'imagenet' in args.true_dataset:
        img_size = 64
    elif 'cifar' in args.true_dataset:
        img_size = 32
    else:
        img_size = 32
    
    # copy dataset
    noise_dataset = load_dataset(args.noise_dataset, markable=False)
    if args.noise_dataset == 'mnist_dist':
        train_noise_dataset = noise_dataset(mode='train', normalize_channels=True, num_fig=args.num_fig, normalize=False)
    elif 'mnist' in args.noise_dataset:
        train_noise_dataset = noise_dataset(mode='train', normalize_channels=True, normalize=False)
    elif 'imagenet' in args.noise_dataset:
        train_noise_dataset = noise_dataset(mode='train', normalize=False, num_train_batch=args.num_train_batch)
    else:
        train_noise_dataset = noise_dataset(mode='train', normalize=False)

    if args.noise_dataset not in ['agnews', 'imdb']:
        sample_shape = train_noise_dataset.get_sample_shape()
        width, height, channels = sample_shape
        args.resize = (width, height)
        
    num_classes = train_noise_dataset.get_num_classes()

    transform =T.Compose([T.ToPILImage(),
                          transforms.SimCLRTransform(input_size=img_size, cj_prob=0.5, gaussian_blur=0)
                        ])
    
    train_noise_dataset = data.LightlyDataset.from_torch_dataset(train_noise_dataset, transform=transform)
    train_noise_dataloader = DataLoader(train_noise_dataset, batch_size=args.batch_size, shuffle=True)

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
        copy_model = copy_model_type()

    # Build the SimCLR model.
    model = SimCLR(copy_model).to(args.device)

    # Lightly exposes building blocks such as loss functions.
    criterion = Loss.NTXentLoss(temperature=0.5)

    # Get a PyTorch optimizer.
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)

    
    save_dir = os.path.join(args.path_prefix, 'saved', 
                            f'sm_{args.source_model}', 
                            f'td_{args.true_dataset}', 
                            f't_drop_{args.train_dropout}',
                            f't_l2_{args.train_l2}','ssl_pretrain')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Train the model.
    min_loss = np.inf
    for epoch in range(args.num_epoch):
        print(f'Epoch {epoch}')
        for (trX0, trX1), trY, _ in tqdm(train_noise_dataloader):
            trX0 = trX0.permute(0,2,3,1).to(args.device)
            trX1 = trX1.permute(0,2,3,1).to(args.device)
            z0 = model(trX0)
            z1 = model(trX1)
            loss = criterion(z0, z1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"loss: {loss.item():.5f}")
        
        if loss.item() < min_loss:
            torch.save(model.backbone.state_dict(), os.path.join(save_dir, f'{args.copy_model}.pth'))
            min_loss = loss.item()