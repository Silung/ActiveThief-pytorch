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

from utils.class_loader import *


def train(args):
    logdir = os.path.join(args.path_prefix, 'logdir', 
                          'source_model_', args.source_model, 
                          'true_dataset', args.true_dataset, 'true')
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
        model = source_model_type(num_classes, args.true_dataset, channels=channels)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_batch)
        
        vocab_size = train_dataset.get_vocab_size()
        model = source_model_type(num_classes, args.true_dataset, vocab_size=vocab_size)
        

    model = model.to(args.device)

    if args.optimizer == 'adagrad':
        optimizer  = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer  = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise Exception('Optimizer not be specified!')

    loss_func = nn.CrossEntropyLoss()

    best_acc = -1
    for epoch in range(args.num_epoch):
        model.train()
        train_dataloader = tqdm(train_dataloader)
        acc_list = []
        loss_list = []
        for items in train_dataloader:
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
            train_dataloader.set_description(f'Epoch: {epoch}')
            train_dataloader.set_postfix({'Loss': '{0:1.4f}'.format(loss.item()), 'ACC': '{0:1.4f}'.format(acc)})
            

        train_loss = np.array(loss_list).mean()
        train_acc = np.array(acc_list).mean()
        print('Epoch: {}\t Train Loss: {:.6}\t Train Acc: {:.6}\t'.format(epoch, train_loss, train_acc))
        writer1.add_scalar('train/loss', train_loss, epoch)
        writer1.add_scalar('train/acc', train_acc, epoch)

        save_dir = os.path.join(args.path_prefix, 'saved', args.source_model, args.true_dataset, 'true')
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
    val_dataloader = tqdm(val_dataloader)
    for items in val_dataloader:
        if args.true_dataset in ['agnews', 'imdb']:
            input, label, offset = items
            output = model(input.to(args.device), offset.to(args.device))
        else:
            input, label = items
            output = model(input.to(args.device))
        pred = torch.max(output, dim=-1, keepdim=False)[-1]
        
        acc = pred.cpu().eq(label.data).numpy().mean()
        acc_list.append(acc)
        val_dataloader.set_description('Evaluating')
        val_dataloader.set_postfix({'ACC': '{0:1.4f}'.format(acc)})
        
    val_acc = np.array(acc_list).mean()
    print('Val Acc: {:.6}\t'.format(val_acc))
    return val_acc

def true_model_test(args):
    save_dir = os.path.join(args.path_prefix, 'saved', args.source_model, args.true_dataset, 'true', 'trained_model.pth')

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

    acc_list = []
    test_dataloader = tqdm(test_dataloader)
    for items in test_dataloader:
        if args.true_dataset in ['agnews', 'imdb']:
            input, label, offset = items
            output = model(input.to(args.device), offset.to(args.device))
        else:
            input, label = items
            output = model(input.to(args.device))
        pred = torch.max(output, dim=-1, keepdim=False)[-1]
        
        acc = pred.cpu().eq(label.data).numpy().mean()
        acc_list.append(acc)
        test_dataloader.set_description('Testing')
        test_dataloader.set_postfix({'ACC': '{0:1.4f}'.format(acc)})
        
    test_acc = np.array(acc_list).mean()
    print('Test Acc: {:.6}\t'.format(test_acc))
    return test_acc