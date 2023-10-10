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


def true_model_test_on_noise_dataset(args):
    # dataset
    noise_dataset = load_noise_dataset(args.noise_dataset)

    val_noise_dataset = noise_dataset(mode='val', normalize_channels=True, val_frac=1)
    sample_shape = val_noise_dataset.get_sample_shape()
    width, height, channels = sample_shape
    num_classes = val_noise_dataset.get_num_classes()
    
    val_noise_dataloader = DataLoader(val_noise_dataset, batch_size=args.batch_size, shuffle=False)
    val_noise_dataloader = tqdm(val_noise_dataloader)

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

    true_model.eval()
    loss_list = []
    acc_list = []
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for trX, label, idx, p in val_noise_dataloader:
            trY = true_model(trX.to(args.device))
            label = label.to(args.device).to(torch.int64)
            loss = loss_func(trY, label)
            
            pred = torch.max(trY, dim=-1, keepdim=False)[-1]
            acc = pred.eq(label).cpu().numpy().mean()
        
            loss_list.append(loss.item())
            acc_list.append(acc)
            val_noise_dataloader.set_postfix({'Val Loss': '{0:1.4f}'.format(loss.item()), 'ACC': '{0:1.4f}'.format(acc)})
            
    val_loss = np.array(loss_list).mean()
    val_acc = np.array(acc_list).mean()
    print(f'True model\'s agreement on ture dataset and noise dataset : {val_acc}')
