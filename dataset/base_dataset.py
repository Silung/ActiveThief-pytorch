import numpy as np
import random, cv2, os
import logging
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, normalize=True, mode='train', val_frac=None, normalize_channels=False, resize=None):
        self.random_state = random.getstate()
        
        print("Loading {} data".format(mode))
        # Initializes self.data and self.labels
        self.load_data(mode, val_frac)

        # Resize dataset if it is not already of the specified size
        if resize is not None and not self.data.shape[1:3] == resize:
            print('Resizing...')
            data = np.empty((self.data.shape[0], resize[0], resize[1], self.data.shape[3]))

            for i, image in enumerate(self.data):
                if self.data.shape[3] == 1:
                    data[i,:,:,0] = cv2.resize(image.squeeze(), resize)
                else:
                    data[i,:,:,:] = cv2.resize(image, resize)

            self.data = data

        # Normalize color channels if requested
        if normalize_channels:
            self.data  = np.mean(self.data, axis=-1)
            self.data  = np.expand_dims(self.data, -1)

            assert len(self.data.shape) == 4
        
        # Normalize data to lie in [0,1]
        if normalize:
            self.data = self.data/float(np.max(self.data))
            
            assert np.abs(np.min(self.data) - 0.0) < 1e-1
            assert np.abs(np.max(self.data) - 1.0) < 1e-1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y
    
    def is_multilabel(self):
        raise NotImplementedError
    
    def get_sample_shape(self):
        return self.data.shape[1:]
    
    def get_num_classes(self):
        raise NotImplementedError
        
    def load_data(self):
        raise NotImplementedError

    def partition_validation_set(self, mode, val_frac):
        train_end = int(len(self.data)*(1-val_frac))
        
        if mode == 'train':
            self.data = self.data[:train_end]
            
            if self.labels is not None:
                self.labels = self.labels[:train_end]
        elif mode == 'val':
            self.data = self.data[train_end:]
            
            if self.labels is not None:
                self.labels = self.labels[train_end:]