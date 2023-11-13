import numpy as np
import random, cv2, os
import logging
import torch
from torch.utils.data import Dataset
from torch import long, as_tensor, cat
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

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
            self.data = (self.data - float(np.min(self.data)))/(float(np.max(self.data)) - float(np.min(self.data)))
            # self.data = (self.data - np.min(self.data))/(np.max(self.data) - np.min(self.data))
            
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
        
    def load_data(self, mode, val_frac):
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

class BaseTextDataset(Dataset):
    def __init__(self, mode='train', val_frac=None, embedding_type='word2vec'):
        print("Loading {} data".format(mode))
        # Initializes self.data and self.labels
        self.load_vocab()
        self.load_data(mode, val_frac)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y
    
    def load_vocab(self):
        data = self.dataset(split='train')
        text_data = [text for _, text in data]

        self.tokenizer = get_tokenizer('basic_english')

        def yield_tokens(data_iter):
            for text in data_iter:
                yield self.tokenizer(text)

        self.vocab = build_vocab_from_iterator(yield_tokens(text_data), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.vocab_size = len(self.vocab)
    
    def load_data(self, mode, val_frac):
        data = self.dataset(split='train' if mode == 'val' else mode)
        self.data, self.labels = [text for _, text in data], [int(label) - 1 for label, _ in data]
        self.data = [self.vocab(self.tokenizer(text)) for text in self.data]

        self.partition_validation_set(mode, val_frac)
    
    def get_num_classes(self):
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

    @staticmethod
    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(_label)
            processed_text = as_tensor(_text, dtype=long)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = as_tensor(label_list, dtype=long)
        offsets = as_tensor(offsets[:-1]).cumsum(dim=0)
        text_list = cat(text_list)
        return text_list, label_list, offsets
    
    def get_vocab_size(self):
        return self.vocab_size