import torch
import numpy as np, os, struct
from dataset.base_dataset import BaseDataset
from os.path import expanduser, join
import h5py
from dataset.markable_dataset import MarkableDataset

class UspsDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None, transform=None):
        if mode == 'val':
            assert val_frac is not None

        if path is None:
            home = expanduser("~")
            self.path = os.path.join('data', 'USPS')
        else:
            self.path = path
        
        super(UspsDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize,
            transform=transform
        )
        
        self.aux_data = {}
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        path = os.path.join(self.path, 'usps.h5')
        with h5py.File(path, 'r') as hf:
            if mode == 'test':
                test = hf.get('test')
                self.data = test.get('data')[:]
                self.labels = test.get('target')[:]
            else:
                assert mode == 'train' or mode == 'val'
                train = hf.get('train')
                self.data = train.get('data')[:]
                self.labels = train.get('target')[:]

        # Perform splitting
        if mode != 'test':
            self.partition_validation_set(mode, val_frac)
        
        self.data = self.data.reshape(-1,16,16,1)
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10
    
    def update(self, i, aux_data=None):
        self.aux_data[i] = aux_data


class UspsMarkableDataset(MarkableDataset, UspsDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        UspsDataset.__init__(self, normalize, mode, val_frac, normalize_channels, path, resize)
        MarkableDataset.__init__(self)