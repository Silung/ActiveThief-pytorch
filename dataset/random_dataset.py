import torch
import numpy as np
from dataset.base_dataset import BaseDataset
from dataset.markable_dataset import MarkableDataset

class RandomDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):
        self.num_samples = 30000
        if mode == 'val':
            assert val_frac is not None
        
        super(RandomDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        # Load the required dataset
        if mode == 'train' or mode == 'val':
            self.data = np.random.randint(0, 256, (int(self.num_samples * 0.8), 64, 64, 3), dtype=np.uint8)
            self.labels = np.random.randint(0, self.get_num_classes(), (int(self.num_samples * 0.8)), dtype=np.uint8)
        else:
            assert mode == 'test'
            self.data = np.random.randint(0, 256, (int(self.num_samples * 0.2), 64, 64, 3), dtype=np.uint8)
            self.labels = np.random.randint(0, self.get_num_classes(), (int(self.num_samples * 0.2)), dtype=np.uint8)

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10

class OrderedDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):
        self.num_samples = 30000
        if mode == 'val':
            assert val_frac is not None
        
        super(OrderedDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        # Load the required dataset
        if mode == 'train' or mode == 'val':
            self.data = np.repeat(range(256), repeats=(int(self.num_samples * 0.8)*64*64*3)//256).astype(float).reshape(int(self.num_samples * 0.8), 64, 64, 3)
            self.labels = np.random.randint(0, self.get_num_classes(), (int(self.num_samples * 0.8)), dtype=np.uint8)
        else:
            assert mode == 'test'
            self.data = np.repeat(range(256), repeats=(int(self.num_samples * 0.2)*64*64*3)//256).astype(float).reshape(int(self.num_samples * 0.2), 64, 64, 3)
            self.labels = np.random.randint(0, self.get_num_classes(), (int(self.num_samples * 0.2)), dtype=np.uint8)

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10

class RandomMarkableDataset(MarkableDataset, RandomDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):
        RandomDataset.__init__(self, normalize, mode, val_frac, normalize_channels, resize)
        MarkableDataset.__init__(self)

class OrderedMarkableDataset(MarkableDataset, OrderedDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):
        OrderedDataset.__init__(self, normalize, mode, val_frac, normalize_channels, resize)
        MarkableDataset.__init__(self)
