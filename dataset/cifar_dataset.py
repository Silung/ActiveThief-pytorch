import numpy as np
from dataset.base_dataset import BaseDataset
from torchvision import datasets
from dataset.markable_dataset import MarkableDataset

class CifarDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None, transform=None):
        if mode == 'val':
            assert val_frac is not None
        
        super(CifarDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize,
            transform=transform
        )
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Load the required dataset
        if mode == 'train' or mode == 'val':
            dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
            self.data = dataset.data
            self.labels = dataset.targets
        else:
            assert mode == 'test'
            dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True)
            self.data = dataset.data
            self.labels = dataset.targets

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        # self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10

class CifarMarkableDataset(MarkableDataset, CifarDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):
        CifarDataset.__init__(self, normalize, mode, val_frac, normalize_channels, resize)
        MarkableDataset.__init__(self)
