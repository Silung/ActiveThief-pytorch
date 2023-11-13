import numpy as np
from dataset.base_dataset import BaseDataset
from keras.datasets import cifar10
from dataset.markable_dataset import MarkableDataset

class CifarDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):

        if mode == 'val':
            assert val_frac is not None
        
        super(CifarDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Load the required dataset
        if mode == 'train' or mode == 'val':
            self.data = x_train
            self.labels = y_train
        else:
            assert mode == 'test'
            self.data = x_test
            self.labels = y_test

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10

class CifarMarkableDataset(MarkableDataset, CifarDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):
        CifarDataset.__init__(self, normalize, mode, val_frac, normalize_channels, resize)
        MarkableDataset.__init__(self)
