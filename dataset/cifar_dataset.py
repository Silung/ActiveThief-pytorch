import numpy as np
from dataset.base_dataset import BaseDataset
from keras.datasets import cifar10

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

class CifarNoiseDataset(CifarDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):
        super().__init__(normalize, mode, val_frac, normalize_channels, resize)
        self.state = 'unmark'
        
        # Initialize hashsets and hashmap
        self.marked = []
        self.marking = []
        self.unmark = list(range(self.data.shape[0]))
        
        self.aux_data = {}
        self._marked_counter = 0
        self._unmark_counter = 0
        
    def set_state(self, state):
        self.state = state
        
    def __getitem__(self, index):
        if self.state == 'marked':
            idx = self.marked[index]
        elif self.state == 'marking':
            idx = self.marking[index]
        else:
            idx = self.unmark[index]
        x = self.data[idx]
        y = self.labels[idx]
        
        if idx in self.aux_data:
            aux = self.aux_data[idx]
        else:
            aux = 0
        return x, y, idx, aux
        
    def __len__(self):
        if self.state == 'marked':
            return len(self.marked)
        elif self.state == 'marking':
            return len(self.marking)
        else:
            return len(self.unmark)
        
    def mark(self, i):
        assert i not in self.marked
        
        if i not in self.marked:
            self.unmark.remove(i)
            self.marking.append(i)
        
    def update(self, i, aux_data=None):
        self.marking.remove(i)
        self.marked.append(i)
        self.aux_data[i] = aux_data