import numpy as np, os, struct
from dataset.base_dataset import BaseDataset
from os.path import expanduser, join

class MnistDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        if mode == 'val':
            assert val_frac is not None

        if path is None:
            home = expanduser("~")
            self.path = os.path.join('data', 'mnist')
        else:
            self.path = path
        
        super(MnistDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        if mode == 'test':
            fname_img = os.path.join(self.path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(self.path, 't10k-labels-idx1-ubyte')
        else:
            assert mode == 'train' or mode == 'val'
            fname_img = os.path.join(self.path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(self.path, 'train-labels-idx1-ubyte')

        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.labels = np.fromfile(flbl, dtype=np.int8)
        
        with open(fname_img, 'rb') as fimg:
            print(fname_img)
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.data = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.labels), rows, cols, 1)

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10
    
class MnistSmallDataset(MnistDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        super().__init__(normalize, mode, val_frac, normalize_channels, path, resize)
        
        # Shrink Data
        reduced_size = int(len(self.data) * 0.1)
        self.data = self.data[:reduced_size]
        self.labels = self.labels[:reduced_size]
        
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