import torch
import numpy as np, os, struct
from dataset.base_dataset import BaseDataset
from os.path import expanduser, join
from dataset.markable_dataset import MarkableDataset

class MnistDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None, transform=None):
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
            resize=resize,
            transform=transform
        )
        
        self.aux_data = {}
        
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
            self.data = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.labels), rows, cols)

        # Perform splitting
        if mode != 'test':
            self.partition_validation_set(mode, val_frac)
            
        self.data = np.stack((self.data, self.data, self.data), axis=-1)
        
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10
    
    def update(self, i, aux_data=None):
        self.aux_data[i] = aux_data
    
class MnistSmallDataset(MarkableDataset, MnistDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        MnistDataset.__init__(self, normalize, mode, val_frac, normalize_channels, path, resize)
        
        # Shrink Data
        if mode == 'train':
            reduced_size = int(len(self.data) * 0.1)
            self.data = self.data[:reduced_size]
            self.labels = self.labels[:reduced_size]
            
        MarkableDataset.__init__(self)
        
        
class MnistDistillationDataset(MarkableDataset, BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None, num_fig=10):
        self.num_fig = num_fig
        if mode == 'val':
            assert val_frac is not None

        BaseDataset.__init__(self,
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
        MarkableDataset.__init__(self)
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        dist_data = torch.load(f'data/mnist_dc/res_DC_MNIST_ConvNet_{self.num_fig}ipc.pt')['data'][0]

        # self.data = dist_data[0].permute(0,2,3,1)
        self.data = dist_data[0].permute(0,2,3,1)
        self.labels = dist_data[1]
                    
        shuffled_indices = torch.randperm(self.data.size(0))
        self.data = self.data[shuffled_indices].numpy()
        self.labels = self.labels[shuffled_indices].numpy()
        
        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
        
    def get_num_classes(self):
        return 10
        

class MnistMarkableDataset(MarkableDataset, MnistDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        MnistDataset.__init__(self, normalize=normalize, mode=mode, val_frac=val_frac, normalize_channels=normalize_channels, path=path, resize=resize)
        MarkableDataset.__init__(self)