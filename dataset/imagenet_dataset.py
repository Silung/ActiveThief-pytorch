import numpy as np, os
import random, cv2
# import json
import ujson as json
from dataset.base_dataset import BaseDataset
from os.path import expanduser, join
from dataset.markable_dataset import MarkableDataset

class ImagenetDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None, transform=None, num_train_batch=1):
        if mode == 'val':
            assert val_frac is not None
        
        self.num_train_batch = num_train_batch
        
        if path is None:
            self.path = os.path.join('data', 'Imagenet64')
        else:
            self.path = path
        
        super(ImagenetDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize,
            transform=transform
        )

    def load_data(self, mode, val_frac):
        # Our training and validation splits are of size 100K and 20K respectively. 
        xs = []
        ys = []

        # if mode in ['train', 'val']:
            # data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(1, 6)]
            # data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(1, 2)]
        # elif mode == 'val':
        #     # data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(9, 10+1)]
        #     data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(9,10)]
        # else:
        #     data_files = [os.path.join(self.path, 'val_data.json')]
            
        if mode == 'train':
            data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(1, self.num_train_batch+1)]
        else:
            # assert mode == 'val', 'Mode not supported.'
            data_files = [os.path.join(self.path, 'val_data.json')]

        for data_file in data_files:
            print('Loading', data_file)
        
            with open(data_file, 'rb') as data_file_handle:
                d = json.load(data_file_handle)
        
            x = np.array(d['data'], dtype=np.uint8)
            # x = np.array(d['data'], dtype=float)
            y = np.array(d['labels'])

            # Labels are indexed from 1, shift it so that indexes start at 0
            y = [i-1 for i in y]
            # print(f'{mode}:max_y={max(y)};min_y={min(y)}')

            img_size  = 64
            img_size2 = img_size * img_size

            x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
            x = x.reshape((x.shape[0], img_size, img_size, 3))

            xs.append(x)
            ys.append(np.array(y))

        if len(xs) == 1:
            self.data   = xs[0]
            self.labels = ys[0]
        else:
            self.data   = np.concatenate(xs, axis=0)
            self.labels = np.concatenate(ys, axis=0)

        # Perform splitting
        if mode != 'test':
            self.partition_validation_set(mode, val_frac)

    def get_num_classes(self):
        return 1000
    
class ImagenetMarkableDataset(MarkableDataset, ImagenetDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        ImagenetDataset.__init__(self, normalize=normalize, mode=mode, val_frac=val_frac, normalize_channels=normalize_channels, path=path, resize=resize)
        MarkableDataset.__init__(self)