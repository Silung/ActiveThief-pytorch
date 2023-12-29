import numpy as np, os
import random, cv2
# import json
import ujson as json
from dataset.base_dataset import BaseDataset
from os.path import expanduser, join
from dataset.markable_dataset import MarkableDataset

class ImagenetDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        if path is None:
            self.path = os.path.join('data', 'Imagenet64')
        else:
            self.path = path
        
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
        
        if normalize:
            # self.data = (self.data - float(np.min(self.data)))/(float(np.max(self.data)) - float(np.min(self.data)))
            self.data = (self.data - np.min(self.data))/(np.max(self.data) - np.min(self.data))
            
            assert np.abs(np.min(self.data) - 0.0) < 1e-1
            assert np.abs(np.max(self.data) - 1.0) < 1e-1

        print(f'Imagenet {mode} dataset size {len(self.labels)}')

    def load_data(self, mode, val_frac):
        # Our training and validation splits are of size 100K and 20K respectively. 
        xs = []
        ys = []

        if mode in ['train', 'val']:
            # data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(1, 6)]
            data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(1, 2)]
        # elif mode == 'val':
        #     # data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(9, 10+1)]
        #     data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(9,10)]
        else:
            data_files = [os.path.join(self.path, 'val_data.json')]

        for data_file in data_files:
            print('Loading', data_file)
        
            with open(data_file, 'rb') as data_file_handle:
                d = json.load(data_file_handle)
        
            # x = np.array(d['data'], dtype=np.uint8)
            x = np.array(d['data'], dtype=float)
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
        ImagenetDataset.__init__(self, normalize, mode, val_frac, normalize_channels, path, resize)
        MarkableDataset.__init__(self)