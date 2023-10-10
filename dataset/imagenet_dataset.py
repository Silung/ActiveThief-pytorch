import numpy as np, os
import json
from dataset.base_dataset import BaseDataset
from os.path import expanduser, join

class ImagenetDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=None, normalize_channels=False, path=None, resize=None, num_to_keep=None, start_batch=1, end_batch=1):
        assert val_frac is None, 'This dataset has pre-specified splits.'

        self.start_batch = start_batch
        self.end_batch = end_batch
        if path is None:
            self.path = os.path.join('data', 'Imagenet64')
        else:
            self.path = path
        
        self.num_to_keep = num_to_keep
        
        super(ImagenetDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )

    def load_data(self, mode, val_frac):
        xs = []
        ys = []


        if mode == 'train':
            data_files = [os.path.join(self.path, 'train_data_batch_%d.json' % idx) for idx in range(self.start_batch, self.end_batch+1)]
        else:
            assert mode == 'val', 'Mode not supported.'
            data_files = [os.path.join(self.path, 'val_data.json')]

        for data_file in data_files:
            print('Loading', data_file)
        
            with open(data_file, 'rb') as data_file_handle:
                d = json.load(data_file_handle)
        
            x = np.array(d['data'], dtype=np.float32)
            y = np.array(d['labels'])

            # Labels are indexed from 1, shift it so that indexes start at 0
            y = [i-1 for i in y]

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

        if self.num_to_keep is not None:
            self.shuffle_data()
            self.data = self.data[:self.num_to_keep]
            self.labels = self.labels[:self.num_to_keep]

    def get_num_classes(self):
        return 1000
    
class ImagenetNoiseDataset(ImagenetDataset):
    def __init__(self, normalize=True, mode='train', val_frac=None, normalize_channels=False, path=None, resize=None, num_to_keep=None, start_batch=1, end_batch=1):
        super().__init__(normalize, mode, val_frac, normalize_channels, path, resize, num_to_keep, start_batch, end_batch)
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