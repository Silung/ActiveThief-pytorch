import numpy as np
import os
import ujson as json
from PIL import Image
from dataset.base_dataset import BaseDataset
from dataset.markable_dataset import MarkableDataset

class ImagenetTinyDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None, transform=None):
        if mode == 'val':
            assert val_frac is not None
            
        if path is None:
            self.path = os.path.join('data', 'Imagenet-tiny')
        else:
            self.path = path
        
        super(ImagenetTinyDataset, self).__init__(
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
        if os.path.exists(os.path.join(self.path, 'data.npy')) and os.path.exists(os.path.join(self.path, 'labels.npy')):
            self.data = np.load(os.path.join(self.path, 'data.npy'))
            self.labels = np.load(os.path.join(self.path, 'labels.npy'))
        else:
            if os.path.exists(os.path.join(self.path, 'class2id.json')):
                with open(os.path.join(self.path, 'class2id.json'), 'r') as f:
                    class2id = json.load(f)
            else:
                class2id = {}
                with open(os.path.join(self.path, 'wnids.txt')) as f:
                    for c in f.read().strip().split('\n'):
                        if c not in class2id:
                            class2id[c] = len(class2id)
                with open(os.path.join(self.path, 'class2id.json'), 'w') as f:
                    json.dump(class2id, f)
                    
            # Load the required dataset
            self.data = []
            self.labels = []
            if mode == 'train' or mode == 'val':
                path = os.path.join(self.path, 'train')
                for class_name in os.listdir(path):
                    for root, dirs, files in os.walk(os.path.join(path, class_name, 'images')):
                        for name in files:
                            image = Image.open(os.path.join(root, name)).convert('RGB')
                            image = np.array(image)
                            self.data.append(image)
                            self.labels.append(class2id[class_name])
            else:
                assert mode == 'test'
                path = os.path.join(self.path, 'val')
                with open(os.path.join(path, 'val_annotations.txt')) as f:
                    for line in f:
                        name, class_name = line.strip().split('\n')
                        image = Image.open(os.path.join(path, 'images', 'name'))
                        image = np.array(image)
                        self.data.append(image)
                        self.labels.append(class2id[class_name])
                
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            
            shuffle_ix = np.random.permutation(np.arange(len(self.labels)))
            self.data = self.data[shuffle_ix]
            self.labels = self.labels[shuffle_ix]
            
            np.save(os.path.join(self.path, 'data.npy'), self.data)
            np.save(os.path.join(self.path, 'labels.npy'), self.labels)

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        # self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 200

class ImagenetTinyMarkableDataset(MarkableDataset, ImagenetTinyDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):
        ImagenetTinyDataset.__init__(self, normalize=normalize, mode=mode, val_frac=val_frac, normalize_channels=normalize_channels, path=path, resize=resize)
        MarkableDataset.__init__(self)
