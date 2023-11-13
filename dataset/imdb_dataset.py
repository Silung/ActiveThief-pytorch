from torchtext.datasets import IMDB
from dataset.base_dataset import BaseTextDataset
from dataset.markable_dataset import MarkableDataset

class ImdbDataset(BaseTextDataset):
    def __init__(self, mode='train', val_frac=0.2):
        if mode == 'val':
            assert val_frac is not None
            
        self.dataset = IMDB
        super(ImdbDataset, self).__init__(mode=mode, val_frac=val_frac)
        
    def get_num_classes(self):
        return 2
    
class ImdbMarkableDataset(MarkableDataset, ImdbDataset):
    def __init__(self, mode='train', val_frac=0.2):
        ImdbDataset.__init__(self, mode, val_frac)
        MarkableDataset.__init__(self)
