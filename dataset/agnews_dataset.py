from torchtext.datasets import AG_NEWS
from dataset.base_dataset import BaseTextDataset
from dataset.markable_dataset import MarkableDataset

class AgnewsDataset(BaseTextDataset):
    def __init__(self, mode='train', val_frac=0.2):
        if mode == 'val':
            assert val_frac is not None
            
        self.dataset = AG_NEWS
        super(AgnewsDataset, self).__init__(mode=mode, val_frac=val_frac)
        
    def get_num_classes(self):
        return 4
    
class AgnewsMarkableDataset(MarkableDataset, AgnewsDataset):
    def __init__(self, mode='train', val_frac=0.2):
        AgnewsDataset.__init__(self, mode, val_frac)
        MarkableDataset.__init__(self)
