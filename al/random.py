import random

class RandomSelectionStrategy():
    def __init__(self, size, Idx, Y_vec):
        self.Y_vec = Y_vec
        self.Idx = Idx
        self.size = size
        
    def get_subset(self):
        s = random.sample(self.Idx, self.size)
        return s