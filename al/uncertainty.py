from scipy.stats import entropy
import numpy as np

class UncertaintySelectionStrategy():
    def __init__(self, size, Idx, Y_vec):
        self.Y_vec = Y_vec
        self.Idx = Idx
        self.size = size
        
    def get_subset(self):
        entropies = np.array([entropy(yv) for yv in self.Y_vec])
        return np.array(self.Idx)[np.argsort(entropies*-1)[:self.size]]