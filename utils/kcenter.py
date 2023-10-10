import torch
import torch.nn as nn
import numpy as np

class KCenter(nn.Module):
    def __init__(self):
        super(KCenter, self).__init__()
        self.D_min_max = None
        self.D_min_argmax = None

    def forward(self, A, B):
        D = self.pairwise_distances(A, B)

        D_min = torch.min(D, dim=1)[0]
        self.D_min_max = torch.max(D_min)
        self.D_min_argmax = torch.argmax(D_min)
        return self.D_min_argmax, self.D_min_max
        
    def pairwise_distances(self, A, B):
        na = torch.sum(A ** 2, dim=1)
        nb = torch.sum(B ** 2, dim=1)

        na = na.reshape(-1, 1)
        nb = nb.reshape(1, -1)

        D = torch.sqrt(torch.maximum(na - 2 * torch.matmul(A, B.t()) + nb, torch.tensor(0.0)))
        return D