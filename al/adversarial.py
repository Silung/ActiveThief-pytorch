import torch
import torch.nn as nn
import numpy as np
from attacks.deepfool import deepfool

class AdversarialSelectionStrategy:
    def __init__(self, size, Idx, Y_vec, X_vec, copy_model, K=10000, perm=None, epochs=20):
        self.Y_vec = Y_vec
        self.Idx = Idx
        self.size = size

        self.copy_model = copy_model
        self.X_vec = X_vec
        self.K = K
        self.perm = perm
        if self.K > len(self.X_vec):
            self.K = len(self.X_vec)

        print("number of epochs for adversarial strategy", epochs)

        self.copy_model.eval()  # Set the copy_model to evaluation mode

        self.xadv = deepfool(copy_model, epochs=epochs)  # Ideally batch=False and epochs>=10

        # difference as an $L_2$ norm
        self.diff = torch.sum(
            torch.sum(
                torch.sum(
                    torch.pow(self.xadv - copy_model.X, 2),
                    dim=1
                ),
                dim=1
            ),
            dim=1
        )

    def get_subset(self):
        diffs = np.ones(len(self.X_vec)) * float('inf')

        if self.perm is None:
            p = np.random.permutation(len(self.X_vec))
        else:
            p = self.perm

        rev_p = np.argsort(p)

        self.X_vec = self.X_vec[p]

        for start in range(0, self.K, args.batch_size):
            end = start + args.batch_size

            if end > len(self.X_vec):
                end = len(self.X_vec)

            diffs[start:end] = list(self.sess.run(self.diff, {self.copy_model.X: self.X_vec[start:end], self.copy_model.dropout_keep_prob: 1.0}))

        self.X_vec = self.X_vec[rev_p]
        diffs = diffs[rev_p]

        assert len(diffs) == len(self.X_vec)

        return np.argsort(diffs)[:self.size]
