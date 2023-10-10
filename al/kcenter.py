import torch
import numpy as np
import math
from utils.kcenter import KCenter

class KCenterGreedyApproach():
    def __init__(self, size, Idx, Y_vec, init_cluster, batch_size):
        self.Y_vec = Y_vec
        self.Idx = Idx
        self.size = size
        self.init_cluster = init_cluster
        self.batch_size = batch_size
        
    def get_subset(self):
        X = self.Y_vec
        Y = self.init_cluster
        
        batch_size = 100*self.batch_size
        
        n_batches  = int(math.ceil(len(X)/float(batch_size)))
        
        m = KCenter()
        
        points = []
        
        # with tf.Session(config=config) as sess:
        #     sess.run(tf.global_variables_initializer())
        for _ in range(self.size):
            p = []
            q = []
            for i in range(n_batches):
                start = i*batch_size
                end   = i*batch_size + batch_size 
                X_b   = X[start:end]
                D_argmax_val, D_min_max_val = m(X_b, Y)

                p.append(D_argmax_val)
                q.append(D_min_max_val)

            q = torch.Tensor(q)
            b_indx = torch.argmax(q).item()
            indx = b_indx*batch_size + p[b_indx].item()
            Y = torch.concat([Y, X[indx].unsqueeze(0)], dim=0)
            points.append(self.Idx[indx])
           
        return points