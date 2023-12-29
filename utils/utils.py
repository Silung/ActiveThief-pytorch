import numpy as np

def f1_score(label, pred, num_class):
    epsilon = 1e-7
    f1_list = []
    for item in range(num_class):
        tp = ((pred == item) & (label == item)).sum()
        tn = ((pred != item) & (label != item)).sum()
        fp = ((pred == item) & (label != item)).sum()
        fn = ((pred != item) & (label == item)).sum()

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
    
        f1 = 2 * (precision*recall) / (precision + recall + epsilon)
        f1_list.append(f1)
    return np.array(f1_list).mean()
