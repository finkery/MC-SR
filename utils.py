import torch
from torch import nn

def calc_rmse(ground_truth,preds):  
    err = torch.sum((ground_truth - preds) ** 2)
    print(err)
    err /= (preds.shape[1] * preds.shape[2])
    return torch.sqrt(err)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count