import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self,h5_file):
        super().__init__()
        self.h5_file = h5_file
    def __getitem__(self,idx):
        with h5py.File(self.h5_file,'r') as f:
            return f['lr1'][idx], f['lr2'][idx], f['lr3'][idx], f['lr4'][idx], f['hr1'][idx], f['hr2'][idx], f['hr3'][idx], f['hr4'][idx] # 归一化处理还需要修改
    def __len__(self):
        with h5py.File(self.h5_file,'r') as f:
            return len(f['lr1'])

class EvalDataset(Dataset):
    def __init__(self,h5_file):
        super().__init__()
        self.h5_file = h5_file
    def __getitem__(self,idx):
        with h5py.File(self.h5_file,'r') as f:
            return f['lr1'][str(idx)][ : , : ], f['lr2'][str(idx)][ : , : ], f['lr3'][str(idx)][ : , : ], f['lr4'][str(idx)][ : , : ], f['hr1'][str(idx)][ : , : ], f['hr2'][str(idx)][ : , : ], f['hr3'][str(idx)][ : , : ], f['hr4'][str(idx)][ : , : ] # 归一化处理还需要修改
    def __len__(self):
        with h5py.File(self.h5_file,'r') as f:
            return len(f['lr1'])