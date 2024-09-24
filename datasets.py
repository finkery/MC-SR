import h5py
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self,h5_file):
        super().__init__()
        self.h5_file = h5_file
    def __getitem__(self,idx):
        with h5py.File(self.h5_file,'r') as f:
            return f['lr'][idx], f['hr'][idx]
    def __len__(self):
        with h5py.File(self.h5_file,'r') as f:
            return len(f['lr'])