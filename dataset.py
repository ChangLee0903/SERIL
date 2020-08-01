import torch
from torch.utils.data import Dataset, DataLoader

class PseudoDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 2, 16000)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

