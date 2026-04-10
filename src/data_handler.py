from torchvision import datasets
from torch.utils.data import Dataset
import torch

class CifarDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = datasets.ImageFolder(root=data_path, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, torch.tensor(label, dtype=torch.long)