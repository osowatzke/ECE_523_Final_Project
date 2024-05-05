import torch
from torch.utils.data import Dataset

class RoiHeadsDataset(Dataset):
    def __init__(self, device=None):
        self.device = device
        self.data = []

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_sample = self.data[idx]
        feature_map = torch.tensor(data_sample[0], dtype=torch.float32)
        proposals = data_sample[1]
        image_sizes = data_sample[2]
        targets = data_sample[3]
        data_sample = (feature_map, proposals, image_sizes, targets)
        return data_sample

    def append(self,data_sample):
        feature_map = data_sample[0].detach().numpy()
        proposals = data_sample[1]
        image_sizes = data_sample[2]
        targets = data_sample[3]
        data_sample = (feature_map, proposals, image_sizes, targets)
        self.data.append(data_sample)