from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import torch
import random

class SampleDataset(Dataset):
    def __init__(self, data_len=10):
        self.data_len = 10

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        return idx
    
class CustomSampler(Sampler):
    def __init__(self, indices=None):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

sampler = CustomSampler()

dataset = SampleDataset()
data_loader = DataLoader(dataset=dataset,batch_size=4,shuffle=False,sampler=sampler)

for epoch in range(4):
    sampler.indices = np.random.permutation(4)
    print(sampler.indices)
    indices_used = []
    for idx in data_loader:
        indices_used.extend(list(idx))
    print(np.array(indices_used))
