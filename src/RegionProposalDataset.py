import torch
from torch.utils.data import Dataset

class RegionProposalDataset(Dataset):
    def __init__(self, use_built_in_rpn=False, device=None):
        self.use_built_in_rpn = use_built_in_rpn
        self.device = device
        self.data = []

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_sample = self.data[idx]
        if self.use_built_in_rpn:
            image = data_sample[0]
            feature_map = data_sample[1]
            targets = data_sample[2]
            if self.device is not None:
                image = image.to(self.device)
                feature_map = feature_map.to(self.device)
            data_sample = (image, feature_map, targets)
        else:
            feature_map = data_sample[0]
            targets = data_sample[1]
            if self.device is not None:
                feature_map = feature_map.to(self.device)
            data_sample = (feature_map, targets)
        return data_sample

    def append(self,data_sample):
        if self.use_built_in_rpn:
            image = data_sample[0].detach().cpu() #.detach().numpy()
            feature_map = data_sample[1].detach().cpu() #detach().numpy()
            targets = data_sample[2]
            data_sample = (image, feature_map, targets)
        else:
            feature_map = data_sample[0].detach().cpu() #detach().numpy()
            targets = data_sample[1]
            data_sample = (feature_map, targets)
        self.data.append(data_sample)