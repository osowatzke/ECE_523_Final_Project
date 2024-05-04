from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data=[]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def append(self,data_sample):
        self.data.append(data_sample)
        # print(len(self.data[-1]))
        # print(len(self.data))
        # pass