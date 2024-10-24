import torch.utils.data as data

class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return torch.FloatTensor(item), index
    
    def __len__(self):
        return len(self.data)