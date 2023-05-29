import torch
from torch.utils.data import Dataset


class AAIdataset(Dataset):
    def __init__(self, x_ab, x_ag, y):
        self.x_ab = torch.tensor(x_ab)
        self.x_ag = torch.tensor(x_ag)
        self.y = torch.tensor(y)

    def __len__(self):
        return self.y.__len__()

    def __getitem__(self, index):
        return (
            self.x_ab[index].to(torch.float32),
            self.x_ag[index].to(torch.float32),
            self.y[index].to(torch.float32),
        )
