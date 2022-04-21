import torch
from torch.utils.data import Dataset


class Subset(Dataset):
    def __init__(self, dataset, limit):
        """
        :param dataset:
        :param limit: num of samples in subset
        """
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
