import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    """
    Helper class to generate random data for training.

    Args:
        size (int): number of samples to generate

    Returns:
        data (list): list of tuples containing a random 20-dimensional tensor and a random 1-dimensional tensor
    """
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]