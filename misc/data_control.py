import torch
from torch.utils.data.dataset import Dataset
import numpy as np

class CIFAR10Dataset(Dataset):
    def __init__(self, data, label, flip_prob = 0, transform = None):
        self.data = data
        self.label = label
        self.transform = transform
        self.flip_prob = flip_prob
        self.length = len(label)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        single_data = self.data[idx]
        single_label = self.label[idx]
        if self.transform is not None:
            single_data = self.transform(single_data)

        th = np.random.rand()
        if self.flip_prob > th:
            single_data = torch.from_numpy(np.flip(single_data.numpy(),1))


        return single_data, single_label