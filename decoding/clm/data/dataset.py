import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class WSJDataset(Dataset):

    def __init__(self, raw, batch_size, sequence_length):
        per_batch_length = int((len(raw) - 1) // batch_size)
        raw = np.asarray(raw[:per_batch_length * batch_size + 1], dtype='int64')
        data = np.reshape(raw[:-1], [batch_size, per_batch_length]).transpose()
        label = np.reshape(raw[1:], [batch_size, per_batch_length]).transpose()

        self.data = torch.from_numpy(data).cuda().contiguous()
        self.label = torch.from_numpy(label).cuda().contiguous()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.length = per_batch_length // sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        single_data = self.data[index * self.sequence_length:(index + 1) * self.sequence_length, :]
        single_label = self.label[index * self.sequence_length:(index + 1) * self.sequence_length, :]
        return single_data, single_label
