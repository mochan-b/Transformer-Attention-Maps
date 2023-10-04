import torch
from torch.utils import data


class LeftRightProductDataset(data.Dataset):
    """
    Dataset that generates random sequences of integers and their reverse.
    """

    def __init__(self, num_categories, seq_len, size, shift=1):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size
        self.shift = shift

        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]

        # Shift inp_data by shift to the right and wrap the elements
        inp_data_left = torch.cat([inp_data[-self.shift:], inp_data[:-self.shift]])
        # Shift inp_data by shift to the left and wrap the elements
        inp_data_right = torch.cat([inp_data[self.shift:], inp_data[:self.shift]])
        # Labels is element multiplication of the left and right halves of the input data and modulo 10
        labels = (inp_data_left * inp_data_right) % 10

        # Labels are the product of the left and right halves of the input data
        # seq_len = self.seq_len
        # labels = torch.tensor([inp_data[(i - 1) % seq_len] * inp_data[(i + 1) % seq_len] % 10 for i in range(seq_len)])
        return inp_data, labels
