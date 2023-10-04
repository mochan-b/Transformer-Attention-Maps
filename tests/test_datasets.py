import torch
from functools import partial

from reverse_dataset import ReverseDataset
from leftright_product_dataset import LeftRightProductDataset


def test_reverse_dataset():
    """
    Test the ReverseDataset class making sure it returns he original data and the reversed data
    :return:
    """
    dataset = partial(ReverseDataset, 10, 16)
    train_dataset = dataset(100)
    data = train_dataset[0]
    assert data[0].equal(data[1].flip(dims=(0,)))
    assert data[0].tolist() == list(reversed(data[1].tolist()))


def test_left_right_product_dataset():
    """
    Test the LeftRightProductDataset class making sure it returns the original data and the product of the left and
    right halves of the data
    """
    dataset = partial(LeftRightProductDataset, 10, 16)
    train_dataset = dataset(100)
    data = train_dataset[0]
    expected_labels = torch.tensor([data[0][(i - 1) % 16] * data[0][(i + 1) % 16] % 10 for i in range(16)])
    assert data[1].equal(expected_labels)
