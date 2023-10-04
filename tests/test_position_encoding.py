import torch

from position_encoding import PositionalEncoding


def test_position_encoding():
    """
    Test the PositionalEncoding class. Very simple test, just make sure the shape is correct and it doesn't crash.
    :return:
    """
    pe = PositionalEncoding(20, 100)
    assert pe.pe.shape == (1, 100, 20)

    x = torch.ones(1, 100, 20)
    assert torch.allclose(pe(x), pe(x))
