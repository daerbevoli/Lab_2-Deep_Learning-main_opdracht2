import torch

import utilities.utils as utils


def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""
    # TENSORS
    # The mean should be all the elements in the amount of dimensions that you have
    return torch.mean((target - input_tensor) ** 2)

