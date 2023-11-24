import torch

import utilities.utils as utils


def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""

    return torch.sum((target - input_tensor) ** 2) / len(target)

