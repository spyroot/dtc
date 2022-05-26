from typing import Any

import numpy as np
import torch


def to_numpy(x: Any) -> np.ndarray:
    """

    :param x:
    :return:
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple, int, float)):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")
