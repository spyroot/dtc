import numpy as np
from scipy.io.wavfile import read
import torch

def fmtl_print(left, *argv):
    """

    Args:
        left:
        *argv:

    Returns:

    """
    if len(argv) == 1:
        print(f"{str(left) + ':' :<32} {argv[0]}")
    else:
        print(f"{str(left) + ':' :<32} {argv}")


def fmt_print(left, *argv):
    """

    Args:
        left:
        *argv:

    Returns:

    """
    if len(argv) == 1:
        print(f"{str(left) + ':' :<25} {argv[0]}")
    else:
        print(f"{str(left) + ':' :<25} {argv}")


def get_mask_from_lengths(lengths):
    """

    Args:
        lengths:

    Returns:

    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    """

    Args:
        full_path:

    Returns:

    """
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    """

    Args:
        filename:
        split:

    Returns:

    """
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    """

    Args:
        x:

    Returns:

    """
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
