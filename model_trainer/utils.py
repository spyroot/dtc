import numpy as np
import torch
from scipy.io.wavfile import read as wav_reader


def fmtl_print(left, *argv):
    """

    :param left:
    :param argv:
    :return:
    """
    if len(argv) == 1:
        print(f"{str(left) + ':' :<32} {argv[0]}")
    else:
        print(f"{str(left) + ':' :<32} {argv}")


def fmt_print(left, *argv):
    """

    :param left:
    :param argv:
    :return:
    """
    if len(argv) == 1:
        print(f"{str(left) + ':' :<25} {argv[0]}")
    else:
        print(f"{str(left) + ':' :<25} {argv}")


def get_mask_from_lengths(lengths, device="cuda"):
    """
    :param lengths:
    :param device:
    :return:
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)).to(device)
    mask = (ids < lengths.unsqueeze(1)).bool().to(device)
    return mask


def load_wav_to_numpy(full_path, mmap=False):
    """
    Just proxy to backend if we need swap latter.
    :param full_path:
    :param mmap:
    :return:
    """
    return wav_reader(full_path, mmap)


def load_wav_to_torch(full_path: str, mmap=False) -> tuple[torch.FloatTensor, int]:
    """
    Read wav file to a tensor
    :param full_path:
    :param mmap: memory mapped or not
    :return: Tuple tensor, sample rate
    """
    sampling_rate, data = wav_reader(full_path, mmap)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    """
    Read txt file and split
    :param filename:
    :param split:
    :return:
    """
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x, device=None):
    """
    :param x:
    :param device:
    :return:
    """
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
