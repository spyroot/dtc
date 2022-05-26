import random
from typing import Optional, Callable

import numpy as np
import torch
import torch.utils.data
from loguru import logger

from .base_sfts_dataset import BaseSFTFDataset
from model_trainer.specs.tacatron_spec import TacotronSpec
from model_trainer.utils import load_wav_to_torch
from text import text_to_sequence
from .tacotron_stft30 import TacotronSTFT3


class SFTF3Dataset(BaseSFTFDataset):
    """

    """
    def __init__(self, model_spec: TacotronSpec,
                 data=None,
                 root: Optional[str] = "dts",
                 data_format: Optional[str] = "numpy_mel",
                 fixed_seed: Optional[bool] = True,
                 shuffle: Optional[bool] = False,
                 is_trace_time: Optional[bool] = False,
                 in_memory: Optional[bool] = True,
                 download: Optional[bool] = False,
                 overfit: Optional[bool] = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 verbose: Optional[bool] = False) -> None:
        super(SFTF3Dataset, self).__init__(model_spec,
                                           root=root,
                                           data=data,
                                           download=download,
                                           data_format=data_format,
                                           fixed_seed=fixed_seed,
                                           shuffle=shuffle,
                                           is_trace_time=is_trace_time,
                                           in_memory=in_memory,
                                           verbose=verbose,
                                           overfit=overfit,
                                           transform=transform,
                                           target_transform=target_transform)
        self.set_logger(verbose)
        # if raw we need transcode to stft's
        if self.is_audio:
            logger.debug("Creating TacotronSTFT for raw file processing.")
            self.stft = TacotronSTFT3(
                    model_spec.filter_length(), model_spec.hop_length(), model_spec.win_length(),
                    model_spec.n_mel_channels(), model_spec.sampling_rate(), model_spec.mel_fmin(),
                    model_spec.mel_fmax())

    def file_to_mel(self, filename, callback=None):
        """

        :param callback:
        :param filename:
        :return:
        """

        # logger.debug("Converting file {} to mel", filename)
        normalized_audio, sampling_rate = self.normalize_audio(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
        self.normalize_audio(filename)

        mel_spec, spectral_flatness = self.stft.mel_spectrogram(normalized_audio)
        if callback is not None:
            callback(mel_spec, spectral_flatness)
        mel_spec = torch.squeeze(mel_spec, 0)

        return mel_spec, spectral_flatness

    def numpy_to_mel(self, filename):
        """
        :param filename:
        :return:
        """
        mel_spec = torch.from_numpy(np.load(filename))
        assert mel_spec.size(0) == self.stft.n_mel_channels, (
            'Mel dimension mismatch: given {}, expected {}'.format(
                    mel_spec.size(0), self.stft.n_mel_channels))

        return mel_spec

    def text_to_tensor(self, text):
        """
        One hot encoder for text seq
        :param text:
        :return:
        """
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        if self.is_a_tensor:
            text, mel, = self._data[index]
            return text, mel
        if self.is_audio:
            if 'meta' not in self._data[index]:
                raise Exception("data must contain meta key")
            if 'path' not in self._data[index]:
                raise Exception("data must contain path key")
            text = self.text_to_tensor(self._data[index]['meta'])
            mel, spectral_flatness = self.file_to_mel(self._data[index]['path'])
            return text, mel, spectral_flatness

        return None, None

    def __len__(self):
        return len(self._data)

    @staticmethod
    def set_logger(is_enable: bool) -> None:
        """
        Sets logging level.
        :param is_enable:
        :return:
        """
        if is_enable:
            logger.enable(__name__)
        else:
            logger.disable(__name__)


