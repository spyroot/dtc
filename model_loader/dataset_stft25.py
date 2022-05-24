from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from loguru import logger

from model_loader.base_sfts_dataset import BaseSFTFDataset, DatasetError
from model_loader.tacotron_stft25 import TacotronSTFT25
from model_trainer.specs.tacatron_spec import TacotronSpec
from model_trainer.trainer_specs import ExperimentSpecs
from text import text_to_sequence


class SFTF2Dataset(BaseSFTFDataset):
    """

    """
    def __init__(self, model_spec: TacotronSpec, data, data_format,
                 fixed_seed=True, shuffle=False, is_trace_time=False) -> None:
        """
        :param model_spec:
        :param data:  data is dict must hold key data
        :param data_format: tensor_mel, numpy_mel, audio_raw
        :param is_trace_time:
        :param fixed_seed: if we want shuffle dataset
        :param shuffle: shuffle or not,  in case DDP you must not shuffle
        :param model_spec:
        :param data:
        :param data_format:
        :param fixed_seed:
        :param shuffle:
        :param is_trace_time:
        """
        super(SFTF2Dataset, self).__init__(model_spec=model_spec,
                                           data=data,
                                           data_format=data_format,
                                           fixed_seed=fixed_seed,
                                           shuffle=shuffle,
                                           is_trace_time=False)
        # if raw we need transcode to stft's
        if self.is_a_raw:
            logger.debug("Creating TacotronSTFT for raw file processing.")
            self.stft = TacotronSTFT25(
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

        _mel_spec = self.stft.mel_spectrogram(normalized_audio)
        if callback is not None:
            _mel_spec = callback(normalized_audio)

        _mel_spec = torch.squeeze(_mel_spec, 0)
        return _mel_spec

    def numpy_to_mel(self, filename) -> torch.Tensor:
        """
        Load numpy mel as dataset
        :param filename: numpy file
        :return:
        """
        logger.debug("Loading file {}", filename)
        mel_spec = torch.from_numpy(np.load(filename))
        if mel_spec.size(0) != self.stft.n_mel_channels:
            raise DatasetError("Mel shape is invalid, mismatch {} {}".
                               format(mel_spec.size(0), self.stft.n_mel_channels))
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
        if self.is_a_raw:
            if 'meta' not in self._data[index]:
                raise DatasetError("data must contain meta key")
            if 'path' not in self._data[index]:
                raise DatasetError("data must contain path key")
            text = self.text_to_tensor(self._data[index]['meta'])
            mel, spectral_flatness = self.file_to_mel(self._data[index]['path'])
            return text, mel

        return None, None


if __name__ == '__main__':
    """
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('encoder')
    pk_dataset = trainer_spec.get_audio_dataset()
        # self.train_dataset.get_audio_dataset()

    # ds = SFTF2Dataset(trainer_spec.)
