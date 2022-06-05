# STFT and Mel dataset
#
# It for dataset that outputs only one hot vector and mel spectrogram.
#
# Mustafa
#
from abc import ABC
from typing import Callable, Optional

import torch
import torch.utils.data
from loguru import logger
from torch import Tensor

from model_loader.base_stft_dataset import BaseSFTFDataset, DatasetError
from model_loader.tacotron_stft25 import TacotronSTFT25
from model_trainer.specs.spectrogram_layer_spec import SpectrogramLayerSpec
from model_trainer.trainer_specs import ExperimentSpecs


class SFTF2Dataset(BaseSFTFDataset, ABC):
    """

    """
    def __init__(self, model_spec: SpectrogramLayerSpec,
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
                 verbose: Optional[bool] = False,
                 overwrite: Optional[bool] = False) -> None:
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
        super(SFTF2Dataset, self).__init__(model_spec,
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
                                           overwrite=overwrite,
                                           transform=transform,
                                           target_transform=target_transform)
        # if raw we need transcode to stft's
        BaseSFTFDataset.set_logger(verbose)
        self.set_logger(verbose)

        if self.is_audio:
            logger.debug("Creating TacotronSTFT for raw file processing.")
            self.stft = TacotronSTFT25(
                    model_spec.filter_length(), model_spec.hop_length(), model_spec.win_length(),
                    model_spec.n_mel_channels(), model_spec.sampling_rate(), model_spec.mel_fmin(),
                    model_spec.mel_fmax())

    def audiofile_to_mel(self, filename: str, callback: callable = None) -> Tensor:
        """
        Take audio file and convert to mel spectrogram. Each audio file normalized.

        :param callback: if called pass callback transformed mel passed to callback.
        :param filename:
        :return:
        """
        # logger.debug("Converting file {} to mel", filename)
        normalized_audio, sampling_rate = self.normalize_audio(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("sample rate mismatch {} {}".format(sampling_rate, self.stft.sampling_rate))

        self.normalize_audio(filename)
        mel_spec = self.stft.mel_spectrogram(normalized_audio)
        if callback is not None:
            mel_spec = callback(normalized_audio)
        mel_spec = torch.squeeze(mel_spec, 0)

        return mel_spec

    def tensor_data(self, index) -> torch.Tensor:
        """

        :param index:
        :return:
        """
        return self._data[index]

    def tensor_from_audio(self, idx: int):
        """

        Returns text and mel
        :param idx: index into internal representation.
               if dataset was created form text file. It a dict.
        :return:
        """
        if 'meta' not in self._data[idx]:
            raise DatasetError("Tensor data must contain meta key in dictionary.")
        if 'path' not in self._data[idx]:
            raise DatasetError("Tensor data must contain meta key in dictionary.")

        text = self.text_to_tensor(self._data[idx]['meta'])
        mel = self.audiofile_to_mel(self._data[idx]['path'])

        return text, mel

    @staticmethod
    def set_logger(is_enable: bool) -> None:
        """
        Sets logging level.
        :param is_enable:
        :return:
        """
        BaseSFTFDataset.set_logger(is_enable)
        if is_enable:
            logger.enable(__name__)
        else:
            logger.disable(__name__)


def test_download():
    """
    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('spectrogram_layer')
    train_dataset = SFTF2Dataset(model_spec, download=True)


def test_save_and_load(dataset_name=""):
    """

    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('spectrogram_layer')
    pk_dataset = trainer_spec.get_audio_dataset(dataset_name)
    as_list = list(pk_dataset['train_set'].values())
    train_dataset = SFTF2Dataset(model_spec,
                                 list(pk_dataset['train_set'].values()),
                                 data_format='audio_raw', in_memory=False)

    # Save
    train_dataset.save_as_numpy("test.npy", overwrite=True)

    # # Load
    # train_dataset.load_from_numpy("test.npy", as_torch=False)
    # # Get iterator
    # ds = train_dataset.example_from_numpy("test.npy", as_torch=False)
    # assert isinstance(next(ds), tuple)


def test_create_from_numpy_in_memory():
    """
    Test in memory
    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('encoder')
    train_dataset = SFTF2Dataset(model_spec, 'dts/subset.npy', data_format='numpy_mel', in_memory=True, download=False)
    assert len(train_dataset) > 0
    assert isinstance(train_dataset[0], tuple)
    assert isinstance(train_dataset[0][0], Tensor)
    assert isinstance(train_dataset[0][1], Tensor)

    train_dataset2 = SFTF2Dataset(model_spec, 'dts/subset.npy', data_format='numpy_mel', in_memory=True, download=True)
    assert len(train_dataset) > 0
    assert isinstance(train_dataset[0], tuple)
    assert isinstance(train_dataset[0][0], Tensor)
    assert isinstance(train_dataset[0][1], Tensor)


def test_create_from_numpy_and_iterator():
    """
    Test none in memory, iterator
    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('spectrogram_layer')
    train_dataset = SFTF2Dataset(model_spec,
                                 'dts/subset.npy',
                                 data_format='numpy_mel',
                                 in_memory=False)
    assert len(train_dataset) > 0
    count = 0
    for d in train_dataset:
        assert isinstance(d, tuple)
        count += 1
    assert count == len(train_dataset)


if __name__ == '__main__':
    """
    """
    # test_download()
    # test_create_from_numpy_in_memory()
    # test_create_from_numpy_and_iterator()
    test_save_and_load('LJSpeechSmallRaw')

