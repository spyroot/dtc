# SFTS, Mel dataset
#
# It for dataset that outputs only one hot vector and mel spectrogram.
#
# Mustafa
from abc import ABC
from typing import Callable, Optional

import torch
import torch.utils.data
from loguru import logger
from torch import Tensor

from model_loader.base_stft_dataset import BaseSFTFDataset
from model_loader.tacotron_stft30 import TacotronSTFT3
from model_trainer.specs.spectrogram_layer_spec import SpectrogramLayerSpec
from model_trainer.trainer_specs import ExperimentSpecs


class SFTF3Dataset(BaseSFTFDataset, ABC):
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
                                           overwrite=overwrite,
                                           transform=transform,
                                           target_transform=target_transform)
        BaseSFTFDataset.set_logger(verbose)
        self.set_logger(verbose)

        # if raw we need transcode to stft's
        if self.is_audio:
            logger.debug("Creating TacotronSTFT for raw file processing.")
            self.stft = TacotronSTFT3(
                    model_spec.filter_length(), model_spec.hop_length(), model_spec.win_length(),
                    model_spec.n_mel_channels(), model_spec.sampling_rate(), model_spec.mel_fmin(),
                    model_spec.mel_fmax())

    def audiofile_to_mel(self, filename: str, callback: callable = None) -> tuple[Tensor, Tensor]:
        """
        Take audio file and convert to mel spectrogram.
        Each audio file normalized.

        :param callback: if called pass callback transformed mel passed to callback.
        :param filename:
        :return:
        """
        normalized_audio, sampling_rate = self.normalize_audio(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))

        self.normalize_audio(filename)

        mel_spec, spectral_flatness = self.stft.mel_spectrogram(normalized_audio)
        if callback is not None:
            callback(mel_spec, spectral_flatness)

        mel_spec = torch.squeeze(mel_spec, 0)
        #  print(spectral_flatness.shape)
        return mel_spec, spectral_flatness

    # def numpy_to_mel(self, filename):
    #     """
    #     :param filename:
    #     :return:
    #     """
    #     mel_spec = torch.from_numpy(np.load(filename))
    #     assert mel_spec.size(0) == self.stft.n_mel_channels, (
    #         'Mel dimension mismatch: given {}, expected {}'.format(
    #                 mel_spec.size(0), self.stft.n_mel_channels))
    #
    #     return mel_spec, spectral_flatness

    # def text_to_tensor(self, text):
    #     """
    #     One hot encoder for text seq
    #     :param text:
    #     :return:
    #     """
    #     text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
    #     return text_norm

    def __getitem__(self, idx):
        """
        :param index:
        :return:
        """
        if self.is_a_tensor:
            text, mel, spectral_flatness = self._data[idx]
            return text, mel, spectral_flatness

        if self.is_audio:
            if 'meta' not in self._data[idx]:
                raise Exception("Each data entry, must contain a meta key.")
            if 'path' not in self._data[idx]:
                raise Exception("Each data entry must contain path key.")

            text = self.text_to_tensor(self._data[idx]['meta'])
            mel, spectral_flatness = self.audiofile_to_mel(self._data[idx]['path'])
            return text, mel, spectral_flatness

        return None, None, None

    def tensor_data(self, index) -> torch.Tensor:
        """
        :param index:
        :return:
        """
        return self._data[index]

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


def save_and_load_test(dataset_name=""):
    """

    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('spectrogram_layer')
    pk_dataset = trainer_spec.get_audio_dataset(dataset_name)
    assert 'train_set' in pk_dataset
    assert 'validation_set' in pk_dataset
    assert 'test_set' in pk_dataset
    # as_list = list(pk_dataset['train_set'].values())
    train_dataset = SFTF3Dataset(model_spec,
                                 list(pk_dataset['train_set'].values()),
                                 data_format='audio_raw', in_memory=False)
    example = train_dataset[0]
    # example = train_dataset[1]


if __name__ == '__main__':
    """
    """
    # test_download()
    # test_create_from_numpy_in_memory()
    # test_create_from_numpy_and_iterator()
    save_and_load_test('lj_speech_1k_raw')
