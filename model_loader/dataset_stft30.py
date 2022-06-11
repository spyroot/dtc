# Short-time Fourier transform, and Mel dataset
#
# This dataset contains one hot vector, mel spectrogram
# and Short-time Fourier transform.
#
# Batch collate pack all and output N bathes.
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
                 root: Optional[str] = "dtc",
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
          Data formats
             tensor_mel  Use case, entire dataset passed, batchified via data loader and saved on disk.
             numpy_mel   Use case, entire dataset already in numpy format.
             numpy_file  Use case, entire dataset saved. i.e. each record one hot encoded text, mel,
                         other spectrum data.
             torch_file  Use case, entire dataset saved.
             audio_raw   Raw audio file that already annotated with text.

          :param model_spec:
          :param data: if data format is audio it must be a list that hold dict_keys(['path', 'meta', 'label'])
                       if data format torch, data must contain tensor is must dict must hold key data.
                       if data format numpy  and string it must point to dataset file
                       if data format torch_file and data string it path to a file.

          :param data_format: tensor_mel, numpy_mel, audio_raw
          :param is_trace_time:
          :param fixed_seed: if we want shuffle dataset
          :param shuffle: shuffle or not,  in case DDP you must not shuffle
          :param in_memory: store entire dataset in memory. (expect raw audio)
                 if in_memory is false, len can't return value, hence you need iterate manually.
          """
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
        It takes audio file and convert to mel spectrogram and Short-time Fourier transform
        Each audio file normalized.
        :param callback: if called pass callback transformed mel passed to callback.
        :param filename:
        :return:
        """
        normalized_audio, sampling_rate = self.normalize_audio(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))

        self.normalize_audio(filename)

        mel_spec, stft = self.stft.mel_spectrogram(normalized_audio)
        if callback is not None:
            callback(mel_spec, stft)

        mel_spec = torch.squeeze(mel_spec, 0)
        return mel_spec, stft

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param idx: index of dataset record.
        :return:
        """
        if self.is_a_tensor:
            text, mel, spectral_flatness = self._data[idx]
            return text, mel, spectral_flatness
        elif self.is_audio:
            if 'meta' not in self._data[idx]:
                raise Exception("Each data entry, must contain a meta key.")
            if 'path' not in self._data[idx]:
                raise Exception("Each data entry must contain path key.")

            text = self.text_to_tensor(self._data[idx]['meta'])
            mel, stft = self.audiofile_to_mel(self._data[idx]['path'])
            return text, mel, stft
        else:
            raise ValueError("Unknown dataset format")

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
