# SFTacotronSTFT3,
#
# Create SFT from audio file.  Unlike original Tacotron
# This layer extracted MEL, STFT.  Note we currently
# pass directly Complex64.  At the moment torch supports
# L1 loss for complex64 but in practive we only use magnitude
# from original  STFT.
#
#
# Mustafa
import timeit
from pathlib import Path
from typing import Tuple

import librosa
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from loguru import logger
from torch import Tensor

from model_loader.audio_processing import dynamic_range_compression, dynamic_range_decompression
from model_loader.stft_module import STFT
from model_trainer.trainer_specs import ExperimentSpecs
from model_trainer.utils import load_wav_to_torch


class TacotronSTFT3(torch.nn.Module):
    """
    Create TacotronSTFT3 nn module
    """

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0, energy=2.0):
        """
        librosa.stft(n_fft=self.filter_length=1024, hop_length=256, win_length=1024)
        :param filter_length:  number of FFT components
        :param hop_length:
        :param win_length:
        :param sampling_rate:  sampling rate of the incoming signal
        :param n_mel_channels: number of Mel bands to generate
        :param mel_fmin: float >= 0 [scalar] lowest frequency (in Hz)
        :param mel_fmax: float >= 0 [scalar] highest frequency (in Hz). If `None`, use ``fmax = sr / 2.0``
        :param energy: float energy value 1.0 - 2.0
        """
        super(TacotronSTFT3, self).__init__()
        logger.info(f"n_fft {filter_length} hop length {hop_length} win_length "
                    f"{win_length} n_mel_channels {n_mel_channels} sample_rate "
                    f"{sampling_rate} mel_fmin {mel_fmin} mel_fmax {mel_fmax} energy {energy}")

        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.energy = energy
        self.mel_fmin = mel_fmin

        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sr=sampling_rate,
                                   n_fft=filter_length,
                                   n_mels=n_mel_channels,
                                   fmin=mel_fmin,
                                   fmax=mel_fmax)

        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis3', mel_basis)

    def mu_compression(self, input, mu: int, quantize=True):
        """
        sign(x) * ln(1 + mu * abs(x)) / ln(1 + mu)
        :param input:
        :param mu: The compression parameter. Values of the form 2**n - 1 (e.g., 15, 31, 63, etc.) are most common.
        :param quantize: True, quantize the compressed values into 1 + mu distinct integer values.
        :return:
        """
        y = librosa.mu_compress(input, quantize=True)

    def spectral_normalize(self, magnitudes):
        """
        Compression without quantization
        :param magnitudes:
        :return:
        """
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        """
        :param magnitudes:
        :return:
        """
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y, filter_length=None, stft=True) -> Tuple[Tensor, Tensor]:
        """
        Computes spectral flatness and mel spectrogram from a batch.

        :param stft: flag disable stft generation. note this mainly for a/b testing.
        :param filter_length:  n_fft
        :param y: tensor shape (batch, tensor), value normalized in range [-1, 1]
        :return:  torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        magnitudes, phases = self.stft_fn.transform(y)

        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis3, magnitudes)
        mel_output = self.spectral_normalize(mel_output)

        _filter_length = self.filter_length
        if filter_length is not None:
            _filter_length = filter_length

        y_numpy = y.squeeze(0).numpy()

        # sfts_y = librosa.stft(y_numpy,
        #                       n_fft=1024,
        #                       hop_length=self.hop_length,
        #                       win_length=self.win_length,
        #                       center=True)
        if stft:
            n = len(y_numpy)
            y_pad = librosa.util.fix_length(y_numpy, size=n + self.filter_length // 2)
            D = librosa.stft(y_pad, n_fft=self.filter_length,
                             hop_length=self.hop_length,
                             win_length=self.win_length)

            return mel_output, torch.from_numpy(D)
        else:
            return mel_output


def normalize_audio(filename):
    audio, sample_rate = load_wav_to_torch(filename)
    audio_norm = audio / 32768.0
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    return audio_norm


def test_audio_jist():
    SETUP_CODE = '''
from __main__ import normalize_audio
from random import randint
import librosa
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from model_loader.audio_processing import dynamic_range_compression, dynamic_range_decompression
from model_loader.stft import STFT
from torch import Tensor
from pathlib import Path
from tacotron2.utils import load_wav_to_torch'''

    TEST_CODE = '''
file_path = Path("~/Dropbox/Datasets/LJSpeech-1.1/wavs/LJ016-0283.wav")
normalize_audio(str(file_path.expanduser()))'''

    print(timeit.timeit(setup=SETUP_CODE,
                        stmt=TEST_CODE,
                        number=10))


if __name__ == '__main__':
    test_audio_jist()

    file_path = Path("~/Dropbox/Datasets/LJSpeech-1.1/wavs/LJ016-0283.wav")
    normalized = normalize_audio(str(file_path.expanduser()))

    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('encoder')

    stft = TacotronSTFT3(
            model_spec.filter_length(), model_spec.hop_length(), model_spec.win_length(),
            model_spec.n_mel_channels(), model_spec.sampling_rate(),
            model_spec.mel_fmin(),
            model_spec.mel_fmax())

    mel_spec, spectral_flatness = stft.mel_spectrogram(normalized)
    print(spectral_flatness.squeeze().shape)
