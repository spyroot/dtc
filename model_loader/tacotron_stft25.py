import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

from model_loader.audio_processing import dynamic_range_compression, dynamic_range_decompression
from model_loader.stft import STFT


class TacotronSTFT(torch.nn.Module):
    def __init__(self,
                 filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 n_mel_channels=80,
                 sampling_rate=22050,
                 mel_fmin=0.0,
                 mel_fmax=8000.0):
        """
        :param filter_length:  number of FFT components
        :param hop_length:
        :param win_length:
        :param sampling_rate:  sampling rate of the incoming signal
        :param n_mel_channels: number of Mel bands to generate
        :param mel_fmin: float >= 0 [scalar] lowest frequency (in Hz)
        :param mel_fmax: float >= 0 [scalar] highest frequency (in Hz). If `None`, use ``fmax = sr / 2.0``
        """
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sr=sampling_rate,
                                   n_fft=filter_length,
                                   n_mels=n_mel_channels,
                                   fmin=mel_fmin,
                                   fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        """
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

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
