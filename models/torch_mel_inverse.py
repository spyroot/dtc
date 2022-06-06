#
# This original torch implementation and it SGD method doesn't
# produce very good result.   Initial attempt was improve this
# But I converged on LBFG implementation.
#
from typing import Optional
import torch
import torchaudio
from torch import Tensor
from torchaudio.functional import melscale_fbanks


class LibrosaMelScale(torch.nn.Module):
    r"""Turn a librosa STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    Args:
        n_mels (int, optional): Number of mel filter banks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)
        norm (str or None, optional): If ``'slaney'``, divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """
    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(LibrosaMelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        assert f_min <= self.f_max, "Require f_min: {} < f_max: {}".format(f_min, self.f_max)
        fb = torchaudio.functional.melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate,
                                                   self.norm, self.mel_scale)
        self.register_buffer("fb", fb)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # (..., time, freq) dot (freq, n_mels) -> (..., n_mels, time)
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

        return mel_specgram


class LibrosaInverseMelScale(torch.nn.Module):

    __constants__ = [
        "n_stft",
        "n_mels",
        "sample_rate",
        "f_min",
        "f_max",
        "max_iter",
        "tolerance_loss",
        "tolerance_change",
        "sgdargs",
    ]

    def __init__(
        self,
        n_stft: int,
        n_mels: int = 80,
        sample_rate: int = 22050,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        max_iter: int = 100000,
        tolerance_loss: float = 1e-5,
        tolerance_change: float = 1e-8,
        sgdargs: Optional[dict] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(LibrosaInverseMelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max or float(sample_rate // 2)
        self.f_min = f_min
        self.max_iter = max_iter
        self.tolerance_loss = tolerance_loss
        self.tolerance_change = tolerance_change
        self.sgdargs = sgdargs or {"lr": 0.1, "momentum": 0.9}

        assert f_min <= self.f_max, "Require f_min: {} < f_max: {}".format(f_min, self.f_max)

        fb = melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, norm, mel_scale)
        self.register_buffer("fb", fb)

    def forward(self, melspec: Tensor) -> Tensor:
        r"""
        Args:
            melspec (Tensor): A Mel frequency spectrogram of dimension (..., ``n_mels``, time)

        Returns:
            Tensor: Linear scale spectrogram of size (..., freq, time)
        """
        # pack batch
        shape = melspec.size()
        melspec = melspec.view(-1, shape[-2], shape[-1])

        n_mels, time = shape[-2], shape[-1]
        freq, _ = self.fb.size()  # (freq, n_mels)
        melspec = melspec.transpose(-1, -2)
        assert self.n_mels == n_mels

        specgram = torch.rand(melspec.size()[0], time, freq,
                              requires_grad=True, dtype=melspec.dtype, device=melspec.device)

        optim = torch.optim.SGD([specgram], **self.sgdargs)

        loss = float("inf")
        for _ in range(self.max_iter):
            optim.zero_grad()
            diff = melspec - specgram.matmul(self.fb)
            new_loss = diff.pow(2).sum(axis=-1).mean()
            # take sum over mel-frequency then average over other dimensions
            # so that loss threshold is applied par unit timeframe
            new_loss.backward()
            optim.step()
            specgram.data = specgram.data.clamp(min=0)

            new_loss = new_loss.item()
            if new_loss < self.tolerance_loss or abs(loss - new_loss) < self.tolerance_change:
                break
            loss = new_loss

        specgram.requires_grad_(False)
        specgram = specgram.clamp(min=0).transpose(-1, -2)

        # unpack batch
        specgram = specgram.view(shape[:-2] + (freq, time))
        return specgram
