import time
from typing import Optional

import librosa
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F
# from torch.nn.functional import melscale_fbanks
from model_loader.dataset_stft30 import SFTF3Dataset
from model_loader.stft_dataloader import SFTFDataloader
from model_trainer.trainer_specs import ExperimentSpecs
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from loguru import logger
from numpy import finfo
from torch import Tensor
from torch import nn
from torch import optim
import argparse
import logging
import os
import pickle
import random
import signal
import sys
from pathlib import Path
import socket
import torch.optim as optim


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
    r"""Solve for a normal STFT from a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    It minimizes the euclidian norm between the input mel-spectrogram and the product between
    the estimated spectrogram and the filter banks using SGD.

    Args:
        n_stft (int): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        max_iter (int, optional): Maximum number of optimization iterations. (Default: ``100000``)
        tolerance_loss (float, optional): Value of loss to stop optimization at. (Default: ``1e-5``)
        tolerance_change (float, optional): Difference in losses to stop optimization at. (Default: ``1e-8``)
        sgdargs (dict or None, optional): Arguments for the SGD optimizer. (Default: ``None``)
        norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    """
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

        fb = F.melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, norm, mel_scale)
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


class MelObjective(nn.Module):
    def __init__(self):
        super(MelObjective, self).__init__()
        self.flat = nn.Flatten()

    def forward(self, x, shape, A: Tensor, B: Tensor):
        """
            Compute the difference matrix
        :param x:
        :param shape:
        :param A:
        :param B:
        :return:
        """
        x = self.flat(x)
        diff = np.einsum("mf,...ft->...mt", A, x) - B
        value = (1 / B.size()) * 0.5 * torch.sum(diff ** 2)

        # And the gradient
        grad = (1 / B.size()) * torch.einsum("mf,...mt->...ft", A, diff)

        # Flatten the gradient
        return value, grad.flatten()

    @staticmethod
    def backward(ctx, grad_output):
        grad = (1 / B.size()) * torch.einsum("mf,...mt->...ft", A, diff)
        value, grad.flatten()
        return grad_input, grad.flatten()




from torch.autograd import Variable

class LibrosaInverseMelScale2(torch.nn.Module):
    r"""Solve for a normal STFT from a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    It minimizes the euclidian norm between the input mel-spectrogram and the product between
    the estimated spectrogram and the filter banks using SGD.

    Args:
        n_stft (int): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        max_iter (int, optional): Maximum number of optimization iterations. (Default: ``100000``)
        tolerance_loss (float, optional): Value of loss to stop optimization at. (Default: ``1e-5``)
        tolerance_change (float, optional): Difference in losses to stop optimization at. (Default: ``1e-8``)
        sgdargs (dict or None, optional): Arguments for the SGD optimizer. (Default: ``None``)
        norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    """
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
        super(LibrosaInverseMelScale2, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max or float(sample_rate // 2)
        self.f_min = f_min
        self.max_iter = max_iter
        self.tolerance_loss = tolerance_loss
        self.tolerance_change = tolerance_change
        self.sgdargs = sgdargs or {"lr": 0.1, "momentum": 0.9}
        self._block_size = 2 ** 8 * 2 ** 10
        assert f_min <= self.f_max, "Require f_min: {} < f_max: {}".format(f_min, self.f_max)

        fb = F.melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, norm, mel_scale)
        self.register_buffer("fb", fb)

    def nnls_obj(self, x, shape, A: Tensor, B: Tensor):
        """Compute the objective and gradient for NNLS"""

        # Scipy's lbfgs flattens all arrays, so we first reshape
        # the iterate x
        print(x.shape)
        print(A.shape)
        print(B.shape)

        x = x.reshape(shape)

        # Compute the difference matrix
        diff = torch.einsum("mf,...ft->...mt", A, x) - B

        # Compute the objective value
        value = (1 / B.storage().size()) * 0.5 * torch.sum(diff ** 2)

        # And the gradient
        # grad = (1 / B.storage().size()) * torch.einsum("mf,...mt->...ft", A, diff)
        # grad.requires_grad = True
        value.requires_grad = True
        #
        # print("Value", value.shape)
        # print("Grad", grad.shape)

        # Flatten the gradient
        return value
               #grad.flatten()

    def _nnls_lbfgs_block(self, A, B, x_init: Optional[Tensor] = None):
        """

        :param A:
        :param B: regression target
        :param x_init:
        :return:
        """
        if x_init is None:
            x_init = torch.einsum("fm,...mt->...ft", np.linalg.pinv(A), B)
            torch.clip(x_init, 0, None, out=x_init)

        # Adapt the hessian approximation to the dimension of the problem
        # kwargs.setdefault("m", A.shape[1])
        # Construct non-negative bounds
        shape = x_init.shape
        # shape.requires_grad = False

        # optimize
        # x, obj_value, diagnostics = scipy.optimize.fmin_l_bfgs_b(
        #         _nnls_obj, x_init, args=(shape, A, B), bounds=bounds, **kwargs
        # )
        print("before lstsq", A.shape)
        print("before lstsq", B.shape)

        mel_objective = MelObjective()
        optimizer = optim.LBFGS([x_init],
                                history_size=10,
                                max_iter=4,
                                line_search_fn="strong_wolfe")

        h_lbfgs = []
        for i in range(100):
            optimizer.zero_grad()
            objective = self.nnls_obj(x_init, shape, A, B)
#            loss = nn.functional.mse_loss(polished_model(trainable.input)[:, perm], A)
            objective.backward()
            optimizer.step(lambda: self.nnls_obj(x_init, shape, A, B))

            #loss_fn += F.cross_entropy(ops, tgts) * (len(subsmpl) / batch_size)

            h_lbfgs.append(objective.item())

        print("Done h_lbfgs", h_lbfgs)
        #
        # def closure():
        #     optimizer.zero_grad()
        #     loss = nn.functional.mse_loss(x_init, trainable.target_matrix)
        #     loss.backward()
        #     return loss

        # for i in range(100):
        #     optimizer.step(closure)

        # h_lbfgs = []
        # for i in range(100):
        #     optimizer.zero_grad()
        #     v, grad = mel_objective(x_init, A, B)
        #     objective.backward()
        #
        #     # def loss_closure():
        #     #     opt.zero_grad()
        #     #     oupt = log_reg(X)
        #     #     loss_val = loss_func(oupt, Y)
        #     #     loss_val.backward()
        #         return loss_val
        #
        #     optimizer.step(mel_objective(shape, A, B))
        #     h_lbfgs.append(objective.item())
        #
        # print("Solution", h_lbfgs)
        # # reshape the solution

        return h_lbfgs.reshape(shape)

    def nnls(self, A, B):

        if B.ndim == 1:
            print("Unsuported")
            sys.exit()
            # return scipy.optimize.nnls(A, B)[0]

        n_columns = self._block_size // (np.prod(B.shape[:-1]) * A.storage().element_size())
        n_columns = max(n_columns, 1)

        # # Process in blocks:
        if B.shape[-1] <= n_columns:
            print("Case one")
            return self._nnls_lbfgs_block(A, B).astype(A.dtype)

        A_inv = torch.linalg.pinv(A)
        B = torch.permute(B, (0, 2, 1))
        print("A inv shape", A_inv.shape)
        print("B inv shape", B.shape)

        x = torch.einsum("fm,...mt->...ft", A_inv, B)
        torch.clip(x, 0, None, out=x)
        x_init = x

        print("X einsum", x.shape)

        for bl_s in range(0, x.shape[-1], n_columns):
            bl_t = min(bl_s + n_columns, B.shape[-1])
            x[..., bl_s:bl_t] = self._nnls_lbfgs_block(A,
                                                       B[..., bl_s:bl_t],
                                                       x_init=x_init[..., bl_s:bl_t])
        return x

    def forward(self, melspec: Tensor) -> Tensor:
        r"""
        Args:
            melspec (Tensor): A Mel frequency spectrogram of dimension (..., ``n_mels``, time)

        Returns:
            Tensor: Linear scale spectrogram of size (..., freq, time)
        """
        # pack batch
        print(f"Original shape melspec: {melspec.shape}")
        shape = melspec.size()
        melspec = melspec.view(-1, shape[-2], shape[-1])

        n_mels, time = shape[-2], shape[-1]
        freq, _ = self.fb.size()  # (freq, n_mels)
        melspec = melspec.transpose(-1, -2)
        assert self.n_mels == n_mels

        # n_mels = M.shape[-2]
        #
        mel_basis = self.fb.T
        # Mel basis torch.Size([513, 80])
        print("Mel basis", mel_basis.shape)

        self.nnls(mel_basis, melspec)

        #
        # n_columns = self._block_size // (np.prod(B.shape[:-1]) * melspec.itemsize)
        # n_columns = max(n_columns, 1)
        #
        # # Process in blocks:
        # if B.shape[-1] <= n_columns:
        #     return self._nnls_lbfgs_block(melspec, B).astype(A.dtype)
        #
        # x = torch.einsum("fm,...mt->...ft", torch.linalg.pinv(melspec), B)
        #
        # print("Mel Shape", melspec.shape)

        # torch.linalg.lstsq(mel_padded)
        # torch.linalg.lstsq(A, B)

        # specgram = torch.rand(
        #         melspec.size()[0], time, freq, requires_grad=True, dtype=melspec.dtype, device=melspec.device
        # )
        #
        # optim = torch.optim.LST([specgram], **self.sgdargs)
        #
        # loss = float("inf")
        #
        # # optimize
        # optimizer = optim.LBFGS([x_lbfgs],
        #                         history_size=10,
        #                         max_iter=4,
        #                         line_search_fn="strong_wolfe")
        # h_lbfgs = []
        # for i in range(100):
        #     optimizer.zero_grad()
        #     objective = f(x_lbfgs)
        #     objective.backward()
        #     optimizer.step(lambda: f(x_lbfgs))
        #     h_lbfgs.append(objective.item())
        #
        # x, obj_value, diagnostics = scipy.optimize.fmin_l_bfgs_b(_nnls_obj,
        #                                                          x_init,
        #                                                          args=(shape, A, B),
        #                                                          bounds=bounds, **kwargs)
        #
        # for _ in range(self.max_iter):
        #     optim.zero_grad()
        #     diff = melspec - specgram.matmul(self.fb)
        #     new_loss = diff.pow(2).sum(axis=-1).mean()
        #     # take sum over mel-frequency then average over other dimensions
        #     # so that loss threshold is applied par unit timeframe
        #     new_loss.backward()
        #     optim.step()
        #     specgram.data = specgram.data.clamp(min=0)
        #
        #     new_loss = new_loss.item()
        #     if new_loss < self.tolerance_loss or abs(loss - new_loss) < self.tolerance_change:
        #         break
        #     loss = new_loss
        #
        # specgram.requires_grad_(False)
        # specgram = specgram.clamp(min=0).transpose(-1, -2)
        #
        # # unpack batch
        # specgram = specgram.view(shape[:-2] + (freq, time))
        # return specgram


def _get_ratio(mat):
    return (mat.sum() / mat.numel()).item()


def inverse_test(dataset_name=""):
    """

    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('spectrogram_layer')
    pk_dataset = trainer_spec.get_audio_dataset(dataset_name)
    assert 'train_set' in pk_dataset
    assert 'validation_set' in pk_dataset
    assert 'test_set' in pk_dataset

    dataloader = SFTFDataloader(trainer_spec, batch_size=2, verbose=True)
    loaders, collate = dataloader.get_loader()

    # get all
    data_loaders, collate_fn = dataloader.get_all()
    _train_loader = data_loaders['train_set']
    print(f"Train datasize {dataloader.get_train_dataset_size()}")

    n_fft = 1024
    n_stft = n_fft // 2 + 1

    inverse_mel = LibrosaInverseMelScale(n_stft, f_max=8000.0)
    for bidx, batch in enumerate(_train_loader):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, stft_padded = batch

        start_time = time.time()
        result = inverse_mel(mel_padded)

        print(f"MEL shape         {mel_padded.shape},  "
              f"output lengths    {output_lengths}  "
              f"stft target shape {stft_padded.shape}  "
              f"inverse shape     {result.shape}")
        print(f"Train datasize {dataloader.get_train_dataset_size()}")
        print("--- %s load time, seconds ---" % (time.time() - start_time))

        result_padded = torch.nn.functional.pad(result, (1, 1), "constant", 0)
        mel_padded = torch.nn.functional.pad(mel_padded, (1, 1), "constant", 0)
        result_mel_padded = inverse_mel(mel_padded)

        epsilon = 1e-60
        relative_diff = torch.abs((result_padded - stft_padded) / (result_padded + epsilon))
        relative_diff_padded = torch.abs((result_mel_padded - stft_padded) / (result_mel_padded + epsilon))

        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(f"Ratio of relative diff smaller than {tol:e} is " f"{_get_ratio(relative_diff < tol)}")

        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(f"Ratio of relative diff padded than {tol:e} is " f"{_get_ratio(relative_diff_padded < tol)}")

        # assert _get_ratio(relative_diff_padded < 1e-1) > 0.2
        # assert _get_ratio(relative_diff_padded < 1e-3) > 5e-3
        # assert _get_ratio(relative_diff_padded < 1e-5) > 1e-5

        S_inv_target = librosa.feature.inverse.mel_to_stft(mel_padded.detach().cpu().numpy(), n_fft=1024, sr=22050)

        S_inv_target = torch.from_numpy(S_inv_target)

        diff_librosa = torch.abs((S_inv_target - stft_padded.numpy()) / (S_inv_target + epsilon))
        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(f"Ratio of relative diff librosa than {tol:e} is " f"{_get_ratio(diff_librosa < tol)}")

        n_fft = 1024
        n_stft = n_fft // 2 + 1
        librosa_module = LibrosaInverseMelScale2(n_stft, f_max=8000.0)
        x = librosa_module(mel_padded)

        #
        # spec = inverse_mel(mel_padded)
        break

    # # example = train_dataset[1]
    # mse = torch.square(melspec - melspec_librosa).mean().item()
    # print("Mean Square Difference: ", mse)


if __name__ == '__main__':
    """
    """
    # test_download()
    # test_create_from_numpy_in_memory()
    # test_create_from_numpy_and_iterator()
    inverse_test('lj_speech_full_625')
