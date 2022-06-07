#
#  It my implementation pleas emake sure you use lbfgs include.
#  It has 3 fixes , current torch implementation has 3 bugs
#
#  I'll do push request.
#
#  Mus
#
import sys
import time
from typing import Optional

import librosa
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.functional import melscale_fbanks

from model_loader.stft_dataloader import SFTFDataloader
from model_trainer.trainer_specs import ExperimentSpecs
from model_trainer.utils import to_gpu
from models.lbfgs_fixed import LBFGS_FIXED
from models.torch_mel_inverse import LibrosaInverseMelScale


class InverseSTFSObjective(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shape, A: Tensor, B: Tensor):
        """
        Compute the difference matrix
        :param ctx:
        :param x:
        :param shape:
        :param A:
        :param B:
        :return:
        """
        x = x.reshape(shape)
        diff = torch.einsum("mf,...ft->...mt", A, x) - B
        value = (1 / B.storage().size()) * 0.5 * torch.sum(diff ** 2)
        ctx.save_for_backward(A, B, diff)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        A, B, diff = ctx.saved_tensors
        grad = (1 / B.storage().size()) * torch.einsum("mf,...mt->...ft", A, diff)
        return (grad_output * grad), None, None, None


class DTCInverseSTFS(torch.nn.Module):
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
        device=torch.device("cuda:0"),
    ) -> None:
        """

        :param n_stft:  n_stft (int): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        :param n_mels:  (int, optional): Number of mel filterbanks. (Default: ``80``)
        :param sample_rate:    sample_rate (int, optional): Sample rate of audio signal. (Default: ``22050``)
        :param f_min:  f_min (float, optional): Minimum frequency. (Default: ``0.``)
        :param f_max:  f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        :param max_iter:  max_iter (int, optional): Maximum number of optimization iterations.
                          (Default: ``100000``)
        :param tolerance_loss:   tolerance_loss (float, optional): Value of loss to stop optimization at.
                                (Default: ``1e-5``)
        :param tolerance_change:   tolerance_change (float, optional): Difference in losses to stop optimization at.
                                   (Default: ``1e-8``)
        :param sgdargs:  sgdargs (dict or None, optional): Arguments for the SGD optimizer. (Default: ``None``)
        :param norm:   norm (str or None, optional): If 'slaney', divide the triangular mel
                    weights by the width of the mel band (area normalization). (Default: ``None``)
        :param mel_scale:
        :param device:
        """
        super(DTCInverseSTFS, self).__init__()
        # num mels
        self.n_mels = n_mels
        # sample rate
        self.sample_rate = sample_rate
        # f_max
        self.f_max = f_max or float(sample_rate // 2)
        # f_min
        self.f_min = f_min
        #
        self.max_iter = max_iter
        #
        self.tolerance_loss = tolerance_loss
        #
        self.tolerance_change = tolerance_change
        #
        self.sgdargs = sgdargs or {"lr": 0.1, "momentum": 0.9}
        #
        self._block_size = 2 ** 8 * 2 ** 10
        #
        assert f_min <= self.f_max, "Require f_min: {} < f_max: {}".format(f_min, self.f_max)

        fb = melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, norm, mel_scale).to(device)
        #
        self.device = device
        #
        self.register_buffer("fb", fb)

    @staticmethod
    def lbfgs_block(A: Tensor, B: Tensor, x_init: Optional[Tensor] = None, attach_grad=False) -> Tensor:
        """
        :param attach_grad:
        :param A:
        :param B: regression target
        :param x_init:
        :return:
        """
        if x_init is None:
            B = torch.permute(B, (0, 2, 1))
            x_init = torch.einsum("fm,...mt->...ft", torch.linalg.pinv(A), B)
            torch.clip(x_init, 0, None, out=x_init)

        if attach_grad:
            x_init.requires_grad = True

        shape = x_init.shape
        optimizer = LBFGS_FIXED([x_init],
                                lr=0.1,
                                max_iter=500,
                                tolerance_grad=1e-3,
                                history_size=10,
                                line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = InverseSTFSObjective.apply(x_init, shape, A, B)
            loss.backward()
            return loss

        optimizer.step(closure)
        return x_init.reshape(shape)

    def nnls(self, A, B):
        """
        :param A:
        :param B:
        :return:
        """
        if B.ndim == 1:
            x = torch.linalg.lstsq(A, B).solution
            print("x for b.ndim == 1 {}", x.shape)
            return torch.linalg.lstsq(A, B).solution
            sys.exit()
            # return scipy.optimize.nnls(A, B)[0]

        n_columns = self._block_size // (np.prod(B.shape[:-1]) * A.storage().element_size())
        n_columns = max(n_columns, 1)

        # Process in blocks:
        if B.shape[-1] <= n_columns:
            return self.lbfgs_block(A, B, attach_grad=True)

        A_inv = torch.linalg.pinv(A)
        B = torch.permute(B, (0, 2, 1))

        x = torch.einsum("fm,...mt->...ft", A_inv, B)
        torch.clip(x, min=0, max=None, out=x)
        x_init = x

        for bl_s in range(0, x.shape[-1], n_columns):
            bl_t = min(bl_s + n_columns, B.shape[-1])
            B_block = B[..., bl_s:bl_t]
            X_block = x_init[..., bl_s:bl_t]
            X_block.requires_grad = True
            x[..., bl_s:bl_t] = self.lbfgs_block(A, B_block, x_init=X_block).detach()
            X_block.requires_grad = False

        return x

    def forward(self, melspec: Tensor, power=2.0) -> Tensor:
        """
         A Mel frequency spectrogram of dimension (..., ``n_mels``, time)

        :param power:
        :param melspec:
        :return: A tensor
        """
        shape = melspec.size()
        melspec = melspec.view(-1, shape[-2], shape[-1])

        n_mels, time_domain = shape[-2], shape[-1]
        freq, _ = self.fb.size()  # (freq, n_mels)
        melspec = melspec.transpose(-1, -2)
        assert self.n_mels == n_mels

        # n_mels = M.shape[-2]

        mel_basis = self.fb.T

        inverse = self.nnls(mel_basis, melspec)
        # return torch.pow(inverse, 1.0 / power, out=inverse)
        return inverse


def _get_ratio(mat):
    return (mat.sum() / mat.numel()).item()


def compute_abs_delta(computed, target,  epsilon: Optional[float] = 1e-60):
    return torch.abs((computed - target) / (computed + epsilon))


def inverse_correctness(mel, stft, n_fft=1024, epsilon: Optional[float] = 1e-60, f_max=8000.0, sr=22050, strict=True):
    """

    :param sr:
    :param strict:
    :param mel:
    :param stft:
    :param n_fft:
    :param epsilon:
    :param f_max:
    :return:
    """

    _device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    n_stft = n_fft // 2 + 1
    torch_inverse_mel = LibrosaInverseMelScale(n_stft, f_max=f_max)
    torch_mel_inverse = torch_inverse_mel(mel)
    torch_mel_inverse = F.pad(torch_mel_inverse, (1, 1), "constant", 0)

    mel_padded = F.pad(mel, (1, 1), "constant", 0)
    torch_inverse_mel_padded = torch_inverse_mel(mel_padded)

    # diff between torch mel and original STFT
    torch_relative_diff = compute_abs_delta(torch_mel_inverse, target=stft, epsilon=epsilon)
    for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
        print(f"Ratio of relative diff smaller than {tol:e} is " f"{_get_ratio(torch_relative_diff < tol)}")
    print("-------------------------------------------------------------------------------------------------")

    relative_diff_padded = compute_abs_delta(torch_inverse_mel_padded, target=stft, epsilon=epsilon)
    for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
        print(f"Ratio of relative diff smaller than {tol:e} is " f"{_get_ratio(relative_diff_padded < tol)}")
    print("-------------------------------------------------------------------------------------------------")

    # librosa version numpy
    librosa_inverse = librosa.feature.inverse.mel_to_stft(mel_padded.detach().cpu().numpy(), n_fft=n_fft, sr=sr)
    librosa_inverse = torch.from_numpy(librosa_inverse)
    diff_librosa = compute_abs_delta(librosa_inverse, target=stft.numpy())
    for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
        print(f"Ratio of relative diff librosa than {tol:e} is " f"{_get_ratio(diff_librosa < tol)}")
    print("-------------------------------------------------------------------------------------------------")

    # mine dtc
    mel_padded = to_gpu(mel_padded).float()
    stft = to_gpu(stft)
    librosa_module = DTCInverseSTFS(1024 // 2 + 1, f_max=8000.0, device=_device)
    x = librosa_module(mel_padded)
    assert x.shape == stft.shape == librosa_inverse.shape

    inverse_abs_error = compute_abs_delta(x, target=stft, epsilon=epsilon)
    for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
        print(f"Ratio of relative diff my impl {tol:e} is " f"{_get_ratio(inverse_abs_error < tol)}")

    # if strict:
    #     assert math.isclose(_get_ratio(inverse_abs_error < 1e-1), 0.2)
    #     assert _get_ratio(inverse_abs_error < 1e-1) > 0.2
    #     assert _get_ratio(inverse_abs_error < 1e-3) > 5e-3
    #     assert _get_ratio(inverse_abs_error < 1e-5) > 1e-5


def inverse_test_correctness(dataset_name=""):
    """
    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    trainer_spec.set_active_dataset(dataset_name=dataset_name)
    pk_dataset = trainer_spec.get_audio_dataset(dataset_name)
    assert 'train_set' in pk_dataset
    assert 'validation_set' in pk_dataset
    assert 'test_set' in pk_dataset

    dataloader = SFTFDataloader(trainer_spec, batch_size=2, verbose=True)
    data_loaders, collate_fn = dataloader.get_all()
    _train_loader = data_loaders['train_set']
    print(f"Train datasize {dataloader.get_train_dataset_size()}")

    for bidx, batch in enumerate(_train_loader):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, stft_padded = batch
        inverse_correctness(mel_padded, stft_padded)
        break

    # # example = train_dataset[1]
    # mse = torch.square(melspec - melspec_librosa).mean().item()
    # print("Mean Square Difference: ", mse)


def compute_epsilon_deltas(computed_inverse):
    for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
        print(f"Ratio of relative diff my impl {tol:e} is " f"{_get_ratio(computed_inverse < tol)}")


def inverse_test_combine_error(dataset_name="", epsilon=1e-60, max_iteration=100, verbose=False):
    """
    :return:
    """

    _device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    trainer_spec.set_active_dataset(dataset_name)
    pk_dataset = trainer_spec.get_audio_dataset(dataset_name)

    assert 'train_set' in pk_dataset
    assert 'validation_set' in pk_dataset
    assert 'test_set' in pk_dataset

    dataloader = SFTFDataloader(trainer_spec, batch_size=1, verbose=True)
    loaders, collate = dataloader.get_loader()

    # get all
    data_loaders, collate_fn = dataloader.get_all()
    _train_loader = data_loaders['train_set']
    print(f"Train datasize {dataloader.get_train_dataset_size()}")

    n_fft = 1024
    n_stft = n_fft // 2 + 1

    inverse_mel = LibrosaInverseMelScale(n_stft, f_max=8000.0)
    librosa_module = DTCInverseSTFS(1024 // 2 + 1, f_max=8000.0, device=torch.device(_device)).to(_device)

    torch_abs_error = 0
    mine_abs_error = 0
    librosa_abs_error = 0
    torch_abs_error_padded = 0
    for bidx, batch in enumerate(_train_loader):
        if bidx == max_iteration:
            break

        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, stft_padded = batch

        # start_time = time.time()
        torch_mel_inverse = inverse_mel(mel_padded)
        torch_mel_inverse = F.pad(torch_mel_inverse, (1, 1), "constant", 0)
        mel_padded = F.pad(mel_padded, (1, 1), "constant", 0)
        torch_inverse_mel_padded = inverse_mel(mel_padded)

        # diff between torch mel and original STFT
        torch_relative_diff = torch.abs((torch_mel_inverse - stft_padded) / (torch_mel_inverse + epsilon))
        relative_diff_padded = torch.abs(
                (torch_inverse_mel_padded - stft_padded) / (torch_inverse_mel_padded + epsilon))
        if verbose:
            compute_epsilon_deltas(torch_relative_diff)
            compute_epsilon_deltas(relative_diff_padded)

        torch_abs_error += nn.L1Loss()(torch_mel_inverse, stft_padded).item()
        torch_abs_error_padded += nn.L1Loss()(torch_inverse_mel_padded, stft_padded).item()

        # librosa version numpy
        librosa_inverse = torch.from_numpy(
                librosa.feature.inverse.mel_to_stft(
                        mel_padded.detach().cpu().numpy(), n_fft=1024, sr=22050))
        diff_librosa = torch.abs((librosa_inverse - stft_padded.numpy()) / (librosa_inverse + epsilon))
        if verbose:
            compute_epsilon_deltas(diff_librosa)
        librosa_abs_error += nn.L1Loss()(librosa_inverse, stft_padded).item()

        # mine
        mel_padded = to_gpu(mel_padded, _device).float()
        x = librosa_module(mel_padded)
        assert x.shape == stft_padded.shape == librosa_inverse.shape
        if verbose:
            compute_epsilon_deltas(x)

        stft_padded = to_gpu(stft_padded)
        mine_abs_error += nn.L1Loss()(x, stft_padded).item()

    print(f"torch {torch_abs_error}, torch padded {torch_abs_error_padded}, "
          f"librosa {librosa_abs_error}, dtc {mine_abs_error}")


def inverse_test_gpu(dataset_name="", epsilon=1e-60, max_iteration=100, batch_size=32, verbose=False):
    """
    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    pk_dataset = trainer_spec.get_audio_dataset(dataset_name)
    assert 'train_set' in pk_dataset
    assert 'validation_set' in pk_dataset
    assert 'test_set' in pk_dataset

    dataloader = SFTFDataloader(trainer_spec, batch_size=2, verbose=True)
    data_loaders, collate_fn = dataloader.get_all()
    _train_loader = data_loaders['train_set']

    n_stft = 1024 // 2 + 1

    abs_error = 0
    start_time = time.time()
    dtc_stfs_module = DTCInverseSTFS(n_stft, f_max=8000.0).to(torch.device('cuda'))

    for bidx, batch in enumerate(_train_loader):
        if bidx == max_iteration:
            break
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, stft_padded = batch

        # mine
        mel_padded = mel_padded.contiguous().cuda(non_blocking=True)
        stft_padded = stft_padded.contiguous().cuda(non_blocking=True)
        mel_padded = F.pad(mel_padded, (1, 1), "constant", 0)
        x = dtc_stfs_module(mel_padded)
        abs_error += nn.L1Loss()(x, stft_padded).item()

    print("--- %s load time, seconds ---" % (time.time() - start_time))
    print(f"torch {abs_error}")


if __name__ == '__main__':
    """
    """
    # test_download()
    # test_create_from_numpy_in_memory()
    # test_create_from_numpy_and_iterator()
    inverse_test_correctness('lj_speech_1k_raw')
# inverse_test_combine_error('lj_speech_1k_raw')
# inverse_test_gpu('lj_speech_1k_raw')
