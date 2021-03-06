import logging

import numpy as np
import torch
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import functional as F

from model_trainer.specs.spectrogram_layer_spec import SpectrogramLayerSpec
from models.lbfs_mel_Inverse import DTCInverseSTFS


def tiny(x):
    return torch.finfo().tiny


def localmax_torch(x, *, axis=0, device=None):
    x_np = x.cpu().clone().detach().requires_grad_(False).numpy()

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)

    # x_pad = F.pad(x.numpy(), paddings, mode="edge")
    x_pad = np.pad(x_np, paddings, mode="edge")

    inds1 = [slice(None)] * x.ndim
    inds1[axis] = slice(0, -2)
    inds2 = [slice(None)] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    callulated = (x_np > x_pad[tuple(inds1)]) & (x_np >= x_pad[tuple(inds2)])
    return torch.tensor(callulated, device=device, requires_grad=False)


def expand_to(x, *, ndim, axes):
    try:
        axes = tuple(axes)
    except TypeError:
        axes = tuple([axes])

    shape = [1] * ndim
    for i, axi in enumerate(axes):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def pitch_track(
    *,
    y=None,
    sr=torch.tensor(22050),
    S=None,
    n_fft=torch.tensor(2048),
    hop_length=None,
    fmin=torch.tensor(150.0),
    fmax=torch.tensor(4000.0),
    threshold=torch.tensor(0.1),
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant",
    ref=None,
    _device=None,
):
    S = torch.abs(S)
    # Truncate to feasible region
    fmin = torch.maximum(fmin, torch.tensor(0))
    fmax = torch.minimum(fmax, sr / 2)

    d = torch.tensor(torch.tensor(1.0, device=_device, requires_grad=False) / sr)
    fft_freqs = torch.fft.rfftfreq(n=n_fft, d=d, device=_device, requires_grad=False)

    avg = torch.tensor(0.5, device=_device, requires_grad=False) * (S[..., 2:, :] - S[..., :-2, :])
    shift = torch.tensor(2, device=_device, requires_grad=False) * S[..., 1:-1, :] - S[..., 2:, :] - S[..., :-2, :]
    shift = avg / (shift + (torch.abs(shift) < tiny(shift)))

    avg = F.pad(avg, (0, 0, 1, 1), mode="constant")
    shift = F.pad(shift, (0, 0, 1, 1), mode="constant")

    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = torch.zeros_like(S)
    mags = torch.zeros_like(S)

    # Clip to the viable frequency range
    freq_mask = (fmin <= fft_freqs) & (fft_freqs < fmax)
    freq_mask = expand_to(freq_mask, ndim=S.ndim, axes=-2)
    ref_value = threshold * torch.amax(S, dim=-2, keepdim=True)

    # Store pitch and magnitude

    idx = torch.nonzero(freq_mask & localmax_torch(S * (S > ref_value), axis=-2, device=_device), as_tuple=True)
    pitches[idx] = (idx[-2] + shift[idx]) * sr / n_fft
    mags[idx] = S[idx] + dskew[idx]
    return pitches, mags


class dtcLoss(nn.Module):
    """

    """

    def __init__(self,
                 spec: SpectrogramLayerSpec,
                 filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 n_mel_channels=80,
                 sampling_rate=22050,
                 mel_fmin=0.0,
                 sr=22050,
                 n_fft=2048,
                 fmax=8000.0,
                 mel_fmax=8000.0, device=None):
        super(dtcLoss, self).__init__()

        self.filter_length = filter_length
        self.sample_rate = sampling_rate

        self.is_stft_compute = spec.is_stft_loss_enabled()
        self.is_reverse_encoder = spec.is_reverse_decoder()

        self.sr = torch.tensor(self.sample_rate, device=device, requires_grad=False)
        self.n_fft = torch.tensor(self.filter_length, device=device, requires_grad=False)
        self.f_min = torch.tensor(mel_fmin, device=device, requires_grad=False)
        self.f_max = torch.tensor(fmax, device=device, requires_grad=False)
        self.threshold = torch.tensor(0.1, device=device, requires_grad=False)
        self.device = device

        self.n_stft = 1024 // 2 + 1
        self.dts_inverse = DTCInverseSTFS(self.n_stft, f_max=fmax).to(device)

        if self.is_stft_compute:
            logging.info("inverse STFS loss enabled.")

        if self.is_reverse_encoder:
            logging.info("Reverse encoder - decoder enabled.")

    def kl_loss(self, q_dist):
        """
        KL loss used for VAE experiment.  it might be used for more experiments.
        In original case model out distribution.
        :param q_dist:
        :return:
        """
        return kl_divergence(q_dist, Normal(torch.zeros_like(q_dist.mean),
                                            torch.ones_like(q_dist.stddev))).sum(-1)

    def alignment_diagonal_error(self, alignments, binary=False):
        """
        Computes alignment prediction score. diagonal alignment
        of encoder.

        accept shape batch
        :param alignments:
        :param binary:
        :return:
        """
        with torch.no_grad():
            maxs = alignments.max(dim=1)[0]
            if binary:
                maxs[maxs > 0] = 1
        return maxs.mean(dim=1).mean(dim=0).item()

    def forward(self, model_output, targets, is_validation=False):
        """
        :param is_validation: The validation for the case if want need,
                              computes STFT during validation or training only.
        :param model_output:
        :param targets:
        :return:
        """
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False

        if not self.is_reverse_encoder:
            mel_out, mel_out_post_net, gate_out, alignment = model_output
            gate_targets = gate_target.view(-1, 1)
            gate_outs = gate_out.view(-1, 1)

            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
            alignment_score = self.alignment_diagonal_error(alignment)
            total = mel_loss + gate_loss

            if self.is_stft_compute:
                stft = targets[2]
                stft.requires_grad = False

                mel_padded = F.pad(mel_out, (1, 1), "constant", 0)
                mel_out_post_net = F.pad(mel_out_post_net, (1, 1), "constant", 0)

                with torch.no_grad():
                    x1 = self.dts_inverse(mel_padded)
                    x2 = self.dts_inverse(mel_out_post_net)

                abs_error = nn.L1Loss()(x1, stft) + nn.L1Loss()(x2, stft)
                total += abs_error

                return {
                    'loss': total,
                    'mel_loss': mel_loss,
                    'gate_loss': gate_loss,
                    'diagonal_score': alignment_score,
                    'abs_error': abs_error
                }
            return {
                'loss': total,
                'mel_loss': mel_loss,
                'gate_loss': gate_loss,
                'diagonal_score': alignment_score,
            }
        else:
            mel_out, mel_out_post_net, gate_out, alignment, rev = model_output
            gate_targets = gate_target.view(-1, 1)

            rev_mel_out, reverse_gate, alignment_rev = rev
            gate_outs = gate_out.view(-1, 1)

            # l1 loss for alignment
            l1_loss = nn.L1Loss()(alignment_rev, alignment)

            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
            alignment_score = self.alignment_diagonal_error(alignment)
            total = mel_loss + gate_loss + l1_loss

            if self.is_stft_compute:

                stft = targets[2]
                stft.requires_grad = False
                mel_padded = F.pad(mel_out, (1, 1), "constant", 0)
                mel_out_post_net = F.pad(mel_out_post_net, (1, 1), "constant", 0)

                with torch.no_grad():
                    x1 = self.dts_inverse(mel_padded)
                    x2 = self.dts_inverse(mel_out_post_net)

                abs_error = nn.L1Loss()(x1, stft) + nn.L1Loss()(x2, stft)
                total += abs_error

                return {
                    'loss': total,
                    'mel_loss': mel_loss,
                    'gate_loss': gate_loss,
                    'diagonal_score': alignment_score,
                    'abs_error': abs_error
                }
            return {
                'loss': total,
                'mel_loss': mel_loss,
                'gate_loss': gate_loss,
                'diagonal_score': alignment_score,
            }

