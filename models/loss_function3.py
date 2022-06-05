import librosa
import numpy as np
import torch
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torchaudio.transforms import InverseMelScale


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


class DTSLoss(nn.Module):
    """

    """

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, sr=22050, n_fft=2048, fmax=8000,
                 mel_fmax=8000.0, device=None):
        super(DTSLoss, self).__init__()
        self.reconstruction_loss = nn.BCELoss(reduction='sum')

        print("Created dts loss.")

        sr = torch.tensor(22050, device=device, requires_grad=False)
        n_fft = torch.tensor(1024, device=device, requires_grad=False)
        fmin = torch.tensor(150.0, device=device, requires_grad=False)
        fmax = torch.tensor(4000.0, device=device, requires_grad=False)
        threshold = torch.tensor(0.1, device=device, requires_grad=False)

        self.transform = InverseMelScale(n_stft=1024, n_mels=80,
                                         sample_rate=22050, f_min=0.0, f_max=8000.0)

    def kl_loss(self, q_dist):
        """
        :param q_dist:
        :return:
        """
        return kl_divergence(q_dist, Normal(torch.zeros_like(q_dist.mean),
                                            torch.ones_like(q_dist.stddev))).sum(-1)

    def alignment_diagonal_score(self, alignments, binary=False):
        """
        Computes alignment prediction score.  i.e diagonal alignment.
        accept shape batch
        :param alignments:
        :param binary:
        :return:
        """
        maxs = alignments.max(dim=1)[0]
        if binary:
            maxs[maxs > 0] = 1
        return maxs.mean(dim=1).mean(dim=0).item()

    def forward(self, model_output, targets, is_validation=False, is_reversed=True):
        """
        :param is_validation:
        :param is_reversed:
        :param model_output:
        :param targets:
        :return:
        """
        mel_target, gate_target, stft = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        stft.requires_grad = False

        # spectral_target = nn.Flatten()(spectral_target)

        if not reversed:
            mel_out, mel_out_post_net, gate_out, _, = model_output
            gate_targets = gate_target.view(-1, 1)
            gate_outs = gate_out.view(-1, 1)

            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
            total = mel_loss + gate_loss
            return {'loss': total,
                    'mel_loss': mel_loss,
                    'gate_loss': gate_loss}
        else:
            mel_out, mel_out_post_net, gate_out, alignment, rev = model_output
            gate_targets = gate_target.view(-1, 1)

            rev_mel_out, b, alignment_rev = rev
            # rev_mel_out = gate_out_rev.view(-1, 1)
            gate_outs = gate_out.view(-1, 1)

            rev_mel_out = torch.flip(rev_mel_out, dims=(1,))

            # second_gate_loss = nn.BCEWithLogitsLoss()(rev_mel_out, gate_targets)
            # second_mse_loss = nn.MSELoss()(rev_mel_out, mel_target)
            l1_loss = nn.L1Loss()(alignment, alignment_rev)

            # alignment_loss = nn.L1Loss()(alignment, rev_alignments)
            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
            total = mel_loss + gate_loss + l1_loss

            # # target mel , padded, mel_out padded to compute inverse, same for mel_out
            # mel_target_padded = torch.nn.functional.pad(mel_target,  (1, 1), "constant", 0)
            # mel_out_post_net_padded = torch.nn.functional.pad(mel_out_post_net,  (1, 1), "constant", 0)
            # mel_out_padded = torch.nn.functional.pad(mel_out,  (1, 1), "constant", 0)
            #
            # for i in range(0, len(stft)):
            #     S_inv_target = librosa.feature.inverse.mel_to_stft(
            #             mel_target_padded[i].detach().cpu().numpy(), n_fft=1024, sr=22050)
            #
            #     S_inv_generated = librosa.feature.inverse.mel_to_stft(
            #             mel_out_post_net_padded[i].detach().cpu().numpy(), n_fft=1024, sr=22050)
            #
            #     S_inv_generated2 = librosa.feature.inverse.mel_to_stft(
            #             mel_out_padded[i].detach().cpu().numpy(), n_fft=1024, sr=22050)
            #
            #     ith_stft = stft[i]
            #     print(ith_stft.shape)
            #     print(S_inv_target[i].shape)
            #     print(S_inv_generated[i].shape)
            #
            #     print(ith_stft.dtype)
            #     print(torch.from_numpy(S_inv_generated[i]).dtype)
            #     print(torch.from_numpy(S_inv_generated2[i]).dtype)
            #
            #     sf_loss1 = nn.L1Loss()(torch.from_numpy(S_inv_target[i]), ith_stft)
            #     sf_loss2 = nn.L1Loss()(torch.from_numpy(S_inv_generated[i]), ith_stft)
            #     sf_loss3 = nn.L1Loss()(torch.from_numpy(S_inv_generated2[i]), ith_stft)
            #
            #     # cross_01 = nn.CrossEntropyLoss()(torch.from_numpy(S_inv_target[i]), ith_stft)
            #     # cross_02 = nn.CrossEntropyLoss()(torch.from_numpy(S_inv_generated[i]), ith_stft)
            #     # cross_03 = nn.CrossEntropyLoss()(torch.from_numpy(S_inv_generated2[i]), ith_stft)
            #     # tensor(2.6611) tensor(14.2158)
            #     # tensor(1283.1836) tensor(2361.4380)
            #     # torch.from_numpy(S_inv)
            #     # mse = torch.square(torch.from_numpy(S_inv) - orginal).mean().item()
            #
            #     print(f"l loss target/target {sf_loss1}, target/generated {sf_loss2} target/generated {sf_loss3}")
            #     # print(f"cross entropy target/target {cross_01}, target/generated {cross_02} target/generated {cross_03}")
            #     # print(cross_01, cross_02)

            return {
                'loss': total,
                'mel_loss': mel_loss,
                'gate_loss': gate_loss
            }