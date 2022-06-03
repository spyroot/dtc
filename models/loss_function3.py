import librosa
from torch import nn, optim
import torch
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
import numpy as np
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

    # if len(axes) != x.ndim:
    #     raise ValueError("Shape mismatch between axes={} and input x.shape={}".format(axes, x.shape))
    #
    # if ndim < x.ndim:
    #     raise ValueError("Cannot expand x.shape={} to fewer dimensions ndim={}".format(x.shape, ndim))

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

        # max_wav_value: 32768.0
        # frames_per_step: 1
        # sampling_rate: 22050
        # filter_length: 1024  # length of the FFT window
        # win_length: 1024  # each frame of audio is windowed by
        # hop_length: 256
        # n_mel_channels: 80
        # mel_fmin: 0.0
        # mel_fmax: 8000.0
        # symbols_embedding_dim: 512

        self.transform = InverseMelScale(n_stft=1024, n_mels=80,
                                         sample_rate=22050, f_min=0.0, f_max=8000.0)

        # n_stft: int,
        # n_mels: int = 128,
        # sample_rate: int = 16000,
        # f_min: float = 0.0,
        # f_max: Optional[float] = None,
        # max_iter: int = 100000,
        # tolerance_loss: float = 1e-5,
        # tolerance_change: float = 1e-8,
        # sgdargs: Optional[dict] = None,
        # norm: Optional[str] = None,
        # mel_scale: str = "htk",
        #

        # self.mel_filters_librosa = self.mel_filters_librosa = librosa.filters.mel(
        #         sr=sampling_rate,
        #         n_fft=n_fft,
        #         fmin=mel_fmin,
        #         fmax=mel_fmax,
        #         norm="slaney",
        #         htk=True,
        # ).T

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

    def _nnls_lbfgs_block(A, B, x_init=None, **kwargs):
        """ Solve the constrained problem over a single block
        :param np. np.ndarray [shape=(m, d)]
        :param B:
        :param x_init: np.ndarray [shape=(d, N)]
        :param kwargs:  Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`
        :return:
        """

        # If we don't have an initial point, start at the projected
        # least squares solution
        if x_init is None:
            x_init = torch.einsum("fm,...mt->...ft", torch.linalg.pinv(A), B, optimize=True)
            np.clip(x_init, 0, None, out=x_init)

        # Adapt the hessian approximation to the dimension of the problem
        kwargs.setdefault("m", A.shape[1])

        # Construct non-negative bounds
        bounds = [(0, None)] * x_init.size
        shape = x_init.shape

        # optimize
        optimizer = optim.LBFGS([x_lbfgs],
                                history_size=10,
                                max_iter=4,
                                line_search_fn="strong_wolfe")
        h_lbfgs = []
        for i in range(100):
            optimizer.zero_grad()
            objective = f(x_lbfgs)
            objective.backward()
            optimizer.step(lambda: f(x_lbfgs))
            h_lbfgs.append(objective.item())

        x, obj_value, diagnostics = scipy.optimize.fmin_l_bfgs_b(_nnls_obj,
                                                                 x_init,
                                                                 args=(shape, A, B),
                                                                 bounds=bounds, **kwargs)
        # reshape the solution
        return x.reshape(shape)

MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10

    def nnls(self, A, B, **kwargs):
        if B.ndim == 1:
            print("Unsuported")
            sys.exit()
            # return scipy.optimize.nnls(A, B)[0]

        n_columns = MAX_MEM_BLOCK // (np.prod(B.shape[:-1]) * A.itemsize)
        n_columns = max(n_columns, 1)

        # Process in blocks:
        if B.shape[-1] <= n_columns:
            return self._nnls_lbfgs_block(A, B, **kwargs).astype(A.dtype)

        x = torch.einsum("fm,...mt->...ft", torch.linalg.pinv(A), B, optimize=True)
        np.clip(x, 0, None, out=x)
        x_init = x

        for bl_s in range(0, x.shape[-1], n_columns):
            bl_t = min(bl_s + n_columns, B.shape[-1])
            x[..., bl_s:bl_t] = self._nnls_lbfgs_block(
                    A, B[..., bl_s:bl_t], x_init=x_init[..., bl_s:bl_t], **kwargs)
        return x

    def mel_to_stft(M, *, sr=22050, n_fft=2048, power=2.0, **kwargs):
        mel_basis = filters.mel(
                sr=sr, n_fft=n_fft, n_mels=M.shape[-2], dtype=M.dtype, **kwargs
        )

        # Find the non-negative least squares solution, and apply
        # the inverse exponent.
        # We'll do the exponentiation in-place.
        inverse = nnls(mel_basis, M)
        return torch.power(inverse, 1.0 / power, out=inverse)

    def forward(self, model_output, targets, is_validation=False, is_reversed=True):
        """
        :param is_reversed:
        :param model_output:
        :param targets:
        :return:
        """
        mel_target, gate_target, stft = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        # sfts.requires_grad = False

        # spectral_target = nn.Flatten()(spectral_target)

        if not reversed:
            mel_out, mel_out_post_net, gate_out, _, = model_output
            gate_targets = gate_target.view(-1, 1)
            gate_outs = gate_out.view(-1, 1)

            # max_wav_value: 32768.0
            # frames_per_step: 1
            # sampling_rate: 22050
            # filter_length: 1024  # length of the FFT window
            # win_length: 1024  # each frame of audio is windowed by
            # hop_length: 256
            # n_mel_channels: 80
            # mel_fmin: 0.0
            # mel_fmax: 8000.0
            # symbols_embedding_dim: 512
            #
            # melspec_librosa = librosa.feature.melspectrogram(
            #         y=waveform.numpy()[0],
            #         sr=22050,
            #         n_fft=1024,
            #         hop_length=1024,
            #         win_length=1024,
            #         center=True,
            #         pad_mode="reflect",
            #         power=2.0,
            #         n_mels=80,
            #         norm="slaney",
            #         htk=True,
            # )

            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
            total = mel_loss + gate_loss
            return {'loss': total,
                    'mel_loss': mel_loss,
                    'gate_loss': gate_loss}
        else:
            mel_out, mel_out_post_net, gate_out, alignment, rev = model_output
            gate_targets = gate_target.view(-1, 1)

            # # tensor(2.6611) tensor(14.2158)
            # # tensor(1283.1836) tensor(2361.4380)
            #
            # # torch.from_numpy(S_inv)
            # # mse = torch.square(torch.from_numpy(S_inv) - orginal).mean().item()
            # print(sf_loss1, sf_loss2)
            # print(cross_01, cross_02)

            # print("type a", inv.dtype)
            # print("type b", orginal.dtype)
            # for i in range(0, len(sfts)):
            #     orginal = sfts[i]
            #     print(orginal.shape)
            #     print(S_inv[i].shape)
            #     print(S_gen[i].shape)
            #
            #     sf_loss1 = nn.MSELoss()(torch.from_numpy(S_inv[i]), orginal)
            #     sf_loss2 = nn.MSELoss()(torch.from_numpy(S_gen[i]), orginal)
            #     cross_01 = nn.CrossEntropyLoss()(torch.from_numpy(S_inv[i]), orginal)
            #     cross_02 = nn.CrossEntropyLoss()(torch.from_numpy(S_gen[i]), orginal)
            #
            #     # tensor(2.6611) tensor(14.2158)
            #     # tensor(1283.1836) tensor(2361.4380)
            #
            #     # torch.from_numpy(S_inv)
            #     # mse = torch.square(torch.from_numpy(S_inv) - orginal).mean().item()
            #     print(sf_loss1, sf_loss2)
            #     print(cross_01, cross_02)

            # end_time = time.monotonic()
            # print(timedelta(seconds=end_time - start_time))

            # stop = datetime.datetime.now()
            # print(delta = b - a)

            # Sinv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)

            rev_mel_out, b, c = rev
            # rev_mel_out = gate_out_rev.view(-1, 1)
            gate_outs = gate_out.view(-1, 1)

            rev_mel_out = torch.flip(rev_mel_out, dims=(1,))

            # second_gate_loss = nn.BCEWithLogitsLoss()(rev_mel_out, gate_targets)
            # second_mse_loss = nn.MSELoss()(rev_mel_out, mel_target)
            l1_loss = nn.L1Loss()(rev_mel_out, mel_out)

            # alignment_loss = nn.L1Loss()(alignment, rev_alignments)
            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
            total = mel_loss + gate_loss + l1_loss

            p1d = (1, 1)
            # target mel , padded, mel_out padded to compute inverse, same for mel_out
            mel_target_padded = torch.nn.functional.pad(mel_target, p1d, "constant", 0)
            mel_out_post_net_padded = torch.nn.functional.pad(mel_out_post_net, p1d, "constant", 0)
            mel_out_padded = torch.nn.functional.pad(mel_out, p1d, "constant", 0)

            for i in range(0, len(stft)):
                S_inv_target = librosa.feature.inverse.mel_to_stft(
                        mel_target_padded[i].detach().cpu().numpy(), n_fft=1024, sr=22050)

                S_inv_generated = librosa.feature.inverse.mel_to_stft(
                        mel_out_post_net_padded[i].detach().cpu().numpy(), n_fft=1024, sr=22050)

                S_inv_generated2 = librosa.feature.inverse.mel_to_stft(
                        mel_out_padded[i].detach().cpu().numpy(), n_fft=1024, sr=22050)

                ith_stft = stft[i]
                print(ith_stft.shape)
                print(S_inv_target[i].shape)
                print(S_inv_generated[i].shape)

                sf_loss1 = nn.MSELoss()(torch.from_numpy(S_inv_target[i]), ith_stft)
                sf_loss2 = nn.MSELoss()(torch.from_numpy(S_inv_generated[i]), ith_stft)
                sf_loss3 = nn.MSELoss()(torch.from_numpy(S_inv_generated2[i]), ith_stft)

                # cross_01 = nn.CrossEntropyLoss()(torch.from_numpy(S_inv_target[i]), ith_stft)
                # cross_02 = nn.CrossEntropyLoss()(torch.from_numpy(S_inv_generated[i]), ith_stft)
                # cross_03 = nn.CrossEntropyLoss()(torch.from_numpy(S_inv_generated2[i]), ith_stft)

                # tensor(2.6611) tensor(14.2158)
                # tensor(1283.1836) tensor(2361.4380)

                # torch.from_numpy(S_inv)
                # mse = torch.square(torch.from_numpy(S_inv) - orginal).mean().item()
                print(f"mse loss target/target {sf_loss1}, target/generated {sf_loss2} target/generated {sf_loss3}")
                # print(f"cross entropy target/target {cross_01}, target/generated {cross_02} target/generated {cross_03}")
                # print(cross_01, cross_02)

            return {'loss': total,
                    'mel_loss': mel_loss,
                    'gate_loss': gate_loss}

        # # kl_loss = self.kl_loss(dist).sum()
        #
        # # bce_loss = nn.BCEWithLogitsLoss()(reconstructed, spectral_target)
        # # spectral_loss = bce_loss + kl_loss
        #
        # # print(f"backward gate: {second_gate_loss.item():.4f} backward mel: {second_mse_loss.item():.4f} "
        # #       f"l1 a/b: {l1_loss.item():.4f}, alignment: {alignment_loss.item():.4f}, "
        # #       f"base mel: {mel_loss.item():.4f}, base: gate {gate_loss.item():.4f} "
        # #       f"spectral: {spectral_loss.item():.4f} bce: {bce_loss.item():.4f}")
        #
        # print(f"backward gate: {second_gate_loss.item():.4f} backward mel: {second_mse_loss.item():.4f} "
        #       f"l1 a/b: {l1_loss.item():.4f}, alignment: {alignment_loss.item():.4f}, "
        #       f"base mel: {mel_loss.item():.4f}, base: gate {gate_loss.item():.4f} ")
        # # spectral_loss = nn.BCELoss()(reconstructed, spectral_target)
        # # print(reconstructed)
        # # print("Spectral loss ", spectral_loss.item())
        # # print("kl loss", kl_loss.item())
        # # print("bce_loss ", kl_loss.item())
        #
        # total = mel_loss + gate_loss + second_gate_loss + second_mse_loss + gate_loss + alignment_loss
        # # print("Spectral loss ", spectral_loss.item())
        # # print("total loss", total.item())
        #
        # return {'loss': total,
        #         'mel_loss': mel_loss,
        #         'gate_loss': gate_loss,
        #         'spectral_loss': spectral_loss}

        # old forward pass with VAE that doesn't produce any good results
        #
        #
        # def forward(self, model_output, targets, is_reversed=True):
        #     """
        #     :param is_reversed:
        #     :param model_output:
        #     :param targets:
        #     :return:
        #     """
        #     mel_target, gate_target, sfts = targets[0], targets[1], targets[2]
        #     mel_target.requires_grad = False
        #     gate_target.requires_grad = False
        #     sfts.requires_grad = False
        #
        #     # spectral_target = nn.Flatten()(spectral_target)
        #
        #     if not reversed:
        #         mel_out, mel_out_post_net, gate_out, _, reconstructed, dist = model_output
        #         gate_targets = gate_target.view(-1, 1)
        #         gate_outs = gate_out.view(-1, 1)
        #
        #         mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
        #         gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
        #         total = mel_loss + gate_loss
        #
        #     else:
        #         mel_out, mel_out_post_net, gate_out, alignment, reconstructed, \
        #         dist, rev_mel_out, gate_out_rev, rev_alignments = model_output
        #
        #         gate_targets = gate_target.view(-1, 1)
        #         rev_mel_out = gate_out_rev.view(-1, 1)
        #         gate_outs = gate_out.view(-1, 1)
        #
        #         rev_mel_out = torch.flip(rev_mel_out, dims=(1,))
        #
        #         second_gate_loss = nn.BCEWithLogitsLoss()(rev_mel_out, gate_targets)
        #         second_mse_loss = nn.MSELoss()(rev_mel_out, mel_target)
        #         l1_loss = nn.L1Loss()(rev_mel_out, mel_out)
        #
        #         alignment_loss = nn.L1Loss()(alignment, rev_alignments)
        #         mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
        #         gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
        #         total = mel_loss + gate_loss + second_gate_loss + second_mse_loss + gate_loss + alignment_loss
        #
        #         return {'loss': total,
        #                 'mel_loss': mel_loss,
        #                 'gate_loss': gate_loss}
