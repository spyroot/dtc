import librosa
from torch import nn
import torch
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F


class DTSLoss(nn.Module):
    """

    """

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, sr=22050, n_fft=2048, fmax=8000,
                 mel_fmax=8000.0):
        super(DTSLoss, self).__init__()
        self.reconstruction_loss = nn.BCELoss(reduction='sum')

        print("Created dts loss.")

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
        Compute  diagonal alignment predictions It is useful
        to measure the alignment consistency of a model
        Args:
            alignments (torch.Tensor): batch of alignments.
            binary (bool): if True, ignore scores and consider attention
            as a binary mask.
        Shape:
            alignments : batch x decoder_steps x encoder_steps
        """
        maxs = alignments.max(dim=1)[0]
        if binary:
            maxs[maxs > 0] = 1
        return maxs.mean(dim=1).mean(dim=0).item()

    def forward(self, model_output, targets, is_reversed=True):
        """
        :param is_reversed:
        :param model_output:
        :param targets:
        :return:
        """
        mel_target, gate_target, spectral_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        spectral_target.requires_grad = False

        spectral_target = nn.Flatten()(spectral_target)

        if not reversed:
            mel_out, mel_out_post_net, gate_out, _, reconstructed, dist = model_output
            gate_targets = gate_target.view(-1, 1)
            gate_outs = gate_out.view(-1, 1)

            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
            total = mel_loss + gate_loss

        else:
            mel_out, mel_out_post_net, gate_out, alignment, reconstructed, \
            dist, rev_mel_out, gate_out_rev, rev_alignments = model_output

            gate_targets = gate_target.view(-1, 1)
            rev_mel_out = gate_out_rev.view(-1, 1)
            gate_outs = gate_out.view(-1, 1)

            rev_mel_out = torch.flip(rev_mel_out, dims=(1,))

            second_gate_loss = nn.BCEWithLogitsLoss()(rev_mel_out, gate_targets)
            second_mse_loss = nn.MSELoss()(rev_mel_out, mel_target)
            l1_loss = nn.L1Loss()(rev_mel_out, mel_out)

            alignment_loss = nn.L1Loss()(alignment, rev_alignments)
            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_outs, gate_targets)
            total = mel_loss + gate_loss + second_gate_loss + second_mse_loss + gate_loss + alignment_loss

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
