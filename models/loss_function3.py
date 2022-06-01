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
        Compute how diagonal alignment predictions are. It is useful
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
        else:
            mel_out, mel_out_post_net, gate_out, alignment, reconstructed, \
            dist, mel_out_rev, gate_out_rev, alignments_rev = model_output
            print("gate)out_rev", gate_out_rev.shape)

        gate_targets = gate_target.view(-1, 1)
        gate_outs = gate_out.view(-1, 1)
        gate_out_revs = gate_out_rev.view(-1, 1)

        print("Shape before", mel_out_rev.shape)
        mel_out_rev = torch.flip(mel_out_rev, dims=(1,))
        print("Shape after", mel_out_rev.shape)

        gate_loss = nn.MSELoss()(gate_outs, gate_out_revs)
        print("Gate loss", gate_loss.item())

        second_mse_loss = nn.MSELoss()(mel_out_rev, mel_target)
        print("mel loss", second_mse_loss.item())

        alignment_loss = nn.MSELoss()(alignment, alignments_rev)
        print("alignments loss", alignment_loss.item())

        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)

        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_targets, gate_outs)

        kl_loss = self.kl_loss(dist).sum()

        # # backward decoder loss (if enabled)
        # if self.config.bidirectional_decoder:
        #     if self.config.loss_masking:
        #         decoder_b_loss = self.criterion(
        #                 torch.flip(decoder_b_output, dims=(1,)), mel_input,
        #                 output_lens)
        #     else:
        #         decoder_b_loss = self.criterion(torch.flip(decoder_b_output, dims=(1,)), mel_input)
        #     decoder_c_loss = torch.nn.functional.l1_loss(torch.flip(decoder_b_output, dims=(1,)), decoder_output)
        #     loss += self.decoder_alpha * (decoder_b_loss + decoder_c_loss)
        #     return_dict['decoder_b_loss'] = decoder_b_loss
        #     return_dict['decoder_c_loss'] = decoder_c_loss


        # print(reconstructed.type())
        # print(spectral_target.type())
        # print("mel loss ", mel_loss.item())
        # print("gate loss ", gate_loss.item())
        #
        # print(reconstructed.shape)
        # print(spectral_target.shape)

        bce_loss = nn.BCEWithLogitsLoss()(reconstructed, spectral_target)
        spectral_loss = bce_loss + kl_loss

        #  spectral_loss = nn.BCELoss()(reconstructed, spectral_target)
        #print(reconstructed)
        #print("Spectral loss ", spectral_loss.item())
        # print("kl loss", kl_loss.item())
        # print("bce_loss ", kl_loss.item())

        total = mel_loss + gate_loss + spectral_loss
        # print("Spectral loss ", spectral_loss.item())
        # print("total loss", total.item())

        # plot_mel_fbank(mel_filters_librosa, "Mel Filter Bank - librosa")
        # mse = torch.square(mel_filters - mel_filters_librosa).mean().item()
        # print("Mean Square Difference: ", self.mel_filters_librosa.mean().item())

        return {'loss': total,
                'mel_loss': mel_loss,
                'gate_loss': gate_loss,
                'spectral_loss': spectral_loss}
