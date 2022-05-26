import librosa
from torch import nn
import torch
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F


class Tacotron2Loss(nn.Module):
    """

    """

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, sr=22050, n_fft=2048, fmax=8000,
                 mel_fmax=8000.0):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        """
        :param model_output:
        :param targets:
        :return:
        """
        mel_target, gate_target, = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False

        mel_out, mel_out_post_net, gate_out, _ = model_output
        gate_targeT = gate_target.view(-1, 1)
        gate_outT = gate_out.view(-1, 1)

        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_targeT, gate_outT)

        total = mel_loss + gate_loss
        return {'loss': total,
                'mel_loss': mel_loss,
                'gate_loss': gate_loss}
