import librosa
from torch import nn
import torch


class Tacotron2Loss(nn.Module):
    """

    """

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, sr=22050, n_fft=2048, fmax=8000,
                 mel_fmax=8000.0):
        super(Tacotron2Loss, self).__init__()

        # self.mel_filters_librosa = self.mel_filters_librosa = librosa.filters.mel(
        #         sr=sampling_rate,
        #         n_fft=n_fft,
        #         fmin=mel_fmin,
        #         fmax=mel_fmax,
        #         norm="slaney",
        #         htk=True,
        # ).T

    def forward(self, model_output, targets):
        """
        :param model_output:
        :param targets:
        :return:
        """
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False

        mel_out, mel_out_post_net, gate_out, _ = model_output
        gate_targeT = gate_target.view(-1, 1)
        gate_outT = gate_out.view(-1, 1)

        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_targeT, gate_outT)

        # plot_mel_fbank(mel_filters_librosa, "Mel Filter Bank - librosa")
        # mse = torch.square(mel_filters - mel_filters_librosa).mean().item()
        # print("Mean Square Difference: ", self.mel_filters_librosa.mean().item())

        return mel_loss + gate_loss
