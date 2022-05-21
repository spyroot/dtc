from torch import nn
import torch


class Tacotron2Loss(nn.Module):
    """

    """

    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mel_filters_librosa = self.mel_filters_librosa = librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=0.0,
                fmax=sample_rate / 2.0,
                norm="slaney",
                htk=True,
        ).T

    def forward(self, model_output, targets):
        """

        :param model_output:
        :param targets:
        :return:
        """
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_post_net, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_post_net, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)


        #plot_mel_fbank(mel_filters_librosa, "Mel Filter Bank - librosa")

        mse = torch.square(mel_filters - mel_filters_librosa).mean().item()
        print("Mean Square Difference: ", mse)

        return mel_loss + gate_loss
