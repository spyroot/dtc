#
# Model based on original Natural TTS Synthesis by Conditioning WaveNet
# on Mel Spectrogram Predictions.
#
#  - My modification focused on additional vector used to compute extra loss term.
#  - Additional VAE as regularization layer.
#  - Modified encoder seq.
#
# Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly,
# Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous,
# Yannis Agiomyrgiannakis, Yonghui Wu
#
# https://arxiv.org/abs/1712.05884
#
# Mustafa B.
import torch
from torch import nn
from torch.nn import functional as F

from .layers import ConvNorm
from .layers import LinearNorm


class Prenet(nn.Module):
    """
    The prediction from the previous time step is first
    passed through a small pre-net containing 2 fully connected layers
    of 256 hidden ReLU units
    """
    def __init__(self, in_dim, sizes):
        """

        :param in_dim:
        :param sizes:
        """
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, experiment_specs, device):
        super(Postnet, self).__init__()
        self.device = device
        self.convolutions = nn.ModuleList()
        self.experiment_specs = experiment_specs
        self.model_spec = experiment_specs.get_model_spec()
        self.encoder_spec = self.model_spec.get_spectrogram()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.encoder_spec.n_mel_channels(), experiment_specs.postnet_embedding_dim,
                         kernel_size=experiment_specs.postnet_kernel_size, stride=1,
                         padding=int((experiment_specs.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(experiment_specs.postnet_embedding_dim))
        )

        for i in range(1, experiment_specs.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(experiment_specs.postnet_embedding_dim,
                             experiment_specs.postnet_embedding_dim,
                             kernel_size=experiment_specs.postnet_kernel_size, stride=1,
                             padding=int((experiment_specs.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(experiment_specs.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(experiment_specs.postnet_embedding_dim, self.encoder_spec.n_mel_channels(),
                         kernel_size=experiment_specs.postnet_kernel_size, stride=1,
                         padding=int((experiment_specs.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(self.encoder_spec.n_mel_channels()))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x
