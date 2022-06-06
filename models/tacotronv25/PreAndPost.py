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
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from .layers import ConvNorm
from .layers import LinearNorm
from model_trainer.trainer_specs import ExperimentSpecs


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
    """
    Default setting post Net Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, specs: ExperimentSpecs, device, is_strict: Optional[bool] = True):
        """

        :param specs:  spec for post net.
        :param device:  device
        :param is_strict:  By default , check if model use same parameter.
        """
        super(Postnet, self).__init__()
        self.device = device
        self.convolutions = nn.ModuleList()

        self.experiment_specs = specs
        self.model_spec = specs.get_model_spec()
        self.specto_spec = self.model_spec.get_spectrogram()

        if is_strict:
            assert self.specto_spec.n_mel_channels() == 80
            assert self.specto_spec.postnet_embedding_dim() == 512
            assert self.specto_spec.postnet_kernel_size() == 5
            assert self.specto_spec.postnet_n_convolutions() == 5

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.specto_spec.n_mel_channels(), self.specto_spec.postnet_embedding_dim(),
                         kernel_size=self.specto_spec.postnet_kernel_size(), stride=1,
                         padding=int((self.specto_spec.postnet_kernel_size() - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(self.specto_spec.postnet_embedding_dim(),
                               affine=self.specto_spec.post_net_batch_affine()))
        )

        for i in range(1, self.specto_spec.postnet_n_convolutions() - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(self.specto_spec.postnet_embedding_dim(),
                             self.specto_spec.postnet_embedding_dim(),
                             kernel_size=self.specto_spec.postnet_kernel_size(), stride=1,
                             padding=int((self.specto_spec.postnet_kernel_size() - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(self.specto_spec.postnet_embedding_dim(),
                                   affine=self.specto_spec.post_net_batch_affine()))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.specto_spec.postnet_embedding_dim(),
                         self.specto_spec.n_mel_channels(),
                         kernel_size=self.specto_spec.postnet_kernel_size(), stride=1,
                         padding=int((self.specto_spec.postnet_kernel_size() - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(self.specto_spec.n_mel_channels(),
                               affine=self.specto_spec.post_net_batch_affine()))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x
