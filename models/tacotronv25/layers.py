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

class LinearNorm(torch.nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    """

    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
