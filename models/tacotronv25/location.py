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
from .layers import ConvNorm, LinearNorm
from torch import nn

class LocationLayer(nn.Module):
    """

    """
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        """

        :param attention_n_filters:
        :param attention_kernel_size:
        :param attention_dim:
        """
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        """

        :param attention_weights_cat:
        :return:
        """
        # shape torch.Size([64, 2, 161])
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention
