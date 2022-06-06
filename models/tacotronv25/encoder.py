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
from torch import nn
from torch.nn import functional as F
from model_trainer.trainer_specs import ExperimentSpecs
from .layers import ConvNorm
from torch import Tensor


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, specs: ExperimentSpecs, device):
        """
        :param spec:
        :param device:
        """
        super(Encoder, self).__init__()
        self.experiment_specs = specs
        self.model_spec = specs.get_model_spec()
        self.specto_spec = self.model_spec.get_spectrogram()
        self.device = device

        self.conv_kernel_size = self.specto_spec.encoder_kernel_size()
        self.embedding_dim = self.specto_spec.encoder_embedding_dim()
        self.num_conv_layers = self.specto_spec.encoder_n_convolutions()
        self.forward_pass_dropout_rate = 0.5
        self.stride_size = 1

        convolutions = []
        for _ in range(self.specto_spec.encoder_n_convolutions()):
            conv_layer = nn.Sequential(
                ConvNorm(self.embedding_dim, self.embedding_dim,
                         kernel_size=self.conv_kernel_size, stride=1,
                         padding=int((self.specto_spec.encoder_kernel_size() - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.specto_spec.encoder_embedding_dim()))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.embedding_dim,
                            int(self.specto_spec.encoder_embedding_dim() / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x: Tensor, input_lengths):
        """

        :param x:
        :param input_lengths:
        :return:
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), self.forward_pass_dropout_rate, self.training)

        x = x.transpose(1, 2)

        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu().numpy(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def inference(self, x):
        """

        :param x:
        :return:
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs
