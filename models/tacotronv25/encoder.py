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

    def __init__(self, experiment_specs: ExperimentSpecs, device):
        """

        :param experiment_specs:
        :param device:
        """
        super(Encoder, self).__init__()
        self.device = device

        self.conv_kernel_size = experiment_specs.encoder_kernel_size
        self.embedding_dim = experiment_specs.encoder_embedding_dim
        self.num_conv_layers = experiment_specs.encoder_n_convolutions
        self.forward_pass_dropout_rate = 0.5
        self.stride_size = 1

        convolutions = []
        for _ in range(experiment_specs.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(self.embedding_dim, self.embedding_dim,
                         kernel_size=self.conv_kernel_size, stride=1,
                         padding=int((experiment_specs.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(experiment_specs.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.embedding_dim,
                            int(experiment_specs.encoder_embedding_dim / 2), 1,
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
        # print("Len", input_lengths)
        # pytorch tensor are not reversible, hence the conversion
        # input_lengths = input_lengths.cpu().numpy()
        # flipped = torch.fliplr(input_lengths)

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