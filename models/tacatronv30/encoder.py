from torch import nn
from torch.nn import functional as F
from model_trainer.trainer_specs import ExperimentSpecs
from .layers import ConvNorm
from torch import Tensor


# We minimize the summed mean squared error (MSE) from before
# and after the post-net to aid convergence. We also experimented
# with a log-likelihood loss by modeling the output distribution with
# a Mixture Density Network [23, 24] to avoid assuming a constant
# variance over time, but found that these were more difficult to train
# and they did not lead to better sounding samples.


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM

    The encoder output is consumed by an attention network which
    summarizes the full encoded sequence as a fixed-length context vector
    for each decoder output step.

    We use the location-sensitive attention
    from [21], which extends the additive attention mechanism [22] to
    use cumulative attention weights from previous decoder time steps
    as an additional feature.
    This encourages the model to move forward consistently through the input,
    mitigating potential failure modes where some subsequences are repeated or
    ignored by the decoder.

    Attention probabilities are computed after projecting inputs and location
    features to 128-dimensional hidden representations. Location
    features are computed using 32 1-D convolution filters of length 31.

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

    def forward(self, x, input_lengths):
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
