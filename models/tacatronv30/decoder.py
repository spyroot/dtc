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
# M
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from model_trainer.trainer_specs import ExperimentSpecs
from .preandpost import Prenet
from .attention import Attention
from .layers import LinearNorm
from model_trainer.utils import get_mask_from_lengths


class Decoder(nn.Module):
    """
    Decoder consumes to predict a spectrogram.

    Input characters are represented using a learned 512-dimensional
    character embedding, which are passed through a stack of 3 convolutional
    layers each containing 512 filters with shape 5   1, i.e., where
    each filter spans 5 characters, followed by batch normalization [18]
    and ReLU activations. As in Tacotron, these convolutional layers
    model longer-term context (e.g., N-grams) in the input character
    sequence. The output of the final convolutional layer is passed into a
    single bi-directional  LSTM layer containing 512 units (256
    in each direction) to generate the encoded features.
    """

    def __init__(self, specs: ExperimentSpecs, device, is_strict=True):
        """
        Decoder based on tacotron2 spec
        :param specs:
        """
        super(Decoder, self).__init__()

        # spec
        self.experiment_specs = specs
        self.model_spec = specs.get_model_spec()
        self.specto_spec = self.model_spec.get_spectrogram()
        self.device = device

        # model param
        self.n_mel_channels = self.specto_spec.n_mel_channels()
        self.decoder_fps = self.specto_spec.decoder_fps()
        self.encoder_embedding_dim = self.specto_spec.encoder_embedding_dim()
        # attention rnn
        self.attention_rnn_dim = self.specto_spec.attention_rnn_dim()

        # decoder
        self.decoder_rnn_dim = self.specto_spec.decoder_rnn_dim()
        self.max_decoder_steps = self.specto_spec.max_decoder_steps()
        self.gate_threshold = self.specto_spec.gate_threshold()
        self.p_attention_dropout = self.specto_spec.p_attention_dropout()
        self.p_decoder_dropout = self.specto_spec.p_decoder_dropout()

        # pre-net
        self.pre_net_dim = self.specto_spec.pre_net_dim()

        if is_strict:
            assert self.decoder_fps == 1
            assert self.encoder_embedding_dim == 512
            assert self.attention_rnn_dim == 1024
            assert self.specto_spec.attention_dim() == 128
            assert self.decoder_rnn_dim == 1024
            assert self.max_decoder_steps == 1000
            assert self.gate_threshold == 0.5
            assert self.p_attention_dropout == 0.1
            assert self.p_decoder_dropout == 0.1
            assert self.pre_net_dim == 256

        # layers
        self.pre_net = Prenet(
                self.specto_spec.n_mel_channels() * self.specto_spec.decoder_fps(),
                [self.specto_spec.pre_net_dim(), self.specto_spec.pre_net_dim()])

        self.attr_rnn = nn.LSTMCell(
                self.specto_spec.pre_net_dim() +
                self.specto_spec.encoder_embedding_dim(),
                self.specto_spec.attention_rnn_dim())

        self.attention_layer = Attention(
                self.specto_spec.attention_rnn_dim(),
                self.specto_spec.encoder_embedding_dim(),
                self.specto_spec.attention_dim(),
                self.specto_spec.attention_location_n_filters(),
                self.specto_spec.attention_location_kernel_size())

        self.decoder_rnn = nn.LSTMCell(
                self.specto_spec.attention_rnn_dim() +
                self.specto_spec.encoder_embedding_dim(),
                self.specto_spec.decoder_rnn_dim(), 1)

        self.linear_projection = LinearNorm(
                self.specto_spec.decoder_rnn_dim() +
                self.specto_spec.encoder_embedding_dim(),
                self.specto_spec.n_mel_channels() * self.specto_spec.decoder_fps())

        self.gate_layer = LinearNorm(
                self.specto_spec.decoder_rnn_dim() +
                self.specto_spec.encoder_embedding_dim(), 1, bias=True, w_init_gain='sigmoid')

        # states
        self.memory = None
        self.attr_hidden = None
        self.attr_cell = None
        self.decoder_hidden = None
        self.decoder_cell = None

        self.attr_weights = None
        self.attr_weights_cum = None
        self.attr_context = None

        self.processed_memory = None
        self.mask = None

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, self.n_mel_channels * self.decoder_fps).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory

        memory shape torch.Size([32, 155, 512])
        mask  shape torch.Size([32, 155])

        :param memory: encoder outputs
        :param mask: Mask for padded data if training, expects None for inference
        :return:
        """
        batch_size = memory.size(0)
        max_frame = memory.size(1)

        self.attr_hidden = Variable(
                memory.data.new(batch_size, self.attention_rnn_dim).zero_())
        self.attr_cell = Variable(
                memory.data.new(batch_size, self.attention_rnn_dim).zero_())
        self.decoder_hidden = Variable(
                memory.data.new(batch_size, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(
                memory.data.new(batch_size, self.decoder_rnn_dim).zero_())

        self.attr_weights = Variable(
                memory.data.new(batch_size, max_frame).zero_())
        self.attr_weights_cum = Variable(
                memory.data.new(batch_size, max_frame).zero_())
        self.attr_context = Variable(
                memory.data.new(batch_size, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """
        Original decoder from tacotron2,
        inputs used for teacher-forced training, i.e. mel-specs

        :param decoder_inputs:
        :return:
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(decoder_inputs.size(0),
                                             int(decoder_inputs.size(1) / self.decoder_fps), -1)

        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attr_context), -1)
        self.attr_hidden, self.attr_cell = self.attr_rnn(cell_input, (self.attr_hidden, self.attr_cell))

        self.attr_hidden = F.dropout(self.attr_hidden, self.p_attention_dropout, self.training)

        attr_weights_cat = torch.cat(
                (self.attr_weights.unsqueeze(1),
                 self.attr_weights_cum.unsqueeze(1)), dim=1)

        self.attr_context, self.attr_weights = self.attention_layer(
                self.attr_hidden, self.memory, self.processed_memory,
                attr_weights_cat, self.mask)

        # self.attr_weights shape torch.Size([32, 161])
        #   print("self.attr_weights shape", self.attr_weights.shape)

        # self.attr_weights_cum shape torch.Size([32, 161])
        self.attr_weights_cum += self.attr_weights

        #  print("self.attr_weights_cum shape", self.attr_weights_cum.shape)

        decoder_input = torch.cat((self.attr_hidden, self.attr_context), -1)

        # Decoder input shape torch.Size([32, 1536])
        # print("Decoder input shape", decoder_input.shape)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input,
                                                                  (self.decoder_hidden, self.decoder_cell))

        self.decoder_hidden = F.dropout(
                self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
                (self.decoder_hidden, self.attr_context), dim=1)

        decoder_output = self.linear_projection(
                decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attr_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.pre_net(decoder_inputs)

        mask = ~get_mask_from_lengths(memory_lengths, self.device)

        assert mask.device == self.device
        assert memory.device == self.device
        self.initialize_decoder_states(memory, mask=mask)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = \
            self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []

        while True:
            decoder_input = self.pre_net(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments
