from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from model_trainer.trainer_specs import ExperimentSpecs
from .PreAndPost import Prenet
from .attention import Attention
from .layers import LinearNorm
from model_trainer.utils import get_mask_from_lengths


class Decoder(nn.Module):
    def __init__(self, specs: ExperimentSpecs, device):
        """
        Decoder based on tacotron2 spec
        :param specs:
        """
        super(Decoder, self).__init__()
        self.experiment_specs = specs
        self.model_spec = specs.get_model_spec()
        self.encoder_spec = self.model_spec.get_encoder()
        self.device = device

        self.n_mel_channels = self.encoder_spec.n_mel_channels()
        self.n_frames_per_step = self.experiment_specs.n_frames_per_step
        self.encoder_embedding_dim = self.experiment_specs.encoder_embedding_dim
        self.attention_rnn_dim = specs.attention_rnn_dim
        self.decoder_rnn_dim = specs.decoder_rnn_dim
        self.pre_net_dim = specs.prenet_dim
        self.max_decoder_steps = specs.max_decoder_steps
        self.gate_threshold = specs.gate_threshold
        self.p_attention_dropout = specs.p_attention_dropout
        self.p_decoder_dropout = specs.p_decoder_dropout

        self.pre_net = Prenet(
                self.encoder_spec.n_mel_channels() * specs.n_frames_per_step,
                [specs.prenet_dim, specs.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
                specs.prenet_dim + specs.encoder_embedding_dim,
                specs.attention_rnn_dim)

        self.attention_layer = Attention(
                specs.attention_rnn_dim, specs.encoder_embedding_dim,
                specs.attention_dim, specs.attention_location_n_filters,
                specs.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
                specs.attention_rnn_dim + specs.encoder_embedding_dim,
                specs.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
                specs.decoder_rnn_dim + specs.encoder_embedding_dim,
                self.encoder_spec.n_mel_channels() * specs.n_frames_per_step)

        self.gate_layer = LinearNorm(
                specs.decoder_rnn_dim + specs.encoder_embedding_dim, 1,
                bias=True, w_init_gain='sigmoid')

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
        decoder_input = Variable(memory.data.new(
                B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """
        Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory

        :param memory: encoder outputs
        :param mask: Mask for padded data if training, expects None for inference
        :return:
        """

        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
                B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
                B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
                B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
                B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
                B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
                B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
                B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
                decoder_inputs.size(0),
                int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
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
        mel_outputs = mel_outputs.view(
                mel_outputs.size(0), -1, self.n_mel_channels)
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
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
                cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
                self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
                (self.attention_weights.unsqueeze(1),
                 self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
                self.attention_hidden, self.memory, self.processed_memory,
                attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
                (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
                decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
                self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
                (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
                decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

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
