from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from model_trainer.model_trainer_specs import ExperimentSpecs
from models.PreAndPost import Postnet
from models.decoder import Decoder
from models.layers import ConvNorm
from tacotron2.utils import to_gpu, get_mask_from_lengths


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, experiment_specs: ExperimentSpecs, device):
        super(Encoder, self).__init__()
        self.device = device

        convolutions = []
        for _ in range(experiment_specs.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(experiment_specs.encoder_embedding_dim,
                         experiment_specs.encoder_embedding_dim,
                         kernel_size=experiment_specs.encoder_kernel_size, stride=1,
                         padding=int((experiment_specs.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(experiment_specs.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(experiment_specs.encoder_embedding_dim,
                            int(experiment_specs.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        """

        :param x:
        :param input_lengths:
        :return:
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

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


class Tacotron2(nn.Module):
    """

    """

    def __init__(self, experiment_specs: ExperimentSpecs, device):
        """

        :param hparams:
        """
        super(Tacotron2, self).__init__()
        self.experiment_specs = experiment_specs
        self.model_trainer_spec = experiment_specs
        self.model_spec = experiment_specs.get_model_spec()
        self.encoder_spec = self.model_spec.get_encoder()
        self.device = device

        self.mask_padding = self.experiment_specs.mask_padding
        self.fp16_run = self.experiment_specs.is_fp16_run()
        self.n_mel_channels = self.encoder_spec.n_mel_channels()
        self.n_frames_per_step = self.experiment_specs.n_frames_per_step

        #
        self.embedding = nn.Embedding(self.experiment_specs.n_symbols, self.experiment_specs.symbols_embedding_dim)
        #
        std = sqrt(2.0 / (self.experiment_specs.n_symbols + self.experiment_specs.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std

        self.embedding.weight.data.uniform_(-val, val)
        #
        self.encoder = Encoder(experiment_specs, device=self.device)
        self.decoder = Decoder(experiment_specs, device=self.device)
        self.postnet = Postnet(experiment_specs, device=self.device)

    def parse_batch(self, batch):
        """

        :param batch:
        :return:
        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (text_padded, input_lengths, mel_padded, max_len, output_lengths), (mel_padded, gate_padded)

    def parse_output(self, outputs, output_lengths=None):
        """

        Args:
            outputs:
            output_lengths:

        Returns:

        """
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        """

        Args:
            inputs:

        Returns:

        """
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        """

        Args:
            inputs:

        Returns:

        """
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
