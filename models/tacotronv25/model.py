from math import sqrt

import torch
from torch import nn

from model_trainer.trainer_specs import ExperimentSpecs
from model_trainer.utils import to_gpu, get_mask_from_lengths
from .PreAndPost import Postnet
from .decoder import Decoder
from .encoder import Encoder


class Tacotron25(nn.Module):
    """

    """
    def __init__(self, experiment_specs: ExperimentSpecs, device):
        """

        :param experiment_specs:
        :param device:
        """
        super(Tacotron25, self).__init__()
        self.experiment_specs = experiment_specs
        self.model_trainer_spec = experiment_specs
        self.model_spec = experiment_specs.get_model_spec()
        self.encoder_spec = self.model_spec.get_encoder()
        self.device = device

        self.mask_padding = self.experiment_specs.mask_padding
        self.fp16_run = self.experiment_specs.is_amp()
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
            mask = ~get_mask_from_lengths(output_lengths, self.device)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        """

        :param inputs:
        :return:
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
        During inference pass input to embedding layer
        transpose shape (batch_size, x ,y ) (batch_size, y, z)
        Pass to decoder and get output mel , gate and alignments.

        :param inputs:
        :return:
        """
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_post_net = self.postnet(mel_outputs)
        mel_outputs_post_net = mel_outputs + mel_outputs_post_net
        return self.parse_output(
            [mel_outputs, mel_outputs_post_net, gate_outputs, alignments])
