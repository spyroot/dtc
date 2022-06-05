import copy
import sys
from math import sqrt

import torch
from torch import nn
from torch import Tensor

from model_trainer.trainer_specs import ExperimentSpecs
from model_trainer.utils import to_gpu, get_mask_from_lengths
from .preandpost import Postnet
from .decoder import Decoder
from .encoder import Encoder
from torch.distributions.normal import Normal
from torch import nn
from torch.nn import functional as F
from .inference_decoder import InferenceDecoder
from .inference_encoder import InferenceEncoder


class Tacotron3(nn.Module):
    """

    """
    def __init__(self, experiment_specs: ExperimentSpecs, device) -> None:
        """

        :param experiment_specs:
        :param device:
        """
        super(Tacotron3, self).__init__()

        # model specs
        self.experiment_specs = experiment_specs
        self.model_trainer_spec = experiment_specs
        self.model_spec = experiment_specs.get_model_spec()
        self.specto_spec = self.model_spec.get_spectrogram()
        self.device = device

        self.mask_padding = self.experiment_specs.mask_padding
        self.fp16_run = self.experiment_specs.is_amp()
        self.n_mel_channels = self.specto_spec.n_mel_channels()

        #
        self.embedding = nn.Embedding(self.experiment_specs.n_symbols,
                                      self.specto_spec.symbols_embedding_dim())
        std = sqrt(2.0 / (self.experiment_specs.n_symbols +
                          self.specto_spec.symbols_embedding_dim()))
        val = sqrt(3.0) * std  # uniform bounds for std

        self.embedding.weight.data.uniform_(-val, val)

        # self.linear_mu = nn.Linear(self.layer_dim, 512)
        # self.linear_var = nn.Linear(self.layer_dim, 512)

        self.encoder = Encoder(experiment_specs, device=self.device)
        self.decoder = Decoder(experiment_specs, device=self.device)
        self.postnet = Postnet(experiment_specs, device=self.device)

        if self.specto_spec.is_vae_enabled():
            self.vae_encode = InferenceEncoder(z_dim=1024)
            self.vae_decode = InferenceDecoder(z_dim=1024)

        self.parallel_decoder = True
        self.reverse_decoder = None

    def reparameterize(self, mu, logvar, mi=False):
        """
        Used for VAE re parameterization trick.
        :param mu:
        :param logvar:
        :param mi:
        :return:
        """
        if not self.training and not mi:
            return mu
        elif not self.training and mi:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu, mu + eps * std
        elif self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

    def parse_batch(self, batch):
        """

        :param batch:
        :return:
        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spectral = batch
        text_padded = to_gpu(text_padded).long()
        spectral = to_gpu(spectral).float()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (text_padded, input_lengths, mel_padded, max_len, output_lengths, spectral), \
               (mel_padded, gate_padded, spectral)

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

    def backward_decoder(self):
        """
        :return:
        """
        if self.reverse_decoder is None:
            self.reverse_decoder = copy.deepcopy(self.decoder)

    # def sequence_mask(sequence_length, max_len=None):
    #     if max_len is None:
    #         max_len = sequence_length.data.max()
    #     seq_range = torch.arange(max_len, dtype=sequence_length.dtype,
    #                              device=sequence_length.device)
    #     # B x T_max
    #     return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)

    def get_mask_from_lengths(self, lengths, max_l=None, device="cuda"):
        """
        :param lengths:
        :param device:
        :return:
        """
        max_len = torch.max(lengths).item()
        if max_l is None:
            max_len = max_l

        # max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, out=torch.LongTensor(max_len), device=device)
        mask = (ids < lengths.unsqueeze(1)).bool(device=device).to(device)
        return mask

    def compute_masks(self, text_lengths, mel_lengths):

        device = text_lengths.device
        input_mask = get_mask_from_lengths(text_lengths).to(device)
        output_mask = None
        if mel_lengths is not None:
            max_len = mel_lengths.max()
            r = self.decoder.r
            max_len = max_len + (r - (max_len % r)) if max_len % r > 0 else max_len
            output_mask = get_mask_from_lengths(mel_lengths, max_len=max_len).to(device)

        return input_mask

    def backward_pass(self, encoder_outputs, mels, text_lengths):
        """
        Reverse decoder pass.
        :param mels:
        :param encoder_outputs:
        :param text_lengths:
        :return:
        """
        mel_flip = torch.flip(mels, dims=(1,))
        mel_outputs, gate_outputs, alignments = self.decoder(
                encoder_outputs, mel_flip, memory_lengths=text_lengths)

        # gate_outputs = gate_outputs.transpose(1, 2).contiguous()
        return mel_outputs, gate_outputs, alignments

    def coarse_decoder_pass(self, mel_specs, encoder_outputs, alignments, input_mask):
        """
        :param mel_specs:
        :param encoder_outputs:
        :param alignments:
        :param input_mask:
        :return:
        """
        T = mel_specs.shape[1]
        if T % self.coarse_decoder.r > 0:
            padding_size = self.coarse_decoder.r - (T % self.coarse_decoder.r)
            mel_specs = F.pad(mel_specs, (0, 0, 0, padding_size, 0, 0))
        decoder_outputs_backward, alignments_backward, _ = self.coarse_decoder(encoder_outputs.detach(), mel_specs, input_mask)

        # scale_factor = self.decoder.r_init / self.decoder.r
        alignments_backward = F.interpolate(
                alignments_backward.transpose(1, 2),
                size=alignments.shape[1],
                mode='nearest').transpose(1, 2)

        decoder_outputs_backward = decoder_outputs_backward.transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward[:, :T, :]

        return decoder_outputs_backward, alignments_backward

    def compute_masks(self, text_lengths, mel_lengths):
        """

        :param text_lengths:
        :param mel_lengths:
        :return:
        """
        # B x T_in_max (boolean)
        device = text_lengths.device
        input_mask = get_mask_from_lengths(text_lengths).to(device)
        output_mask = None

        if mel_lengths is not None:
            max_len = mel_lengths.max()
            r = self.decoder.r
            max_len = max_len + (r - (max_len % r)) if max_len % r > 0 else max_len
            output_mask = get_mask_from_lengths(mel_lengths, max_len=max_len).to(device)

        return input_mask

    def forward(self, inputs, is_reversed=True):
        """
        Forward pass inputs a batch that contains text, mel , spectral data.
        :param is_reversed:  We can iterate between during training between normal and dual decoder.
        :param inputs:
        :return:
        """

        if is_reversed:
            self.backward_decoder()

        text_inputs, text_lengths, mels, max_len, output_len = inputs
        text_lengths, output_len = text_lengths.data, output_len.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        enc_out = self.encoder(embedded_inputs, text_lengths)

        mel_out, gate_out, alignments = self.decoder(enc_out, mels, memory_lengths=text_lengths)

        # VAE pass that didn't produce better result
        # q_mean, q_var = self.vae_encode(spectral)
        # q_dist = Normal(q_mean, q_var)
        # z_sample = q_dist.rsample()
        # vae_decoder_out = self.vae_decode(z_sample)

        mel_outputs_postnet = self.postnet(mel_out)
        mel_outputs_postnet = mel_out + mel_outputs_postnet

        if self.parallel_decoder and is_reversed:
            # mel_out_rev, gate_outputs_rev, alignments_rev = self.backward_pass(encoder_outputs, mels, text_lengths)
            rev = self.backward_pass(enc_out, mels, text_lengths)
            return self.parse_output([mel_out,
                                      mel_outputs_postnet,
                                      gate_out,
                                      alignments,
                                      rev
                                      ],
                                     output_len)

        return self.parse_output(
                [mel_out, mel_outputs_postnet, gate_out, alignments], output_len)

    def inference(self, inputs):
        """
        During inference pass input to embedding layer
        transpose shape (batch_size, x ,y ) (batc_size, y, z)
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
