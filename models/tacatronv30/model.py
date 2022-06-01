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


class InferenceEncoder(nn.Module):
    """

    """

    def __init__(self, z_dim, y_dim=0, hidden_dim=512):
        super(InferenceEncoder, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(z_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 2 * z_dim),
        )

    def gaussian_parameters(self, h, dim=-1):
        """
        Converts generic real-valued representations into mean and variance
        parameters of a Gaussian distribution

        :param h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        :param dim: int: (): Dimension along which to split the tensor for mean and
                             variance
        :return:   m: tensor: (batch, ..., dim / 2, ...): Mean
                   v: tensor: (batch, ..., dim / 2, ...): Variance
        """
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    def forward(self, x, y=None):
        """

        :param x:
        :param y:
        :return:
        """
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = self.gaussian_parameters(h, dim=1)
        return m, v


class InferenceDecoder(nn.Module):
    def __init__(self, z_dim, y_dim=0, hidden_dim=512):
        super(InferenceDecoder, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
                nn.Linear(z_dim + y_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1024)
        )

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)


class VaeEncoder(nn.Module):
    """
    Encoder Class
    Values:
    im_chan: the number of channels of the output image, a scalar
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, im_chan=1, z_dim=1024, hidden_dim=512):
        super(VaeEncoder, self).__init__()
        self.z_dim = z_dim
        self.enc = InferenceEncoder(self.z_dim)
        self.dec = InferenceDecoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        # self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        # self.z_prior = (self.z_prior_m, self.z_prior_v)

    def forward(self, image):
        """
        Function for completing a forward pass of the Encoder: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
        image: a flattened image tensor with dimension (im_dim)
        """
        disc_pred = self.enc(image)
        # print(disc_pred.shape)
        encoding = disc_pred.view(len(disc_pred), -1)
        # The stddev output is treated as the log of the variance of the normal
        # distribution by convention and for numerical stability
        return encoding[:, :self.z_dim], \
               encoding[:, self.z_dim:].exp()


class VaeDecoder(nn.Module):
    """
    Decoder Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white, so that's our default
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, z_dim=1024, im_chan=1, hidden_dim=512):
        super(VaeDecoder, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
                self.make_gen_block(z_dim, hidden_dim * 4),
                self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
                self.make_gen_block(hidden_dim * 2, hidden_dim),
                self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a Decoder block of the VAE,
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                    nn.Sigmoid(),
            )

    def forward(self, noise):
        """
        Function for completing a forward pass of the Decoder: Given a noise vector,
        returns a generated image.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        """
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


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
        self.encoder_spec = self.model_spec.get_spectrogram()
        self.device = device

        self.mask_padding = self.experiment_specs.mask_padding
        self.fp16_run = self.experiment_specs.is_amp()
        self.n_mel_channels = self.encoder_spec.n_mel_channels()
        self.n_frames_per_step = self.experiment_specs.n_frames_per_step

        #
        self.embedding = nn.Embedding(self.experiment_specs.n_symbols,
                                      self.experiment_specs.symbols_embedding_dim)
        #
        std = sqrt(2.0 / (self.experiment_specs.n_symbols + self.experiment_specs.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std

        self.embedding.weight.data.uniform_(-val, val)

        # self.linear_mu = nn.Linear(self.layer_dim, 512)
        # self.linear_var = nn.Linear(self.layer_dim, 512)

        self.encoder = Encoder(experiment_specs, device=self.device)
        self.decoder = Decoder(experiment_specs, device=self.device)
        self.postnet = Postnet(experiment_specs, device=self.device)

        self.vae_encode = InferenceEncoder(z_dim=1024)
        self.vae_decode = InferenceDecoder(z_dim=1024)

        self.bidirectional_decoder = True
        self.reverse_decoder = None

    def reparameterize(self, mu, logvar, mi=False):
        """

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
        Reverse pass

        :param mels:
        :param encoder_outputs:
        :param text_lengths:
        :return:
        """

        print("mel ", mels.shape)
        mel_flip = torch.flip(mels, dims=(1,))
        print("mel ", mels.shape)

        # mel  torch.Size([32, 80, 825])
        # mel_outputs from backward  torch.Size([32, 80, 825])
        # gate_outputs from backward  torch.Size([32, 825])
        # alignments from backward  torch.Size([32, 825, 157])
        # Dimension out of range (expected to be in range of [-2, 1], but got 2)

        #         memory: Encoder outputs
        #         decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        #         memory_lengths: Encoder output lengths for attention masking.
        #
        #         RETURNS
        #         -------
        #         mel_outputs: mel outputs from the decoder
        #         gate_outputs: gate outputs from the decoder
        #         alignments: sequence of attention weights from the decoder

        mel_outputs, gate_outputs, alignments = self.decoder(
                encoder_outputs, mel_flip, memory_lengths=text_lengths)

        # print("mel_outputs from backward ", mel_outputs.shape)
        # print("gate_outputs from backward ", gate_outputs.shape)
        # print("alignments from backward ", alignments.shape)

        # mel_outputs, gate_outputs, alignments = self.decoder_backward(
        #         encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask,
        #         self.speaker_embeddings_projected)
        #

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
            mel_specs = torch.nn.functional.pad(mel_specs, (0, 0, 0, padding_size, 0, 0))
        decoder_outputs_backward, alignments_backward, _ = self.coarse_decoder(encoder_outputs.detach(), mel_specs,
                                                                               input_mask)

        # scale_factor = self.decoder.r_init / self.decoder.r
        alignments_backward = torch.nn.functional.interpolate(
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

    def forward(self, inputs):
        """
        Forward pass inputs a batch that contains text, mel , spectral data.
        :param inputs:
        :return:
        """

        self.backward_decoder()

        text_inputs, text_lengths, mels, max_len, output_lengths, spectral = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
                encoder_outputs, mels, memory_lengths=text_lengths)

        q_mean, q_var = self.vae_encode(spectral)

        # print("q_mean", q_mean.shape)
        # print("q_var", q_var.shape)
        # epsilon = torch.randn_like(v)
        # z = m + torch.sqrt(v) * epsilon
        #
        # S, phase = librosa.magphase(librosa.stft(y))
        q_dist = Normal(q_mean, q_var)
        # print("q_var", q_dist)
        z_sample = q_dist.rsample()
        # print("z_samep", z_sample.shape)
        vae_decoder_out = self.vae_decode(z_sample)
        # print("gate_out dim", decoding.shape)
        #  mu_ = self.linear_mu(embedded_inputs)
        # alignments
        # logvar_ = self.linear_var(output)
        # mu_ = self.linear_mu(output)
        # logvar_ = self.linear_var(output)
        #
        # if self.z_mode == 'normal':
        #     mu = mu_[-1]
        #     logvar = logvar_[-1]
        #     z = self.reparameterize(mu, logvar)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        #    memory: Encoder outputs
        #         decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        #         memory_lengths: Encoder output lengths for attention masking.
        #
        #         RETURNS
        #         -------
        #         mel_outputs: mel outputs from the decoder
        #         gate_outputs: gate outputs from the decoder
        #         alignments: sequence of attention weights from the decoder

        if self.bidirectional_decoder:
            mel_out_rev, gate_outputs_rev, alignments_rev = self.backward_pass(encoder_outputs, mels, text_lengths)
            return self.parse_output([mel_outputs,
                                      mel_outputs_postnet,
                                      gate_outputs,
                                      alignments,
                                      vae_decoder_out,
                                      q_dist,
                                      mel_out_rev,
                                      gate_outputs_rev,
                                      alignments_rev,
                                      ],
                                     output_lengths)

        return self.parse_output(
                [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, vae_decoder_out, q_dist],
                output_lengths)

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
