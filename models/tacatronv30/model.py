from math import sqrt

import torch
from torch import nn
from torch import Tensor

from model_trainer.trainer_specs import ExperimentSpecs
from tacotron2.utils import to_gpu, get_mask_from_lengths
from .preandpost import Postnet
from .decoder import Decoder
from .encoder import Encoder
from torch.distributions.normal import Normal


class VaeEncoder(nn.Module):
    """
    Encoder Class
    Values:
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white (1 channel), so that's our default.
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, im_chan=1, output_chan=32, hidden_dim=16):
        super(VaeEncoder, self).__init__()
        self.z_dim = output_chan
        self.disc = nn.Sequential(
                self.make_disc_block(im_chan, hidden_dim),
                self.make_disc_block(hidden_dim, hidden_dim * 2),
                self.make_disc_block(hidden_dim * 2, output_chan * 2, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a encoder block of the VAE,
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                    nn.BatchNorm2d(output_channels),
                    nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        """
        Function for completing a forward pass of the Encoder: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
        image: a flattened image tensor with dimension (im_dim)
        """
        disc_pred = self.disc(image)
        encoding = disc_pred.view(len(disc_pred), -1)
        # The stddev output is treated as the log of the variance of the normal
        # distribution by convention and for numerical stability
        return encoding[:, :self.z_dim], encoding[:, self.z_dim:].exp()


class VaeDecoder(nn.Module):
    """
    Decoder Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white, so that's our default
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
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

    def __init__(self, experiment_specs: ExperimentSpecs, device):
        """

        :param experiment_specs:
        :param device:
        """
        super(Tacotron3, self).__init__()
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

        self.vae_encode = VaeEncoder(output_chan=512)
        self.vae_decode = VaeDecoder(z_dim=512)

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

        q_mean, q_stddev = self.vae_encode(gate_outputs)
        q_dist = Normal(q_mean, q_stddev)
        z_sample = q_dist.rsample()
        decoding = self.vae_decode(z_sample)

        # q_mean, q_stddev = self.encode(embedded_inputs)
        # q_dist = Normal(q_mean, q_stddev)
        # z_sample = q_dist.rsample() # Sample once from each distribution, using the `rsample` notation
        #
        # mu_ = self.linear_mu(embedded_inputs)
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

        return self.parse_output(
                [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
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
