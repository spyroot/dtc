import torch
from torch import nn
from torch.nn import functional as F

from .layers import LinearNorm
from .location import LocationLayer


class Attention(nn.Module):
    """
    Attention layer used by decoder to keep attention
    """

    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size, is_strict=True) -> None:
        super(Attention, self).__init__()

        if is_strict:
            assert attention_rnn_dim == 1024
            assert attention_dim == 128
            assert attention_location_n_filters == 32
            assert attention_location_kernel_size == 31
            assert embedding_dim == 512

        self.query_layer = LinearNorm(attention_rnn_dim,
                                      attention_dim,
                                      bias=False,
                                      w_init_gain='tanh')

        self.memory_layer = LinearNorm(embedding_dim,
                                       attention_dim,
                                       bias=False,
                                       w_init_gain='tanh')

        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        #
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        pquery = self.query_layer(query.unsqueeze(1))
        p_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(pquery + p_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        :param attention_state:  attention rnn last output
        :param memory: encoder memory
        :param processed_memory: processed encoder output
        :param attention_weights_cat: previous and cummulative attention weights
        :param mask:  binary mask for padded data
        :return:
        """
        alignment = self.get_alignment_energies(attention_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
