import torch
from torch import nn


class LinearWithInitial(torch.nn.Module):
    """
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='tanh'):
        """
        :param in_dim:
        :param out_dim:
        :param bias:
        :param w_init_gain:
        """
        super(LinearWithInitial, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight,
                                      gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.linear_layer(x)


class LocationLayer(nn.Module):
    """Layers for Location Sensitive Attention
    Args:
        attention_dim (int): number of channels in the input tensor.
        attention_n_filters (int, optional): number of filters in convolution. Defaults to 32.
        attention_kernel_size (int, optional): kernel size of convolution filter. Defaults to 31.
    """

    def __init__(self,
                 attention_dim,
                 attention_n_filters=32,
                 attention_kernel_size=31):
        super(LocationLayer, self).__init__()

        # location
        self.location_conv1d = nn.Conv1d(
                in_channels=2,
                out_channels=attention_n_filters,
                kernel_size=attention_kernel_size,
                stride=1,
                padding=(attention_kernel_size - 1) // 2,
                bias=False)

        self.location_dense = LinearWithInitial(
                attention_n_filters, attention_dim, bias=False, w_init_gain='tanh')

    def forward(self, attention_cat):
        """
        Shapes:
            attention_cat: [B, 2, C]
        """
        processed_attention = self.location_conv1d(attention_cat)
        processed_attention = self.location_dense(processed_attention.transpose(1, 2))
        return processed_attention
