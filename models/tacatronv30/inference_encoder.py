import torch
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