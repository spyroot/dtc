import torch
from torch import nn


class InferenceDecoder(nn.Module):
    """

    """
    def __init__(self, z_dim, y_dim=0, hidden_dim=512):
        super(InferenceDecoder, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
                nn.Linear(z_dim + y_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, self.z_dim)
        )

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)