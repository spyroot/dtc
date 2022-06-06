import torch.nn as nn


class Reshaper(nn.Module):
    """
    Reshaper as a layer in a sequential model.
    """

    def __init__(self, shape=None):
        super(Reshaper, self).__init__()
        if shape is None:
            shape = []
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        return 'shape={}'.format(self.shape)
