import torch.nn as nn


class Reshaper(nn.Module):
    """
    reshaper as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshaper, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        return 'shape={}'.format(self.shape)