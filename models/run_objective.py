from typing import Optional, Callable

import numpy as np
import torch
# from torch.nn.functional import melscale_fbanks
import torchaudio.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as F


def compute_grad(optimizer, x: Tensor,
                 y: Tensor, objective: Callable,
                 batch_size: Optional[int] = 128):
    """
     Computes objective and gradient of neural network over data sample.
    Inputs:
        optimizer (Optimizer): the PBQN optimizer
        X_Sk (nparray): set of training examples over sample Sk
        y_Sk (nparray): set of training labels over sample Sk

        opfun (callable):  forward pass callback
        batch (int): maximum size of effective batch (default: 128)

    Outputs:
        grad (tensor): stochastic gradient over sample Sk
        obj (tensor): stochastic function value over sample Sk
    """

    if torch.cuda.is_available():
        obj = torch.tensor(0, dtype=torch.float).cuda()
    else:
        obj = torch.tensor(0, dtype=torch.float)

    x_size = x.shape[0]
    optimizer.zero_grad()

    # loop through relevant data
    for idx in np.array_split(np.arange(x_size), max(int(x_size / batch_size), 1)):
        # define ops
        ops = objective(x[idx])

        # define targets
        if torch.cuda.is_available():
            tgts = Variable(torch.from_numpy(y[idx]).cuda().long().squeeze())
        else:
            tgts = Variable(torch.from_numpy(y[idx]).long().squeeze())

        # define loss and perform forward-backward pass
        loss = F.cross_entropy(ops, tgts)*(len(idx)/x_size)
        loss.backward()
        obj += loss

    # gather flat gradient
    grad = optimizer._gather_flat_grad()
    return grad, obj