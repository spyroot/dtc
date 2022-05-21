import os

import torch
from torch.nn.modules import Module

from distributed_wrapper import DistributedDataWrapper


class LinearNorm(torch.nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
                self.linear_layer.weight,
                gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.linear_layer(x)


def init_distributed():
    """

    :return:
    """
    #  if self.rank != 0:
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "54321"
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
    # Set cuda device so everything is done on the right GPU.
    # Initialize distributed communication

    torch.distributed.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:54321",
            world_size=2,
            rank=0)


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_distributed()
torch.cuda.set_device(0 % torch.cuda.device_count())
ll = LinearNorm(in_dim=128, out_dim=128).to(_device)
model = DistributedDataWrapper(ll, device_ids=[0], output_device=0)
model.train()