from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel


class DistributedDataWrapper(DistributedDataParallel):

    def __init__(self, module: Module, device_ids=None, output_device=None):
        super(DistributedDataWrapper, self).__init__(module=module, device_ids=device_ids, output_device=output_device)
        self.device_ids = device_ids
        self.output_device = output_device
        self._module = module

        # for p in self.module.state_dict().values():
        #     if not torch.is_tensor(p):
        #         continue
        #     dist.broadcast(p, 0)

    def train(self, mode: bool = True):
        return self._module.train()

    #     def allreduce_params():
    #         if self.needs_reduction:
    #             self.needs_reduction = False
    #             buckets = {}
    #             for param in self.module.parameters():
    #                 if param.requires_grad and param.grad is not None:
    #                     tp = type(param.data)
    #                     if tp not in buckets:
    #                         buckets[tp] = []
    #                     buckets[tp].append(param)
    #             if self.warn_on_half:
    #                 if torch.cuda.HalfTensor in buckets:
    #                     print("WARNING: gloo dist backend for half parameters may be extremely slow." +
    #                           " It is recommended to use the NCCL backend in this case. This currently requires" +
    #                           "PyTorch built from top of tree master.")
    #                     self.warn_on_half = False
    #
    #             for tp in buckets:
    #                 bucket = buckets[tp]
    #                 grads = [param.grad.data for param in bucket]
    #                 coalesced = _flatten_dense_tensors(grads)
    #                 dist.all_reduce(coalesced)
    #                 coalesced /= dist.get_world_size()
    #                 for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
    #                     buf.copy_(synced)
    #
    #     for param in list(self.module.parameters()):
    #         def allreduce_hook(*unused):
    #             param._execution_engine.queue_callback(allreduce_params)
    #
    #         if param.requires_grad:
    #             param.register_hook(allreduce_hook)
    #
    # def forward(self, *inputs, **kwargs):
    #     self.needs_reduction = True
    #     return self.module(*inputs, **kwargs)
