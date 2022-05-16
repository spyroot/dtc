import torch
# server side
torch.distributed.init_process_group(backend="nccl", init_method="tcp://192.168.254.230:54322", world_size=2, rank=1)
