import torch
# server side
torch.distributed.init_process_group(backend="nccl", init_method="tcp://192.168.254.230:54321", world_size=1, rank=1)
