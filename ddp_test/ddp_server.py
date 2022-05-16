import torch
torch.distributed.init_process_group(backend="nccl", init_method="tcp://localhost:54321", world_size=1, rank=0)
