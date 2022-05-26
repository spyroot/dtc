import os
import time
import torch
import collections
import numpy as np
import torch.distributed as dist
from typing import Any


class AverageMeter:
    """
    Computes and stores the average and current value.

        Attributes:
        val - last value
        avg - true average
        avg_smooth - smoothed average
    """

    def __init__(self, name="Meter", avg_mom=0.9):
        self.avg_mom = avg_mom
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.avg_smooth = 0
        self.count = 0

    def update(self, val):
        self.val = val
        if self.count == 0:
            self.avg_smooth = val
        else:
            self.avg_smooth *= self.avg_mom
            self.avg_smooth += val * (1 - self.avg_mom)
        self.count += 1
        self.avg *= (self.count - 1) / self.count
        self.avg += val / self.count

    def __call__(self, val):
        return self.update(val)

    def __repr__(self):
        return f"AverageMeter(name={self.name}, avg={self.avg:.3f}, count={self.count})"


class TimeMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.start = time.time()

    def batch_start(self):
        self.data_time.update(time.time() - self.start)

    def batch_end(self):
        self.batch_time.update(time.time() - self.start)
        self.start = time.time()


# def env_world_size() -> int:
#     return int(os.environ.get("WORLD_SIZE", 1))

#
# def reduce_meter(meter: AverageMeter) -> AverageMeter:
#     """Args: meter (AverageMeter): meter to reduce"""
#     if env_world_size() == 1:
#         return meter
#     # can't reduce AverageMeter so need to reduce every attribute separately
#     reduce_attributes = ["val", "avg", "avg_smooth", "count"]
#     for attr in reduce_attributes:
#         old_value = to_tensor([getattr(meter, attr)]).float().cuda()
#         setattr(meter, attr, reduce_tensor(old_value).cpu().numpy()[0])
