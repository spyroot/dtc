import time
from .average_meter import AverageMeter


class TimeMeter:
    def __init__(self):
        self.start = None
        self.data_time = None
        self.batch_time = None
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
