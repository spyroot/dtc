from .call_interface import Callback
from .time_meter import TimeMeter
from loguru import logger


# log to tensorboard
class BatchTimer(Callback):
    """

    """

    def __init__(self):
        super().__init__()
        self.has_printed = False
        self.timer = TimeMeter()

    def on_batch_begin(self):
        self.timer.batch_start()

    def on_batch_end(self):
        self.timer.batch_end()

    def on_loader_begin(self):
        self.timer.reset()

    def on_loader_end(self):
        if not self.has_printed:
            self.has_printed = True
            d_time = self.timer.data_time.avg_smooth
            b_time = self.timer.batch_time.avg_smooth
            logger.info(f"TimeMeter profiling. Data time: {d_time:.2E}s. Model time: {b_time:.2E}s \n")
