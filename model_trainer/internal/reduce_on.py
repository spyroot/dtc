import numpy as np

from .call_interface import Callback
from .save_best import ReduceMode
from .time_meter import TimeMeter
from loguru import logger


class ReduceLROnPlateau(Callback):
    """
    Reduces learning rate by `factor` after `patience`epochs without improving
    Args:
        factor (float): by how to reduce learning rate
        patience (int): how many epochs to wait until reducing lr
        min_lr (float): minimum learning rate which could be achieved
        monitor (str): quantity to monitor. Implicitly prefers validation metrics over train. One of:
            `loss` or name of any metric passed to the runner.
        mode (str): one of "min" of "max". Whether to decide reducing based
            on minimizing or maximizing loss
        vebose (bool): Whether or not to print messages about updating lr to console
    """

    def __init__(self, factor=0.5, patience=5, min_lr=1e-6, monitor="loss", mode="min", verbose=True):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.monitor = monitor
        self.min_lr = min_lr
        mode = ReduceMode(mode)
        self.best = np.inf if mode == ReduceMode.MIN else -np.inf
        self.monitor_op = np.less if mode == ReduceMode.MIN else np.greater
        self._steps_since_best = 0
        self.verbose = verbose

    def on_epoch_end(self):
        current = self.get_monitor_value()
        self._steps_since_best += 1
        if self.monitor_op(current, self.best):
            self._steps_since_best = 0
        elif self._steps_since_best > self.patience:
            for param_group in self.state.optimizer.param_groups:
                if param_group["lr"] * self.factor > self.min_lr:
                    param_group["lr"] *= self.factor
            if self.verbose:
                logger.info(f"ReduceLROnPlateau reducing learning rate to {param_group['lr'] * self.factor}")

    def get_monitor_value(self):
        value = None
        if self.monitor == "loss":
            value = self.state.loss_meter.avg
        else:
            for name, metric_meter in self.state.metric_meters.items():
                if name == self.monitor:
                    value = metric_meter.avg
        if value is None:
            raise ValueError(f"ReduceLROnPlateau can't find {self.monitor} value to monitor")
        return value
