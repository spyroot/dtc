from torch.cuda import amp

from model_trainer.callbacks.base import listify
from model_trainer.callbacks.callback import Callback
from model_trainer.callbacks.time_meter import AverageMeter
import torch

from model_trainer.callbacks.utils import to_numpy


class BatchMetrics(Callback):
    """
    Computes metrics values after each batch
    Args:
        metrics (List): Metrics to measure during training. All metrics
            must have `name` attribute.
    """

    def __init__(self, metrics):
        super().__init__()
        self.metrics = listify(metrics)
        self.metric_names = [m.name for m in self.metrics]

    def on_begin(self):
        for name in self.metric_names:
            self.state.metric_meters[name] = AverageMeter(name=name)

    @torch.no_grad()
    def on_batch_end(self):
        _, target = self.state.input
        output = self.state.output
        with amp.autocast(self.state.use_fp16):
            for metric, name in zip(self.metrics, self.metric_names):
                self.state.metric_meters[name].update(to_numpy(metric(output, target).squeeze()))
