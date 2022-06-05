from torch.cuda import amp

from .call_interface import listify, Callback
from model_trainer.internal.average_meter import AverageMeter
import torch

from model_trainer.internal.utils import to_numpy


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
            self.trainer.state.metric_meters[name] = AverageMeter(name=name)

    @torch.no_grad()
    def on_batch_end(self):
        _, target = self.trainer.state.input
        output = self.trainer.state.output
        #
        with amp.autocast(self.trainer.state.is_amp):
            for metric, name in zip(self.metrics, self.metric_names):
                self.trainer.state.metric_meters[name].update(to_numpy(metric(output, target).squeeze()))
