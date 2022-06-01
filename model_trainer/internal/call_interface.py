import collections
from typing import Any, Optional

from model_trainer.internal.abstract_trainer import AbstractTrainer
from model_trainer.trainer_metrics import Metrics


class Callback(object):
    """
    Metric Concrete type , late maybe to abstract
    """
    def __init__(self, *args, **kwargs):
        self.metric: Metrics = Optional[None]
        self.trainer: AbstractTrainer = Optional[None]

    def update_trainer(self, trainer):
        print("Registed trainer")
        self.trainer = trainer

    def update_metric(self, metric):
        print("Registed metric")
        self.metric = metric

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_loader_begin(self):
        pass

    def on_loader_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_begin(self):
        pass

    def on_end(self):
        pass

    def saved(self):
        pass

    def on_after_backward(self):
        pass

    def validation_start(self):
        pass

    def validation_end(self):
        pass


class BaseCallbacks(Callback):
    """

    """

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = listify(callbacks)

    def set_state(self, state):
        for callback in self.callbacks:
            callback.set_state(state)

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_loader_begin(self):
        for callback in self.callbacks:
            callback.on_loader_begin()

    def on_loader_end(self):
        for callback in self.callbacks:
            callback.on_loader_end()

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_after_backward(self):
        for callback in self.callbacks:
            callback.on_after_backward()

    def saved(self):
        for callback in self.callbacks:
            callback.saved()

    def validation_start(self):
        for callback in self.callbacks:
            callback.validation_start()
        pass

    def validation_end(self):
        for callback in self.callbacks:
            callback.validation_end()
        pass

    def saving_start(self):
        pass


def listify(p: Any) -> collections.Iterable:
    if p is None:
        p = []
    elif not isinstance(p, collections.Iterable):
        p = [p]
    return p
