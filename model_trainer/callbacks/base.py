import collections
from typing import Any


class Callback(object):
    def __init__(self, *args, **kwargs):
        pass

    def set_state(self, state):
        pass

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
        """Called after ``loss.backward()`` but before optimizer does anything."""
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
        # self.state

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

