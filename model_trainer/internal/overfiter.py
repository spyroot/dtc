import torch

from .call_interface import Callback


class BatchOverfit(Callback):
    """Remembers first batch and tries to overfit it. Useful for debug.
    NOTE: Should go after all other callbacks to make sure it's the last thing to change the input
    Args:
        save_batch (bool): If True will save first batch. Useful for visualization"""

    def __init__(self, save_batch=False):
        super().__init__()
        self.has_saved = False
        self.save_batch = save_batch
        self.batch = None

    def on_batch_begin(self):
        if not self.has_saved:
            self.has_saved = True
            self.batch = self.state.input[0].clone(), self.state.input[1].clone()
            if self.save_batch:
                torch.save(self.batch[0], "b_img")
                torch.save(self.batch[1], "b_target")
        else:
            self.state.input = self.batch