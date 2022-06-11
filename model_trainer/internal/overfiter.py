import torch
from .call_interface import Callback


class BatchOverfiter(Callback):
    """

    """
    def __init__(self, save=False):
        """
        First batch and tries to overfit it.
        Last callback in seq.

        :param save:
        """
        super().__init__()
        self.has_saved = False
        self.save_batch = save
        self.batch = None

    def on_batch_begin(self):
        """
        :return:
        """
        if not self.has_saved:
            self.has_saved = True
            self.batch = self.trainer.state.input[0].clone(), \
                         self.trainer.state.input[1].clone()

            if self.save_batch:
                self.trainer.save(self.batch[0], "b_img")
                self.trainer.save(self.batch[1], "b_target")
        else:
            self.trainer.state.input = self.batch
