from collections import defaultdict
from .call_interface import Callback
from loguru import logger


class ResetOptimizer(Callback):
    """Set's Optimizers state to empty for epoch in `reset_epoch`. Could be used for restarts.
    Args:
        reset_epoch (List[int]): after which epochs to reset optimizer
        verbose (bool): Flag to print that optimizer was reset."""

    def __init__(self, reset_epochs=None, verbose=True):
        super().__init__()
        if reset_epochs is None:
            reset_epochs = []
        self.reset_epochs = set(reset_epochs)
        self.verbose = verbose

    def on_epoch_end(self):
        if self.trainer.state.epoch_log in self.reset_epochs:
            # any optimizer inherited from torch.Optimizer has state which can be reset
            if hasattr(self.trainer.state.optimizer, "optimizer"):  # for lookahead
                self.trainer.state.current_optimizer.state = defaultdict(dict)
            else:
                self.trainer.state.current_optimizer.state = defaultdict(dict)

            if self.verbose:
                logger.info("Reseting optimizer")
