class CheckpointSaver(Callback):
    """
    Save best model every epoch based on loss
    Args:
        save_dir (str): path to folder where to save the model
        save_name (str): name of the saved model. can additionally
            add epoch and metric to model save name
        monitor (str): quantity to monitor. Implicitly prefers validation metrics over train. One of:
            `loss` or name of any metric passed to the runner.
        mode (str): one of "min" of "max". Whether to decide to save based
            on minimizing or maximizing loss
        include_optimizer (bool): if True would also save `optimizers` state_dict.
            This increases checkpoint size 2x times.
        verbose (bool): If `True` reports each time new best is found
    """

    def __init__(
        self,
        save_dir,
        save_name="model_{ep}_{metric:.2f}.chpn",
        monitor="loss",
        mode="min",
        include_optimizer=False,
        verbose=True,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.monitor = monitor
        mode = ReduceMode(mode)
        if mode == ReduceMode.MIN:
            self.best = np.inf
            self.monitor_op = np.less
        elif mode == ReduceMode.MAX:
            self.best = -np.inf
            self.monitor_op = np.greater
        self.include_optimizer = include_optimizer
        self.verbose = verbose

    def on_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current, self.best):
            ep = self.state.epoch_log
            if self.verbose:
                logger.info(f"Epoch {ep:2d}: best {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            self.best = current
            save_name = os.path.join(self.save_dir, self.save_name.format(ep=ep, metric=current))
            self._save_checkpoint(save_name)

    def _save_checkpoint(self, path):
        if hasattr(self.state.model, "module"):  # used for saving DDP models
            state_dict = self.state.model.module.state_dict()
        else:
            state_dict = self.state.model.state_dict()
        save_dict = {"epoch": self.state.epoch, "state_dict": state_dict}
        if self.include_optimizer:
            save_dict["optimizer"] = self.state.optimizer.state_dict()
        torch.save(save_dict, path)

    def get_monitor_value(self):
        value = None
        if self.monitor == "loss":
            value = self.state.loss_meter.avg
        else:
            for name, metric_meter in self.state.metric_meters.items():
                if name == self.monitor:
                    value = metric_meter.avg
        if value is None:
            raise ValueError(f"CheckpointSaver can't find {self.monitor} value to monitor")
        return value