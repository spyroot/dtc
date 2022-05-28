# import torch.distributed as dist
# import queue
from abc import ABC
# from multiprocessing import Queue
from collections import deque
from typing import Optional

import torch
from loguru import logger
from yaml import warnings

from model_loader.stft_dataloader import SFTFDataloader
from model_trainer.base_trainer import TrainerError, AbstractTrainer
from model_trainer.callbacks.base import BaseCallbacks, Callback
# from model_trainer.trainer_logger import TensorboardTrainerLogger
from model_trainer.trainer_metrics import Metrics
from model_trainer.trainer_specs import ExperimentSpecs
# from distributed import apply_gradient_allreduce
from models.loss_function import Tacotron2Loss


# from frozendict import frozendict


@AbstractTrainer.register
class TrainerState(AbstractTrainer, ABC):
    """

    """

    # root: Optional[str] = "dtc",
    def __init__(self,
                 trainer_spec: ExperimentSpecs,
                 data_loader: Optional[SFTFDataloader] = None,
                 verbose: Optional[bool] = False,
                 is_notebook: Optional[bool] = False,
                 rank: Optional[int] = 0,
                 world_size: Optional[int] = 2,
                 disable_pbar: Optional[int] = False,
                 device: Optional[int] = torch.device,
                 cuda_device_id: Optional[int] = 0,
                 is_inference: Optional[bool] = False,
                 callback: Optional[list[Callback]] = None,
                 hp_tunner=False,
                 config=None,
                 checkpoint_dir=None) -> None:
        """

        :param trainer_spec:  a trainer spec , trainer uses to train model
        :param data_loader:   a data loader
        :param verbose:       enabled verbose output
        :param is_notebook:   if trainer run it notebook mode. ( tqdm and pbar need to re-adjusted)
        :param rank:          node rank
        :param world_size:    world_size
        :param disable_pbar:  disabled pbar
        :param device:        device we run
        :param is_inference:  inference mode or not
        """
        super(TrainerState, self).__init__(trainer_spec=trainer_spec,
                                           data_loader=data_loader,
                                           verbose=verbose,
                                           is_notebook=is_notebook,
                                           rank=rank,
                                           world_size=world_size,
                                           disable_pbar=disable_pbar,
                                           device=device,
                                           cuda_device_id=cuda_device_id,
                                           is_inference=is_inference)


        self.is_hp_tunner = hp_tunner
        self.set_logger(verbose)
        self.device = device

        # dict that hold all schedulers that trainer need to use. TODO This will be moved.
        self._schedulers = {}
        # dict hold all optimizers. ( note trainer can train 2 model like in gan settings)
        self._optimizers = {}

        if not is_inference:
            if not trainer_spec.is_initialized():
                raise TrainerError("you need initialize trainer specs first.")

        # if self.trainer_spec.is_distributed_run():
        #     self.init_distributed()

        self._tkey = self.trainer_spec.get_default_train_set_key()
        self._vkey = self.trainer_spec.get_default_val_set_key()

        if not self.is_inference:
            if data_loader is None:
                raise TrainerError("Trainer need torch data loader.")
            self._data_loaders, self.collate_fn = data_loader.get_all()
            self._train_loader = self._data_loaders[self._tkey]
            self._validation_loader = self._data_loaders[self._vkey]

            if len(self._train_loader.dataset) == 0:
                warnings.warn("Training dataset empty")
            if len(self._validation_loader.dataset) == 0:
                warnings.warn("Training dataset empty")

        # TODO need refactor that, and move to dict and abstract,
        self.criterion = Tacotron2Loss()
        # dict store all model
        self._models = {}
        # store last epoch
        self._last_ckt_epochs: dict[str, dict[str, int]] = {}
        self._steps: dict[str, int] = {}   # dict holds model name = last iterator value
        self.scaler = None
        self.tqdm_iter = None               # tqdm_iter, if we need fix post of iter
        self.epoch = None                   # current epoch trainer executing.
        self.saved_run = None               # last saved run

        self.clip_grad = False
        self.total_batches = 0
        if self.is_inference is False:
            # total batches
            self.total_batches = len(self._data_loaders[self._tkey])
            # clip or not grad
            self.clip_grad = trainer_spec.is_grad_clipped()

            # self.tf_logger = TensorboardTrainerLogger(trainer_spec.tensorboard_update_rate())
            self.metric = Metrics(metric_step_file_path=trainer_spec.model_files.get_metric_file_path(),
                                  metric_batch_file_path=trainer_spec.model_files.get_metric_batch_file_path(),
                                  metric_perf_trace_path=trainer_spec.model_files.get_time_file_path(),
                                  num_epochs=self.trainer_spec.epochs(),
                                  num_batches=self.total_batches,
                                  batch_size=self.trainer_spec.batch_size(),
                                  verbose=False)

        self.model_creator, self.trainer_dispatcher, self._batch_loader = self.create_model_dispatch()
