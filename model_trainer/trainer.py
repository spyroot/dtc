import os
import random
import socket
import time
from abc import ABC
from typing import Callable, Optional

from models.tacatronv30.model import Tacotron3
from models.tacotronv25.model import Tacotron25
from tacotron2.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy

import numpy as np
import torch
# import torch.distributed as dist
import math

from loguru import logger
import torch.distributed as dist
from torch import Tensor
import dill
from pathlib import Path
from torch import nn

from model_trainer.distributed_wrapper import DistributedDataWrapper
from model_trainer.trainer_metrics import Metrics
from model_trainer.trainer_logger import TensorboardTrainerLogger
from model_trainer.trainer_specs import ExperimentSpecs
# from distributed import apply_gradient_allreduce
import models
from models.loss_function import Tacotron2Loss
from tacotron2.utils import fmtl_print, fmt_print, to_gpu
from numpy import finfo
from torch.nn.utils import clip_grad_norm_
import argparse
from generator_trainer import GeneratorTrainer
from tqdm import tqdm, tnrange
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim

# from torch.nn.parallel import DistributedDataParallel
# from torch.autograd import Variable
# import numpy as np

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
except ImportError:
    logger.info("ray not found")
    pass

import matplotlib.pylab as plt

from text import sequence_to_text, text_to_sequence
import dill
from pathlib import Path
from timeit import default_timer as timer


class Trainer(GeneratorTrainer, ABC):
    """

    """
    # root: Optional[str] = "dtc",
    def __init__(self,
                 trainer_spec: ExperimentSpecs,
                 data_loader=None,
                 verbose: Optional[bool] = False,
                 is_notebook: Optional[bool] = False,
                 rank: Optional[int] = 0,
                 world_size: Optional[int] = 2,
                 disable_pbar: Optional[int] = False,
                 device: Optional[int] = torch.device,
                 cuda_device_id: Optional[int] = 0,
                 is_inference: Optional[bool] = False) -> None:
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

        super(Trainer, self).__init__(verbose=verbose, is_notebook=is_notebook)
        # cuda device id
        self.cuda_device_id = cuda_device_id
        #
        self.disable_pbar = disable_pbar
        if not is_inference:
            logger.info("Trainer created, active model {}", trainer_spec.get_active_mode())

        self.trainer_spec = trainer_spec
        # inference or training
        self.is_inference = is_inference
        #
        self.n_gpus = world_size
        #
        self.device = device
        # dict that hold all schedulers that trainer need to use
        self.schedulers = {}
        # dict hold all optimizers. ( note trainer can train 2 model like in gan settings)
        self.optimizers = {}

        if not is_inference:
            if not trainer_spec.is_initialized():
                raise Exception("you need initialize trainer specs first.")

        # if self.trainer_spec.is_distributed_run():
        #     self.init_distributed()

        if not self.is_inference:
            if data_loader is None:
                raise Exception("Trainer need torch data loader.")
            self.dataloader = data_loader
            self.train_loader, self.validation_loader, self.collate_fn = data_loader.get_loader()

        self.rank = rank
        # TODO need refactor that, and move to dict and abstract,
        self.criterion = Tacotron2Loss()
        # dict store all model
        self._models = {}
        # store last epoch
        self._last_epochs = {}
        # dict holds model name = last iterator value
        self.iters = {}
        #
        self.scaler = None
        # tqdm_iter, if we need fix post of iter
        self.tqdm_iter = None
        # current epoch trainer executing.
        self.epoch = None
        # last saved run
        self.saved_run = None

        self.clip_grad = False
        self.total_batches = 0
        if self.is_inference is False:
            # total batches
            self.total_batches = len(self.train_loader)
            # clip or not grad
            self.clip_grad = trainer_spec.is_grad_clipped()

            self.tf_logger = TensorboardTrainerLogger(trainer_spec.tensorboard_update_rate())
            self.metric = Metrics(metric_step_file_path=trainer_spec.model_files.get_metric_file_path(),
                                  metric_batch_file_path=trainer_spec.model_files.get_time_file_path(),
                                  metric_perf_trace_path=trainer_spec.model_files.get_metric_batch_file_path(),
                                  num_epochs=self.trainer_spec.epochs(), num_batches=self.total_batches,
                                  verbose=False)

        # tensorboard writer
        self.t_writer = None
        logger.info("Device ID {}".format(self.cuda_device_id))

        # late we will move this to factory
        # type class that will do creation
        self.model_creator = self.create_model_dispatch()
        # init trainer
        self.init_trainer()

    def create_model_dispatch(self) -> tuple[dict[str, Callable], dict[str, Callable]]:
        """
        Create two dispatcher,
        Model dispatcher and trainer dispatcher.
        The first dispatch model creator, where each key model name
        as it defined in config.yaml and value is callable creator function,

        method, class etc.
        Similarly, second dispatch is trainer callable objects.

        :return: model creator Callable , and trainer creator Callable.
        """
        model_dispatch = {
            'tacotron25': self.create_tacotron25,
            'dts': self.create_tacotron30,
        }
        trainer_dispatch = {
            # 'GraphGRU': RnnGenerator,
            # 'GraphLSTM': RnnGenerator,
        }
        return model_dispatch, trainer_dispatch

    def init_trainer(self) -> None:
        """
        :return:
        """
        if self.is_inference:
            return

        if self.trainer_spec.is_distributed_run():
            torch.manual_seed(self.trainer_spec.seed())
            torch.cuda.manual_seed(self.trainer_spec.seed())
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(self.trainer_spec.seed())
            random.seed(self.trainer_spec.seed())

            logger.info("Starting distributed DDP training.")
            # self.init_distributed()
            dist.barrier()

        self.create_models()
        self.create_optimizers()
        self.create_lr_schedulers()

        if self.trainer_spec.fp16_run():
            self.scaler = torch.cuda.amp.GradScaler()

    def init_distributed(self):
        """
        Initialize DDP
        :return:
        """
        os.environ['MASTER_ADDR'] = self.trainer_spec.get_master_address()
        os.environ['MASTER_PORT'] = self.trainer_spec.get_master_port()
        # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"

        assert torch.cuda.is_available(), "Distributed mode requires CUDA."
        logger.info("Distributed Available".format(torch.cuda.device_count()))
        logger.info("Distribute protocol nccl available {}".format(torch.distributed.is_nccl_available()))
        logger.info("Distribute protocol mpi available {}".format(torch.distributed.is_mpi_available()))
        logger.info("Distribute protocol glow available {}".format(torch.distributed.is_gloo_available()))
        logger.info("Distribute endpoint {} my rank {}".format(self.trainer_spec.get_backend(), self.rank))

        # Set cuda device so everything is done on the right GPU.
        # torch.cuda.set_device(self.rank % torch.cuda.device_count())
        logger.info("Set cuda device".format(self.rank % torch.cuda.device_count()))
        # Initialize distributed communication
        if self.rank == 0:
            host = socket.gethostname()
            address = socket.gethostbyname(host)
            logger.info("resolve hostname {}".format(host))
            logger.info("resolve hostname {}".format(address))

        torch.distributed.init_process_group(
                backend=self.trainer_spec.get_backend(),
                init_method=self.trainer_spec.dist_url(),
                world_size=self.n_gpus,
                rank=self.rank)

        logger.debug("Done initializing distributed {}".format(dist.get_rank()))

    def create_tacotron25(self, is_set_cuda=False):
        """
        :return:
        """
        if self.is_inference:
            model = Tacotron25(self.trainer_spec, self.device).to(self.device)
        elif self.trainer_spec.is_distributed_run():
            # device = torch.device(f"cuda:{dist.get_rank()}")
            if torch.cuda.is_available():
                n = torch.cuda.device_count() // self.n_gpus

            if is_set_cuda:
                device = f"cuda:{dist.get_rank()}"
                logger.info("Number gpu on the node {} device".format(n, device))
                torch.cuda.set_device(self.cuda_device_id)
            else:
                device = self.device

            logger.info("Creating DDP on cuda device "
                        "{} torch device {} device received {}".format(self.cuda_device_id, device, self.device))
            model = Tacotron3(self.trainer_spec, device).cuda()
            model = DistributedDataWrapper(model,
                                           device_ids=[self.cuda_device_id],
                                           output_device=self.cuda_device_id).cuda()

        else:
            model = Tacotron3(self.trainer_spec, self.device).to(self.device)

        if self.trainer_spec.is_fp16_run():
            model.decoder.attention_layer.score_mask_value = finfo('float16').min

    def create_tacotron30(self, is_set_cuda=False):
        """

        :return:
        """
        if self.is_inference:
            model = Tacotron3(self.trainer_spec, self.device).to(self.device)
        elif self.trainer_spec.is_distributed_run():
            # device = torch.device(f"cuda:{dist.get_rank()}")
            if torch.cuda.is_available():
                n = torch.cuda.device_count() // self.n_gpus

            if is_set_cuda:
                device = f"cuda:{dist.get_rank()}"
                logger.info("Number gpu on the node {} device".format(n, device))
                torch.cuda.set_device(self.cuda_device_id)
            else:
                device = self.device

            logger.info("Creating DDP on cuda device "
                        "{} torch device {} device received {}".format(self.cuda_device_id, device, self.device))
            model = Tacotron3(self.trainer_spec, device).cuda()
            model = DistributedDataWrapper(model,
                                           device_ids=[self.cuda_device_id],
                                           output_device=self.cuda_device_id).cuda()

        else:
            model = Tacotron3(self.trainer_spec, self.device).to(self.device)

        if self.trainer_spec.is_fp16_run():
            model.decoder.attention_layer.score_mask_value = finfo('float16').min

    def create_model(self, model_name, is_set_cuda=False):
        """
          Method lookup model from dispatch and call respected creator.
          Later will move to a separate dispatcher creator or maybe
          register creator.

        :param model_name:
        :return: nothing
        :param is_set_cuda: If need pin directly
        """
        if model_name in self.model_creator:
            raise ValueError("Unknown model, supported models", self.model_creator.keys())

        creator = self.model_creator[model_name]
        self._models[model_name] = creator(is_set_cuda)
        self._last_epochs[model_name] = 0

    def create_models(self):
        """
        For each active model, which might consist two or more models.
        we create a target model
        :return: nothing, create_model will add each model to a dict[model_name] = model
        """
        _models = self.trainer_spec.get_active_sub_models()
        for m in _models:
            print(m)
            self.create_model(m)

    def load(self, model_name: str, to_device=True, ignore_layers=None, model_file=None):
        """Method loads model from a checkpoint.
           It will update internal state, that include model, last epoch, last step.

        :param model_file:
        :param model_name: - mode name that method need to load
        :param ignore_layers: - a list contain of layers we skip
        :param to_device: - if true will load to device indicate as main device
        :return: epoch, step
        """
        # setting in config, if set don't load trainer won't load model
        if not self.trainer_spec.is_load_model():
            return 0

        if model_file is None:
            model_file = self.trainer_spec.model_files.get_model_file_path(model_name)
        else:
            model_file = self.trainer_spec.model_files.get_model_file_path(model_name)

        logger.info("Loading model '{}' model file name {}".format(model_name, model_file))
        # load trained optimizer state_dict
        try:
            if to_device:
                if self.trainer_spec.is_distributed_run():
                    self._models[model_name].to(self.rank)
                else:
                    self._models[model_name].to(self.device)

            if to_device:
                checkpoint = torch.load(model_file, map_location=self.device)
            else:
                checkpoint = torch.load(model_file)

            if 'model_state_dict' not in checkpoint:
                raise Exception("model has no state dict")

            self._models[model_name].load_state_dict(checkpoint['model_state_dict'])
            if 'model_state_dict' not in checkpoint:
                raise Exception("model has no state dict")

            self.optimizers[model_name].load_state_dict(checkpoint['optimizer_state_dict'])
            if 'optimizer_state_dict' not in checkpoint:
                raise Exception("model has no optimizer_state_dict")

            # if model has scheduler, re-create.
            if model_name in self.schedulers:
                self.schedulers[model_name].load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scheduler_state_dict' not in checkpoint:
                    raise Exception("model has no scheduler_state_dict")

            #  ignore layers.
            if ignore_layers is not None and len(ignore_layers) > 0:
                model_dict = {k: v for k, v in self._models[model_name].items()
                              if k not in ignore_layers}
                new_state = self._models[model_name].state_dict()
                new_state.update(model_dict)
                self._models[model_name] = new_state

            # self.trainer_spec.set_lr(0.00001)
            if 'epoch' not in checkpoint:
                raise Exception("saved checkpoint has no epoch key")
            self._last_epochs[model_name] = checkpoint['epoch']

            # load last iterator.
            if 'it' not in checkpoint:
                raise Exception("saved checkpoint has no last iteration key.")
            self.iters[model_name] = checkpoint['it']
            logger.info("Last checkpoint. epoch {} step {}".format(checkpoint['epoch'], checkpoint['it']))

            # load metric
            self.metric.load()

            return checkpoint['epoch'], checkpoint['it']
        except FileNotFoundError as e:
            print("Failed load model files {}. No saved model found.".format(model_file))
            logger.info("No model file to load model file, return default epoch 0, iteration 0")

        return 0, 0

    def load_for_inference(self, model_name, model_file=None, to_device=True, ignore_layers=None):
        """Method loads model from a checkpoint.
           It will update internal state, that include model, last epoch, last step.

        :param model_file:
        :param model_name: - mode name that method need to load
        :param ignore_layers: - a list contain of layers we skip
        :param to_device: - if true will load to device indicate as main device
        :return: epoch, step
        """

        logger.info("Loading model '{}' model file name {} to device {}".format(model_name,
                                                                                model_file,
                                                                                self.device))

        try:
            checkpoint = torch.load(model_file, map_location=self.device)
            if 'model_state_dict' not in checkpoint:
                raise Exception("model has no state dict")

            self.create_model(model_name)

            self._models[model_name].load_state_dict(checkpoint['model_state_dict'])
            if 'model_state_dict' not in checkpoint:
                raise Exception("model has no state dict")

            if ignore_layers is not None and len(ignore_layers) > 0:
                model_dict = {k: v for k, v in self._models[model_name].items()
                              if k not in ignore_layers}
                new_state = self._models[model_name].state_dict()
                new_state.update(model_dict)
                self._models[model_name] = new_state

            if 'epoch' not in checkpoint:
                raise Exception("saved checkpoint has no epoch key")
            self._last_epochs[model_name] = checkpoint['epoch']

            # load last iterator.
            if 'it' not in checkpoint:
                raise Exception("saved checkpoint has no last iteration key.")
            self.iters[model_name] = checkpoint['it']

            logger.info("Last checkpoint. epoch {} step {}".format(checkpoint['epoch'], checkpoint['it']))
            return checkpoint['epoch'], checkpoint['it']
        except FileNotFoundError as e:
            print("Failed load model files {}. No saved model found.".format(model_file))
            logger.info("No model file to load model file, return default epoch 0, iteration 0")

        return 0, 0

    def trainer_iterator(self, model_name: str, last_epoch=0, max_epochs=0):
        """

        Args:
            model_name:
            last_epoch:
            max_epochs:

        Returns:

        """
        # for notebook, we need a bit of hack for tqdm_iter.
        if self.is_notebook:
            tqdm_iter = tnrange(last_epoch, max_epochs)
            return tqdm_iter

        # load last epoch in case we do re-summing.
        last_epoch = self._last_epochs[model_name]
        # early stopping
        early_stopping = None

        max_epochs = self.trainer_spec.epochs()
        # what tqdm to use notebook for colab or normal one.
        logger.info("Creating tqdm last epoch {} max epoch", last_epoch, max_epochs)

        if self.is_notebook:
            tqdm_iter = tnrange(last_epoch, max_epochs)
        else:
            tqdm_iter = tqdm(range(last_epoch, max_epochs),
                             desc=f"Training in progress, device {self.device} total batches {self.total_batches}",
                             disable=self.disable_pbar)
            # total=self.get_last_iterator(model_name) * self.total_batches,
            # desc="training")

        # for b in trange(
        #         epochs * self., unit_scale=0.1, unit="epoch",
        #         bar_format="{l_bar}{bar}|{n:.1f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        # ):

        # tqdm_iter.set_postfix({'total_epoch_loss': 0})
        return tqdm_iter

    def load_models(self) -> None:
        """
        For each active model, which might consist of more than one model
        :return:
        """
        models = self.trainer_spec.get_active_sub_models()
        for m in models:
            self.load(m)

    def reduce_if_needed(self, loss):
        """
        TODO need proper test FP16
        :param loss:
        :return:
        """
        if self.trainer_spec.is_fp16_run():
            reduced_loss = self.split_tensor(loss.data).item()
        else:
            reduced_loss = loss.item()

    def create_optimizer(self, model_name, alias_name: str):
        """
        Creates an optimizer based on model specs.

        Args:
            model_name: model that optimizer will bind to
            alias_name: optimizer configuration alias name.
        Returns:

        """

        if model_name not in self._models:
            raise Exception("config.yaml must contains valid active settings. "
                            "Failed create {} model".format(model_name))

        logger.info("Creating optimizer for {}", model_name, type(self._models[model_name]))
        model = self._models[model_name]

        optimizer_type = self.trainer_spec.optimizer_type(alias_name)
        spec = self.trainer_spec

        if optimizer_type == 'Adam':
            if self.verbose:
                fmtl_print("{} type".format(alias_name), "Adam")
                fmtl_print("{} lr".format(alias_name), spec.optimizer_learning_rate(alias_name))
                fmtl_print("{} betas".format(alias_name), spec.adam_betas(alias_name))
                fmtl_print("{} eps".format(alias_name), spec.adam_eps(alias_name))
                fmtl_print("{} weight decay".format(alias_name), spec.weight_decay(alias_name))
                fmtl_print("{} amsgrad".format(alias_name), spec.amsgrad(alias_name))

            opt = optim.Adam(list(model.parameters()),
                             lr=spec.optimizer_learning_rate(alias_name),
                             betas=spec.adam_betas(alias_name),
                             eps=spec.adam_eps(alias_name),
                             weight_decay=spec.weight_decay(alias_name),
                             amsgrad=spec.amsgrad(alias_name))
        elif optimizer_type == 'SGD':
            if self.verbose:
                fmtl_print("Creating {} optimizer.".format(alias_name), "SGD")
            opt = optim.opt = optim.SGD(list(self._models[model_name].parameters(alias_name)),
                                        lr=spec.optimizer_learning_rate(alias_name),
                                        momentum=spec.momentum(alias_name),
                                        dampening=spec.dampening(alias_name),
                                        weight_decay=spec.weight_decay(alias_name),
                                        nesterov=spec.nesterov(alias_name))
        elif self.trainer_spec.optimizer_type == 'none':
            opt = None
        else:
            raise ValueError("unknown optimizer: {}".format(optimizer_type))

        self.optimizers[model_name] = opt
        return opt

    # @logger.catch()
    def create_optimizers(self) -> None:
        """
        Creates all required optimizers based on model specs.
        Each optimize in self.optimizers dict
        Returns: Nothing
        """
        _models = self.trainer_spec.get_active_sub_models()
        if len(_models) == 0:
            logger.warning("trainer spec has no model")

        for model_name in _models:
            logger.info("Loading {} optimizer settings".format(model_name))
            opt_spec_alias = self.trainer_spec.get_sub_model_optimizer(model_name)
            optimizer = self.create_optimizer(model_name, opt_spec_alias)
            self.optimizers[model_name] = optimizer

    def create_lr_scheduler(self, model_name: str, optimizer) ->None:
        """
        Creates lr scheduler based on specs and attach to optimizer
        Args:
            model_name:  a model name
            optimizer: target optimizer

        Returns: lr_scheduler

        """
        alias_name = self.trainer_spec.get_sub_model_lr_scheduler(model_name)
        if len(alias_name) == 0:
            if self.verbose:
                fmtl_print("Model {}".format(model_name), "no scheduler attached")
            return

        lr_scheduler_type = self.trainer_spec.lr_scheduler_type(alias_name)
        if lr_scheduler_type == 'cos':
            if self.verbose:
                fmtl_print("Creating {} lr scheduler.".format(alias_name), "cos")
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=self.trainer_spec.t_max(alias_name),
                                                       eta_min=self.trainer_spec.eta_min(alias_name))
        elif lr_scheduler_type == 'multistep':
            fmtl_print("Creating {} lr scheduler.".format(alias_name), "multistep")
            fmtl_print("Creating {} milestone.".format(alias_name), self.trainer_spec.milestones(alias_name))
            if self.verbose:
                fmtl_print("Creating {} lr scheduler.".format(alias_name), "multistep")
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=self.trainer_spec.milestones(alias_name),
                                                 gamma=self.trainer_spec.gamma(alias_name))
        elif lr_scheduler_type == 'exp-warmup':
            if self.verbose:
                fmtl_print("Creating {} lr_scheduler_type.".format(alias_name), "exp-warmup")
            lr_lambdas = self.trainer_spec.lr_lambdas(alias_name)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
        elif lr_scheduler_type == 'none' or lr_scheduler is None:
            if self.verbose:
                fmtl_print("Creating {} optimizer.".format(alias_name), "none")
            scheduler = None
        else:
            raise ValueError("unknown scheduler: {}".format(lr_scheduler_type))

        self.schedulers[model_name] = scheduler
        return scheduler

    def create_lr_schedulers(self) -> None:
        """
        Create all scheduler and bind to optimizer
        :return:
        """
        _models = self.trainer_spec.get_active_sub_models()
        for model_name in _models:
            if model_name not in self.optimizers:
                raise Exception("make sure optimizer spec created.")
            opt = self.optimizers[model_name]
            self.create_lr_scheduler(model_name, opt)

    def save_if_need(self, model_name, it, epoch, last_epoch=False):
        """ Method called by trainer, to check if model need to save or not.
        :param epoch: current epoch
        :param it: Current iteration counter.
        :param model_name:  active model
        :param last_epoch:
        """
        # by default condition to save epoch , if save per iteration we check iteration.
        if self.trainer_spec.is_save() is False:
            return False

        if last_epoch is False and it == 0:
            return False

        # in case we run distributed no need save.
        if self.trainer_spec.is_distributed_run() and self.rank > 0:
            return

        # model save predicate condition , either iteration or epoch counter.
        # for large model it makes sense to track iteration vs epoch counter.
        save_condition = epoch
        model_file = self.trainer_spec.model_files.get_model_file_path(model_name)
        if self.trainer_spec.is_save_per_iteration():
            save_condition = it

        if save_condition % self.trainer_spec.epochs_save() == 0 or last_epoch is True:
            if self.trainer_spec.is_train_verbose():
                logger.info('Saving node model {}'.format(model_file))

            if model_name in self.schedulers:
                logger.info("Model saving with optimizer and scheduler state.")
                torch.save({
                    'epoch': epoch,
                    'it': it,
                    'model_state_dict': self._models[model_name].state_dict(),
                    'optimizer_state_dict': self.optimizers[model_name].state_dict(),
                    'scheduler_state_dict': self.schedulers[model_name].state_dict()
                }, model_file)
            else:
                logger.info("Model saving with optimizer without a scheduler state.")
                torch.save({
                    'epoch': epoch,
                    'it': it,
                    'model_state_dict': self._models[model_name].state_dict(),
                    'optimizer_state_dict': self.optimizers[model_name].state_dict(),
                    #    'scheduler_state_dict': self.schedulers[model_name].state_dict()
                }, model_file)

            self.metric.save()
            return True
        return False

    def log_if_needed(self, it, reduced_loss, grad_norm, duration):
        """

        Args:
            it:
            reduced_loss:
            grad_norm:
            duration:

        Returns:

        """
        if self.rank == 0 and it % self.trainer_spec.epochs_log() == 0:
            print("Train loss "
                  "{} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(it,
                                                                 reduced_loss,
                                                                 grad_norm,
                                                                 duration))

    def validate(self, model, model_name: str, it):
        """

        Args:
            model:
            it:
            n_gpus:

        Returns:

        """
        t_writer = self.trainer_spec.get_tensorboard_writer()

        model.eval()
        with torch.no_grad():
            total_prediction_loss = 0.0
            for i, batch in enumerate(self.validation_loader):
                x, y = self.parse_batch(batch)
                y_pred = model(x)
                # our loss mel_loss + gate_loss
                loss = self.criterion(y_pred, y)
                print("validation loss", loss.item())
                if self.trainer_spec.is_distributed_run():
                    reduced_val_loss = self.split_tensor(loss.data, self.n_gpus).item()
                else:
                    reduced_val_loss = loss.item()
                total_prediction_loss += reduced_val_loss
            # normalize
            total_prediction_loss = total_prediction_loss / (len(self.validation_loader) + 1)

        model.train()
        if self.rank == 0:
            # t_writer.add_scalar('val_loss_' + model_name, loss.item(), it)
            # print("Validation loss {}: {:9f}  ".format(it, total_prediction_loss))
            self.tf_logger.log_validation(total_prediction_loss, model, y, y_pred, it)

        return total_prediction_loss

    def get_last_iterator(self, model_name):
        """
        Return last iterator,  during a model save each model might have own iterator counter.
        thus, trainer has a dict that hold self.iters[mode_name] = it
        Args:
            model_name:

        Returns:
        """
        it = 0
        if model_name in self.iters:
            it = self.iters[model_name]
        return it

    def plot_data(self, data, figsize=(16, 4)):
        """

        :param data:
        :param figsize:
        :return:
        """
        fig, axes = plt.subplots(1, len(data), figsize=figsize)
        for i in range(len(data)):
            axes[i].imshow(data[i],
                           aspect='auto',
                           inorigin='bottom',
                           interpolation='none')

    def inference(self, input_seq=None, model_name='encoder', plot=True,
                  mel_output_path="mel_out.png",
                  mel_post_path="mel_post.png",
                  mel_alignment_path="mel_alignment.png"):
        """
        Perform inference on input
        :param mel_alignment_path:
        :param mel_output_path:
        :param mel_post_path:
        :param input_seq:
        :param model_name:
        :return:
        """
        # model returns  outputs
        # [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        if model_name not in self._models:
            raise Exception("You need load model {}".format(model_name))

        model = self._models[model_name]
        model.eval()

        if isinstance(input_seq, str):
            # input_seq = np.array(text_to_sequence(input_seq, ['english_cleaners']))[None, :]
            # print(input_seq.shape)
            sequence = np.array(text_to_sequence(input_seq, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(
                    torch.from_numpy(sequence)).to(self.device).long()

        # sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
            print(mel_outputs.shape)
            print(mel_outputs_postnet.shape)
            print(alignments.shape)

            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
            if plot:
                plot_spectrogram_to_numpy(mel_outputs.data.cpu().numpy()[0],
                                          file_name=mel_output_path)
                plot_spectrogram_to_numpy(mel_outputs_postnet.data.cpu().numpy()[0],
                                          file_name=mel_post_path)
                plot_alignment_to_numpy(alignments.data.cpu().numpy()[0].T,
                                        file_name=mel_alignment_path)

            return mel_outputs, mel_outputs_postnet, alignments

    # def average_gradients(model):
    #     size = float(dist.get_world_size())
    #     for param in model.parameters():
    #         dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    #         param.grad.data /= size
    #
    def train_epoch(self, model, model_name, optimizer, scheduler=None, save_callback=None):
        """

        :param model:
        :param model_name:
        :param optimizer:
        :param scheduler:
        :param save_callback:
        :return:
        """

        device = self.device

        # if self.trainer_spec.is_distributed_run():
        #     device = torch.device(f"cuda:{dist.get_rank()}")
        # else:
        #     device = self.device

        current_total_loss = 0
        total_accuracy = 0
        step = self.get_last_iterator(model_name)
        for batch_idx, batch in enumerate(self.train_loader):
            # start = time.perf_counter()
            # for param_group in self.optimizer[model_name].param_groups:
            #     param_group['lr'] = learning_rate
            model.zero_grad(set_to_none=True)
            x, y = self.parse_batch(batch, device)
            y_pred = model(x)

            # if self.trainer_spec.is_distributed_run():
            #     reduced_loss = dist.reduce_tensor(loss.data, n_gpus).item()
            loss = self.criterion(y_pred, y)
            current_total_loss += loss.item()
            normal_loss = loss.item()

            loss.backward()
            if self.clip_grad:
                grad_norm = clip_grad_norm_(model.parameters(), self.trainer_spec.grad_clip_thresh())
                self.metric.update(batch_idx, step, normal_loss, grad_norm=grad_norm.item())
            else:
                grad_norm = loss
                self.metric.update(batch_idx, step, normal_loss)

            optimizer.step()
            # run lr_scheduler
            if scheduler is not None:
                scheduler.step()

            # self.log_if_needed(it, loss, grad_norm, duration)
            if self.rank == 0 and step != 0 and step % self.trainer_spec.console_log_rate() == 0:
                self.tqdm_iter.set_postfix({'step': step,
                                            'loss': normal_loss,
                                            'batch_loss': current_total_loss // max(1, batch_idx + 1),
                                            'avg loss': self.metric.total_mean_loss(),
                                            # 'acc': prediction_accuracy,
                                            # 'acc_total': prediction_accuracy,
                                            'grad_norm': grad_norm.item(),
                                            'batch': batch_idx,
                                            'lr': optimizer.param_groups[0]['lr'],
                                            'saved step': self.saved_run})
            #
            # # run prediction if_needed
            # if step != 0 and step % self.trainer_spec.predict() == 0:
            #     prediction_accuracy = self.validate(model, model_name, step)
            #     total_accuracy += prediction_accuracy

            # save model checkpoint if needed
            if self.rank == 0 and self.save_if_need(model_name, step, self.epoch):
                self.saved_run = step

            # dist.barrier()

            hparams = \
                {
                    'lr': optimizer.param_groups[0]['lr'],
                    'batch_size': self.trainer_spec.batch_size}, \
                {
                    # 'hparam/accuracy': prediction_accuracy,
                    'hparam/loss': float(loss.item()),
                    'hparam/grad_norm': float(grad_norm.item())
                }

            self.tf_logger.log_training(normal_loss, step, grad_norm,
                                        optimizer.param_groups[0]['lr'],
                                        hparams)
            step += 1

        self.iters[model_name] = step
        return step

    def parse_batch(self, batch, device):
        """

        :param device:
        :param batch:
        :return:
        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spectral = batch
        text_padded = to_gpu(text_padded, device).long()
        spectral = to_gpu(spectral, device).float()
        input_lengths = to_gpu(input_lengths, device).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded, device).float()
        gate_padded = to_gpu(gate_padded, device).float()
        output_lengths = to_gpu(output_lengths, device).long()

        # assert text_padded.get_device() == 0
        # assert input_lengths.get_device() == 0
        # assert mel_padded.get_device() == 0
        # assert gate_padded.get_device() == 0
        # assert output_lengths.get_device() == 0

        return (text_padded, input_lengths, mel_padded, max_len, output_lengths, spectral), \
               (mel_padded, gate_padded, spectral)

    def cleanup(self):
        """
        Cleanup call
        :param rank:
        :return:
        """
        dist.destroy_process_group()
        logger.info(f"Rank {self.rank} is done.")

    @staticmethod
    def reduce_tensor(self, tensor: Tensor, n_gpus) -> Tensor:
        """
        Reduce tensor based on number of gpu
        :param tensor:
        :param n_gpus:
        :return:
        """
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= n_gpus
        return rt

    def train(self, model_name='encoder'):
        """

        :param model_name:
        :return:
        """
        # torch.manual_seed(self.model_spec.seed())
        # torch.cuda.manual_seed(self.model_spec.seed())
        #
        # if self.rank == 0:
        #     self.load_models()

        # if self.trainer_spec.is_distributed_run():
        #     #torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()

        if self.is_trained():
            print("It looks like model already trained. \
            Check file {}".format(self.trainer_spec.model_files.get_model_file_path(model_name)))
            return

        t_writer = self.trainer_spec.get_tensorboard_writer()

        if model_name in self.iters:
            logger.info("Epoch saved out of", self._last_epochs[model_name], self.trainer_spec.epochs())
            logger.info("Last iteration saved", self.iters[model_name])

        #
        if self.trainer_spec.is_distributed_run():
            # device = torch.device(f"cuda:{dist.get_rank()}")
            device = self.device
        else:
            device = self.device

        self.criterion.to(device)
        model = self._models[model_name]

        self.tqdm_iter = self.trainer_iterator(model_name)
        optimizer = self.optimizers[model_name]

        scheduler = None
        if model_name in self.schedulers:
            logger.info("Model {} contains a scheduler".format(model_name))
            scheduler = self.schedulers[model_name]

        # if self.trainer_spec.is_distributed_run():
        #     logger.info("Running in distributed model. applying gradient reduce.".format(model_name))
        #     model = apply_gradient_allreduce(model)

        it = self.get_last_iterator(model_name)
        if self._last_epochs[model_name] == self.trainer_spec.epochs():
            prediction_accuracy = self.validate(model, model_name, it)

        # TODO add option if epoch changed after save
        self.metric.set_num_iteration(self.trainer_spec.epochs() * self.total_batches)
        self.metric.init()
        logger.info("Staring training num epochs {}, epoch trained {} num batches {} expected total iteration {}",
                    self.trainer_spec.epochs(), self._last_epochs, len(self.train_loader),
                    self.trainer_spec.epochs() * len(self.train_loader))

        # if self.trainer_spec.is_distributed_run():
        #     model = dist
        model.train()
        self.tqdm_iter.set_postfix({'step': it})
        for epoch in self.tqdm_iter:
            if self.trainer_spec.is_distributed_run():
                dist.barrier()
            # update epoch
            self.epoch = epoch
            # train epoch's batch
            self.metric.start_epoch_timer(epoch)
            self.train_epoch(model, model_name, optimizer)
            self.metric.update_epoch_timer(epoch)

            # model logs
            # for name, weight in model.named_parameters():
            #     t_writer.add_histogram(name, weight, epoch)
            #     t_writer.add_histogram(f'{name}.grad', weight.grad, epoch)

            t_writer.flush()
            #  tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

        if self.rank == 0 and self.save_if_need(model_name, it, self.trainer_spec.epochs(), last_epoch=True):
            if self.verbose:
                fmtl_print("Saved last epoch", self.trainer_spec.epochs())

        self.cleanup()

        def train_scaled(self, model_name='encoder'):
            """Training and validation logging results to tensorboard and stdout

            Params
            ------
            output_directory (string): directory to save checkpoints
            log_directory (string) directory to save tensorboard logs
            checkpoint_path(string): checkpoint path
            n_gpus (int): number of gpus
            rank (int): rank of current gpu
            hparams (object): comma separated list of "name=value" pairs.
            """

            # torch.manual_seed(self.model_spec.seed())
            # torch.cuda.manual_seed(self.model_spec.seed())
            #
            self.load_models()

            #
            model = self._models[model_name].to(self.device)
            tqdm_iter = self.trainer_iterator(model_name)
            learning_rate = self.trainer_spec.learning_rate

            optimizer = self.optimizers[model_name]
            #
            if self.trainer_spec.is_fp16_run():
                from apex import amp
                model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
            #
            #    if self.trainer_spec.is_distributed_run():
            # model = apply_gradient_allreduce(model)
            #
            #
            # # logger = prepare_directories_and_logger(
            # #     output_directory, log_directory, rank)
            #
            #
            # # Load checkpoint if one exists
            iteration = 0
            epoch_offset = 0
            # if checkpoint_path is not None:
            #     if warm_start:
            #         # model = warm_start_model(
            #         #     checkpoint_path, model, hparams.ignore_layers)
            #     else:
            #         model, optimizer, _learning_rate, iteration = self.load_checkpoint(checkpoint_path, model, optimizer)
            #         if self.model_spec.use_saved_learning_rate:
            #             learning_rate = _learning_rate
            #         iteration += 1  # next iteration is iteration + 1
            #         epoch_offset = max(0, int(iteration / len(self.train_loader)))

            model.train()
            is_overflow = False
            for epoch in tqdm_iter:
                total_epoch_loss = 0
                for i, batch in enumerate(self.train_loader):
                    start = time.perf_counter()
                    # for param_group in self.optimizer[model_name].param_groups:
                    #     param_group['lr'] = learning_rate

                    model.zero_grad()
                    x, y = self.parse_batch(batch)
                    with torch.cuda.amp.autocast():
                        y_pred = model(x)

                    loss = self.criterion(y_pred, y)
                    total_epoch_loss += loss.item()

                    # reduced_loss = self.split_tensor(loss.data).item()
                    #     else:
                    #         reduced_loss = loss.item()
                    #

                    self.scaler.scale(loss).backward()
                    grad_norm = clip_grad_norm_(model.parameters(), self.trainer_spec.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm)

                    self.scaler.step(optimizer)
                    self.scaler.update()

                    # save model checkpoint
                    if self.save_if_need(model_name, iteration, epoch):
                        tqdm_iter.set_postfix({'total_epoch_loss': total_epoch_loss, 'saved': True})
                    iteration += 1

    def is_trained(self):
        """

        Returns:

        """
        models = self.trainer_spec.get_active_sub_models()
        num_finished = 0
        for m in models:
            if m in self._last_epochs:
                if int(self._last_epochs[m]) >= int(self.trainer_spec.epochs()):
                    num_finished += 1

        if num_finished == len(models):
            return True

        return False

    def get_model(self, model_name):
        """

        :param model_name:
        :return:
        """
        if model_name in self._models:
            return self._models[model_name]
#
# def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
#     config = {
#         "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([2, 4, 8, 16])
#     }
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     experiment_specs = ExperimentSpecs(verbose=False)
#     dataloader = Mel_Dataloader(experiment_specs, verbose=False)
#     trainer = Trainer(experiment_specs, dataloader, verbose=True, device=device)
#
#     scheduler = ASHAScheduler(
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2)
#     result = tune.run(
#         tune.with_parameters(trainer.train()),
#         resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
#         config=config,
#         metric="loss",
#         mode="min",
#         num_samples=num_samples,
#         scheduler=scheduler
#     )
#
#     best_trial = result.get_best_trial("loss", "min", "last")
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(
#         best_trial.last_result["loss"]))
#     print("Best trial final validation accuracy: {}".format(
#         best_trial.last_result["accuracy"]))
#
#     if ray.util.client.ray.is_connected():
#         # If using Ray Client, we want to make sure checkpoint access
#         # happens on the server. So we wrap `test_best_model` in a Ray task.
#         # We have to make sure it gets executed on the same node that
#         # ``tune.run`` is called on.
#         from ray.util.ml_utils.node import force_on_current_node


#     remote_fn = force_on_current_node(ray.remote(test_best_model))
#  ray.get(remote_fn.remote(best_trial))
# else:
# test_best_model(best_trial)


#
# if is_inference:
#     # trainer = Trainer(experiment_specs, dataloader, verbose=True, device=device)
#     # text = "Hello world, I missed you so much."
#     # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
#     # sequences, lengths = utils.prepare_input_sequence([text])
#
#     text = "artist would have difficulty in doing such accurate work"
#     sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
#     sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
#
#     # model_math = 'fp16'
#     waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow')
#     waveglow = waveglow.remove_weightnorm(waveglow)
#     waveglow = waveglow.to('cuda')
#     waveglow.eval()
#
#     with torch.no_grad():
#         print(sequence)
#         mel_outputs, mel_outputs_post_net, alignments = trainer.inference(sequence)
#         audio_from_mel = waveglow.infer(mel_outputs)
#         audio_from_post = waveglow.infer(mel_outputs_post_net)
#         print("waveglow return mel", audio_from_mel.shape)
#         print("waveglow return post", audio_from_post.shape)
#
#         audio_numpy_mel = audio_from_mel[0].data.cpu().numpy()
#         audio_numpy_post = audio_from_post[0].data.cpu().numpy()
#         #
#         rate = 22050
#         from scipy.io.wavfile import write
#
#         write("audio_mel.wav", rate, audio_numpy_mel)
#         write("audio_from_post.wav", rate, audio_numpy_post)

# mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
# plot_data((mel_outputs.float().data.cpu().numpy()[0],
#            mel_outputs_postnet.float().data.cpu().numpy()[0],
#            alignments.float().data.cpu().numpy()[0].T))

# train_loader, val_loader, xx = dataloader.create()
# for i, batch in enumerate(train_loader):
#     print(dir(batch))

# model.zero_grad()
# x, y = model.parse_batch(batch)
# y_pred = model(x)

# trainer.train()

# train_set = TextMelLoader(encoder_spec, list(training_set.values()))
#
# start = timeit.timeit()
# print("hello")
# end = timeit.timeit()
# for mel, text in train_set:
#     mel, text = train_set.__getitem__(0)
# print(end - start)

# print("size of dataset", len(train_set))
# print(text.shape)
# print(mel.shape)
# model_trainer_spec.build_training_set_from_files()
# print(model_trainer_spec.dataset_specs['dir'])
# training_set, validation_set, test_set = model_trainer_spec.get_audio_ds_files()
# trainer = Trainer(model_trainer_spec)
# train_set = TextMelLoader(model_trainer_spec, list(training_set.values()))
# val_set = TextMelLoader(hparams.validation_files, hparams)
# collate_fn = TextMelCollate(hparams.n_frames_per_step)
