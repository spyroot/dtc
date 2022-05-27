#
#  TODO check shuffle
#  TODO Metric plot
#  TODO Loss plot.
# import torch.distributed as dist
import os
import queue
import random
from abc import ABC
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from loguru import logger
from numpy import finfo
from torch import Tensor
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, tnrange

from model_loader.mel_dataloader import SFTFDataloader
from model_trainer.base_trainer import TrainerError, AbstractTrainer
from model_trainer.callbacks.base import BaseCallbacks, Callback
from model_trainer.distributed_wrapper import DistributedDataWrapper
# from model_trainer.trainer_logger import TensorboardTrainerLogger
from model_trainer.trainer_metrics import Metrics
from model_trainer.trainer_specs import ExperimentSpecs
# from distributed import apply_gradient_allreduce
from models.loss_function import Tacotron2Loss
from models.tacatronv30.model import Tacotron3
from models.tacotronv25.model import Tacotron25
from model_trainer.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from model_trainer.utils import fmtl_print, to_gpu

# from multiprocessing import Queue
from collections import deque

# from torch.nn.parallel import DistributedDataParallel
# from torch.autograd import Variable
# import numpy as np
import ray
from ray import tune

# try:
#     import ray
#     from ray import tune
#     from ray.tune.schedulers import ASHAScheduler
# except ImportError:
#     logger.info("ray not found")
#     pass

import matplotlib.pylab as plt
from text import text_to_sequence


# try:
#     O_BINARY = os.O_BINARY
# except:
#     O_BINARY = 0
#
# READ_FLAGS = os.O_RDONLY | O_BINARY
# WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_BINARY
# BUFFER_SIZE = 128 * 1024
#
#
# def copyfile(src, dst):
#     try:
#         fin = os.open(src, READ_FLAGS)
#         stat = os.fstat(fin)
#         fout = os.open(dst, WRITE_FLAGS, stat.st_mode)
#         for x in iter(lambda: os.read(fin, BUFFER_SIZE), ""):
#             os.write(fout, x)
#     finally:
#         try:
#             os.close(fin)
#         except:
#             pass
#         try:
#             os.close(fout)
#         except:
#             pass

@AbstractTrainer.register
class Trainer(AbstractTrainer, ABC):
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
                 callback: Optional[list[Callback]] = None) -> None:
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
        super(Trainer, self).__init__(trainer_spec=trainer_spec,
                                      data_loader=data_loader,
                                      verbose=verbose,
                                      is_notebook=is_notebook,
                                      rank=rank,
                                      world_size=world_size,
                                      disable_pbar=disable_pbar,
                                      device=device,
                                      cuda_device_id=cuda_device_id,
                                      is_inference=is_inference)
        self.set_logger(verbose)
        #
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
            self._dataloaders, self.collate_fn = data_loader.get_all()
            self._train_loader = self._dataloaders[self._tkey]
            self._validation_loader = self._dataloaders[self._vkey]

        # TODO need refactor that, and move to dict and abstract,
        self.criterion = Tacotron2Loss()
        # dict store all model
        self._models = {}
        # store last epoch
        self._last_ckt_epochs: dict[str, dict[str, int]] = {}
        # dict holds model name = last iterator value
        self._steps: dict[str, int] = {}
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
            self.total_batches = len(self._dataloaders[self._tkey])
            # clip or not grad
            self.clip_grad = trainer_spec.is_grad_clipped()

            # self.tf_logger = TensorboardTrainerLogger(trainer_spec.tensorboard_update_rate())
            self.metric = Metrics(metric_step_file_path=trainer_spec.model_files.get_metric_file_path(),
                                  metric_batch_file_path=trainer_spec.model_files.get_time_file_path(),
                                  metric_perf_trace_path=trainer_spec.model_files.get_metric_batch_file_path(),
                                  num_epochs=self.trainer_spec.epochs(), num_batches=self.total_batches,
                                  verbose=False)

        self._callback = BaseCallbacks(callbacks=callback)
        logger.info("Device ID {}".format(self.cuda_device_id))
        # main queue note that train loop can re-add model back for training.
        self.q = deque()
        # late we will move this to factory
        # type class that will do creation
        self.model_creator, self.trainer_dispatcher, self._batch_loader = self.create_model_dispatch()
        # init trainer
        self.init_trainer()

    def create_model_dispatch(self) -> tuple[dict[str, dict[str: Callable]],
                                             dict[str, Callable],
                                             dict[str, dict[str: Callable]]]:
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
            'tacotron25': {
                'spectrogram_layer': self.create_tacotron25,
                'vocoder': self.create_tacotron25
            },
            'dts': {
                'spectrogram_layer': self.create_tacotron30,
                'vocoder': self.create_tacotron25
            }
        }

        trainer_dispatch = {
            # 'GraphGRU': RnnGenerator,
            # 'GraphLSTM': RnnGenerator,
        }

        batch_loader = {
            'tacotron25': {
                'spectrogram_layer': self.tacotron25_batch,
            },
            'dts': {
                'spectrogram_layer': self.tacotron30_batch,
            }
        }

        return model_dispatch, trainer_dispatch, batch_loader

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

    def create_tacotron25(self, is_set_cuda=False) -> nn.Module:
        """
        Factor method, creator for tacotron 2.5
        :return:
        """
        logger.debug("Creating tacotron25 model.")
        if self.is_inference:
            model = Tacotron25(self.trainer_spec, self.device).to(self.device)
        elif self.trainer_spec.is_distributed_run():
            # device = torch.device(f"cuda:{dist.get_rank()}")
            device = self._loop_up_device(is_set_cuda)
            logger.info("Creating DDP on cuda device "
                        "{} torch device {} device received {}".format(self.cuda_device_id, device, self.device))
            model = Tacotron3(self.trainer_spec, device).cuda()
            model = DistributedDataWrapper(model,
                                           device_ids=[self.cuda_device_id],
                                           output_device=self.cuda_device_id).cuda()

        else:
            model = Tacotron25(self.trainer_spec, self.device).to(self.device)

        if self.trainer_spec.is_fp16_run():
            model.decoder.attention_layer.score_mask_value = finfo('float16').min

        return model

    def create_tacotron30(self, is_set_cuda=False) -> nn.Module:
        """
        Factor method, creator for tacotron 3
        :return:
        """
        if self.is_inference:
            model = Tacotron3(self.trainer_spec, self.device).to(self.device)
        elif self.trainer_spec.is_distributed_run():
            device = self._loop_up_device(is_set_cuda)
            logger.info("Creating DDP on cuda device "
                        "{} torch device {} device received {}".format(self.cuda_device_id,
                                                                       device, self.device))
            model = Tacotron3(self.trainer_spec, device).cuda()
            model = DistributedDataWrapper(model,
                                           device_ids=[self.cuda_device_id],
                                           output_device=self.cuda_device_id).cuda()

        else:
            model = Tacotron3(self.trainer_spec, self.device).to(self.device)

        if self.trainer_spec.is_fp16_run():
            model.decoder.attention_layer.score_mask_value = finfo('float16').min

        return model

    def _create_model_layers(self, model_name: str, layer_name: Optional[str] = None, is_set_cuda=False):
        """
          Method lookup model from dispatch and call factory method.
          Later will move to a separate dispatcher creator or maybe
          register creator.

        :param model_name:
        :param layer_name:
        :return: nothing
        :param is_set_cuda: will pin directly
        """
        logger.info(f"Creating layer {model_name} {layer_name}")
        if model_name not in self.model_creator:
            raise ValueError("Unknown {} model for a "
                             "trainer, supported models are {}".format(model_name, list(self.model_creator.keys())))

        creator = self.model_creator[model_name][layer_name]
        if model_name not in self._models:
            self._models[model_name] = {}

        m = creator(is_set_cuda)
        if m is None:
            raise TrainerError("Failed create a model. {}".format(creator.__name__))

        self._models[model_name][layer_name] = m
        logger.debug(f"Added model layer {layer_name} to a model {model_name}")
        if model_name not in self._last_ckt_epochs:
            self._last_ckt_epochs[model_name] = {}

        # update last epoch
        self._last_ckt_epochs[model_name][layer_name] = 0

    def create_models(self):
        """
        For each active model, which might consist two or more models.
        we create a target model
        :return: nothing, create_model will add each model to a dict[model_name] = model
        """
        active_model = self.trainer_spec.get_active_mode()
        _models = self.trainer_spec.get_active_sub_models()
        for m in _models:
            self._create_model_layers(active_model, m)

    def load(self, model_name: str, layer_name: str, to_device=True, ignore_layers=None, model_file=None):
        """
        Method loads model from a checkpoint.  It will update internal state,
        that include model, last epoch, last step.

        :param model_name: - mode name that method need to load
        :param layer_name: - layer of model that we need load.
        :param model_file: - model file.
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
                raise TrainerError("model has no state dict")

            self._models[model_name][layer_name].load_state_dict(checkpoint['model_state_dict'])
            if 'model_state_dict' not in checkpoint:
                raise TrainerError("model has no state dict")

            self._optimizers[model_name][layer_name].load_state_dict(checkpoint['optimizer_state_dict'])
            if 'optimizer_state_dict' not in checkpoint:
                raise TrainerError("model has no optimizer_state_dict")

            # if model has scheduler, re-create.
            if model_name in self._schedulers:
                self._schedulers[model_name].load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scheduler_state_dict' not in checkpoint:
                    raise TrainerError("model has no scheduler_state_dict")

            #  ignore layers.
            if ignore_layers is not None and len(ignore_layers) > 0:
                model_dict = {k: v for k, v in self._models[model_name].items()
                              if k not in ignore_layers}
                new_state = self._models[model_name].state_dict()
                new_state.update(model_dict)
                self._models[model_name] = new_state

            # self.trainer_spec.set_lr(0.00001)
            if 'epoch' not in checkpoint:
                raise TrainerError("saved checkpoint has no epoch key")
            self._last_ckt_epochs[model_name][layer_name] = checkpoint['epoch']

            # load last iterator.
            if 'it' not in checkpoint:
                raise TrainerError("saved checkpoint has no last iteration key.")
            self._steps[layer_name] = checkpoint['it']
            logger.info("Last checkpoint. epoch {} step {}".format(checkpoint['epoch'], checkpoint['it']))

            # load metric
            self.metric.load()

            return checkpoint['epoch'], checkpoint['it']
        except FileNotFoundError as e:
            print("Failed load model files {}. No saved model found.".format(model_file))
            logger.info("No model file to load model file, return default epoch 0, iteration 0")

        return 0, 0

    def load_for_inference(self, model_name: str, layer_name: str, model_file=None, to_device=True, ignore_layers=None):
        """
        Method loads model from a checkpoint for inference.

        :param model_name: - mode name that method need to load
        :param layer_name: - layer we need load
        :param model_file: - torch model file.
        :param ignore_layers: - a list contain of layers we skip
        :param to_device: - if true will load to device indicate as main device
        :return: epoch, step
        """

        logger.info("Loading model '{}' "
                    "model file name {} to device {}".format(model_name, model_file, self.device))

        try:
            checkpoint = torch.load(model_file, map_location=self.device)
            if 'model_state_dict' not in checkpoint:
                raise TrainerError("model has no state dict")

            self._create_model_layers(model_name)
            assert model_name in self._models
            self._models[model_name].load_state_dict(checkpoint['model_state_dict'])
            if 'model_state_dict' not in checkpoint:
                raise TrainerError("model has no state dict")

            if ignore_layers is not None and len(ignore_layers) > 0:
                model_dict = {k: v for k, v in self._models[model_name].items()
                              if k not in ignore_layers}
                new_state = self._models[model_name].state_dict()
                new_state.update(model_dict)
                self._models[model_name] = new_state

            if 'epoch' not in checkpoint:
                raise TrainerError("saved checkpoint has no epoch key.")
            self._last_ckt_epochs[model_name][layer_name] = checkpoint['epoch']

            # load last iterator.
            if 'it' not in checkpoint:
                raise TrainerError("saved checkpoint has no last step key.")

            self._steps[layer_name] = checkpoint['it']
            logger.info("Last checkpoint. epoch {} step {}".format(checkpoint['epoch'], checkpoint['it']))
            return checkpoint['epoch'], checkpoint['it']
        except FileNotFoundError as e:
            print("Failed load model files {}. No saved model found. {}".format(model_file, e))
            logger.info("No model file to load model file, return default epoch 0, iteration 0")

        return 0, 0

    def trainer_iterator(self, model_name: str, layer_name: str, last_epoch=0, max_epochs=0):
        """
        Method return iterator, i.e if need compute, offset , update tqdm etc.
        :param model_name:
        :param layer_name:
        :param last_epoch:
        :param max_epochs:
        :return:
        """
        # for notebook, we need a bit of hack for tqdm_iter.
        if self.is_notebook:
            tqdm_iter = tnrange(last_epoch, max_epochs)
            return tqdm_iter

        # load last epoch in case we do re-summing.
        last_epoch = self._last_ckt_epochs[model_name][layer_name]
        # early stopping
        early_stopping = None

        max_epochs = self.trainer_spec.epochs()
        # what tqdm to use notebook for colab or normal one.
        logger.info(f"Creating tqdm last epoch {last_epoch} max epoch {max_epochs}")

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

    # def load_models(self) -> None:
    #     """
    #     For each active model, which might consist of more than one model.
    #     :return:
    #     """
    #     models = self.trainer_spec.get_active_sub_models()
    #     for m in models:
    #         self.load(m)

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

    def _create_optimizer(self, model_name: str, layer_name: str, alias_name: str):
        """
        Creates an optimizer based on model specification.
        Each layer in model might have different optimizers.

        For example,  we train GAN we might optimize for Generator and Optimizer for Discriminator.
        Thus,  we define spec for each and strategy.

        For example, we want train model X n epochs and then move to our main model.

        We want to define different scenarios and observers as a result. Hence, we can define a spec for optimizer
        a, b, and c and bind them to the same model and observer result.

        Internally we store dict that store model -> layer -> optimizer

        :param model_name: main model name
        :param layer_name: layer name in that model. For example in case GAN you have two sub-model ie layers
        :param alias_name: alias name, bind specific configuration.
        :return:
        """
        if model_name is None or len(model_name) == 0:
            raise TrainerError("Can't create optimizer, empty model name.")

        if layer_name is None or len(layer_name) == 0:
            raise TrainerError("Can't create optimizer empty model layer.")

        if model_name not in self._models:
            raise TrainerError(f"Can't create optimizer, {model_name} not found in model list.")

        if layer_name not in self._models[model_name]:
            raise TrainerError(f"{str(self.trainer_spec.config_file_name)} "
                               f"must contains valid binding for optimizer. "
                               f"Failed create '{model_name}' model")

        logger.info(f"Creating optimizer for {model_name} layer {layer_name} "
                    f"and optimizer type {self.trainer_spec.optimizer_type(alias_name)}")

        model_layer = self._models[model_name][layer_name]

        optimizer_type = self.trainer_spec.optimizer_type(alias_name)
        spec = self.trainer_spec

        if optimizer_type == 'Adam':
            logger.debug(f"Adam {alias_name} "
                         f"rl:{spec.optimizer_learning_rate(alias_name)} "
                         f"betas:{spec.adam_betas(alias_name)} "
                         f"eps: {spec.adam_eps(alias_name)} "
                         f"decays: {spec.weight_decay(alias_name)} "
                         f"amsgrad: {spec.amsgrad(alias_name)}")
            opt = optim.Adam(list(model_layer.parameters()),
                             lr=spec.optimizer_learning_rate(alias_name),
                             betas=spec.adam_betas(alias_name),
                             eps=spec.adam_eps(alias_name),
                             weight_decay=spec.weight_decay(alias_name),
                             amsgrad=spec.amsgrad(alias_name))

        elif optimizer_type == 'SGD':
            logger.debug(f"SGD {alias_name} "
                         f"rl:{spec.optimizer_learning_rate(alias_name)} "
                         f"momentum:{spec.momentum(alias_name)} "
                         f"eps: {spec.dampening(alias_name)} "
                         f"weight_decay: {spec.weight_decay(alias_name)} "
                         f"nesterov: {spec.nesterov(alias_name)}")
            opt = optim.opt = optim.SGD(list(self._models[model_name].parameters(alias_name)),
                                        lr=spec.optimizer_learning_rate(alias_name),
                                        momentum=spec.momentum(alias_name),
                                        dampening=spec.dampening(alias_name),
                                        weight_decay=spec.weight_decay(alias_name),
                                        nesterov=spec.nesterov(alias_name))
        elif self.trainer_spec.optimizer_type == 'none':
            opt = None
        else:
            raise TrainerError("Unknown optimizer: {}".format(optimizer_type))

        logger.info(f"Bounded optimizer to a model {model_name}, model layer {layer_name}")
        if model_name not in self._optimizers:
            self._optimizers[model_name] = {}

        self._optimizers[model_name][layer_name] = opt
        return opt

    # @logger.catch()
    def create_optimizers(self) -> None:
        """
          Creates all required optimizers based on model specification.
          Read comment for create_optimizer

        :return:
        """
        model_name = self.trainer_spec.get_active_mode()
        model_layers = self.trainer_spec.get_active_sub_models()
        if len(model_layers) == 0:
            logger.warning("Trainer spec has no model layer defined..")

        for model_layer_name in model_layers:
            logger.debug(f"Loading {model_layer_name} optimizer settings.")
            opt_spec_alias = self.trainer_spec.get_sub_model_optimizer(model_layer_name)
            self._create_optimizer(model_name, model_layer_name, opt_spec_alias)

    def create_lr_scheduler(self, model_name: str, model_layer: str, optimizer: torch.optim.Optimizer) -> None:
        """
        Creates lr scheduler based on specs and attach to target optimizer.
        Note we can attach many scheduler.

        :param model_name:
        :param model_layer:
        :param optimizer:
        :return:
        """
        # scheduler is optional
        alias_name = self.trainer_spec.get_sub_model_lr_scheduler(model_layer)
        if len(alias_name) == 0:
            logger.info(f"Model {model_layer} layer {model_layer} no scheduler attached.")
            return

        if optimizer is None:
            raise TrainerError("Can't create lr scheduler. Optimizer is None.")

        lr_scheduler_type = self.trainer_spec.lr_scheduler_type(alias_name)
        if lr_scheduler_type == 'cos':
            logger.info(f"Creating cos lr scheduler. model: {model_name} layer: {model_layer} spec: {alias_name}")
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=self.trainer_spec.t_max(alias_name),
                                                       eta_min=self.trainer_spec.eta_min(alias_name))
        elif lr_scheduler_type == 'multistep':
            logger.info(f"Creating multistep lr scheduler. model: {model_name} "
                        f"layer: {model_layer} spec: {alias_name}")
            logger.info(f"Creating {self.trainer_spec.milestones(alias_name)} milestone.")
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=self.trainer_spec.milestones(alias_name),
                                                 gamma=self.trainer_spec.gamma(alias_name))
        elif lr_scheduler_type == 'exp-warmup':
            logger.info(f"Creating exp-warmup lr scheduler. model: {model_name} "
                        f"layer: {model_layer} spec: {alias_name}")
            lr_lambdas = self.trainer_spec.lr_lambdas(alias_name)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
        elif lr_scheduler_type == 'none' or lr_scheduler is None:
            if self.verbose:
                fmtl_print("Creating {} optimizer.".format(alias_name), "none")
            scheduler = None
        else:
            raise ValueError("unknown scheduler: {}".format(lr_scheduler_type))

        self._schedulers[model_name] = scheduler
        return scheduler

    def create_lr_schedulers(self) -> None:
        """

        Create all scheduler and bind to optimizer.

        :return:
        """
        model_name = self.trainer_spec.get_active_model_name()
        if model_name not in self._optimizers:
            raise TrainerError(f"Model {model_name} must be created first.")

        # model_layers = self.trainer_spec.get_active_sub_models()
        # for layer in model_layers:
        #     if layer not in self._optimizers[model_name]:
        #         raise TrainerError(f"Model {model_name}, model layer {layer} must contain attached optimizer.")
        #     opt = self._optimizers[model_name][layer]
        #     logger.debug("Attaching lr scheduler.")
        #     self.create_lr_scheduler(model_name, layer, opt)

    def save_if_need(self, model_name: str, layer_name: str,
                     epoch: int,
                     step: Optional[int] = 0,
                     last_epoch=False) -> bool:
        """
        Method called by trainer, to check if model 
        or model layer need to save or not.
        
        :param layer_name:
        :param epoch: current epoch
        :param step: Current step in iteration , monotonic counter.
        :param model_name:  active model
        :param last_epoch: if it last we save a model.
        """
        # by default condition to save epoch , if save per iteration we check iteration.
        if self.trainer_spec.is_save_required() is False or step == 0 or \
                self.trainer_spec.epochs_save() == 0:
            return False

        # in case we run distributed no need save.
        if self.trainer_spec.is_distributed_run() and self.rank > 0:
            return False

        # model save predicate condition, either iteration or epoch counter.
        # for large model it makes sense to track iteration vs epoch counter.
        model_file = self.trainer_spec.model_files.get_model_file_path(layer_name)
        save_condition = epoch if self.trainer_spec.is_save_iteration() else step

        # do nothing
        if last_epoch is True or save_condition % self.trainer_spec.epochs_save() == 0:
            if self.trainer_spec.is_train_verbose():
                logger.info('Saving node model {}'.format(model_file))
                self._callback.saving_start()
                # backup before safe.
                # if self.trainer_spec.is_backup_before_save():
                #     self.backup_model(model_file)

            if model_name in self._schedulers:
                logger.info("Model saving with optimizer and scheduler state.")
                torch.save({
                    'epoch': epoch, 'it': step,
                    'model_state_dict': self._models[model_name][layer_name].state_dict(),
                    'optimizer_state_dict': self._optimizers[model_name][layer_name].state_dict(),
                    'scheduler_state_dict': self._schedulers[model_name].state_dict()
                }, model_file)
            else:
                logger.info("Model saving with optimizer without a scheduler state.")
                torch.save({
                    'epoch': epoch,
                    'it': step,
                    'model_state_dict': self._models[model_name][layer_name].state_dict(),
                    'optimizer_state_dict': self._optimizers[model_name][layer_name].state_dict(),
                    #    'scheduler_state_dict': self.schedulers[model_name].state_dict()
                }, model_file)

            self._callback.saved()
            # save metrics
            self.metric.save()
            return True

        return False

    def validate_epoch(self, model: nn.Module, model_name: str, layer_name: str, step: Optional[int] = None):
        """
        Validation epoch

        :param model:  model we are training.
        :param model_name:  model_name a model we are training.
        :param layer_name: layer in that model we are training.
        :param step: optional,  if we do validation in some n step before
                                end of epoch, we want log and track.
        :return:
        """

        # take a batch.
        logger.info(f"Running validation for {model_name} {layer_name}")
        self._callback.validation_start()
        take_batch = self._batch_loader[model_name][layer_name]
        model.eval()
        with torch.no_grad():
            total_prediction_loss = 0.0
            for batch_idx, batch in enumerate(self.validation_loader):
                x, y = take_batch(batch, self.device)
                y_pred = model(x)
                # our loss mel_loss + gate_loss
                criterion_out = self.criterion(y_pred, y)
                mel_loss = criterion_out['mel_loss']
                gate_loss = criterion_out['gate_loss']
                loss = criterion_out['loss']

                if self.trainer_spec.is_distributed_run():
                    reduced_val_loss = self.split_tensor(loss.data, self.n_gpus).item()
                else:
                    reduced_val_loss = loss.item()
                total_prediction_loss += reduced_val_loss

                self.tqdm_iter.set_postfix({'step': step,
                                            'loss': loss.item(),
                                            'batch_loss': total_prediction_loss // max(1, batch_idx + 1),
                                            'avg loss': self.metric.total_mean_loss(),
                                            'mel_loss': mel_loss.item(),
                                            'gate_loss': gate_loss.item(),
                                            'clip_loss': loss.item(),
                                            'batch': batch_idx,
                                            'saved step': self.saved_run})
            # normalize
            total_prediction_loss = total_prediction_loss / (len(self.validation_loader) + 1)
        self._callback.validation_end()

        # tune.report(loss=total_prediction_loss)
        model.train()
        # if self.rank == 0:
        # t_writer.add_scalar('val_loss_' + model_name, loss.item(), it)
        # print("Validation loss {}: {:9f}  ".format(it, total_prediction_loss))
        # criterions = {
        #     "train_normal_loss": normal_loss,
        #     "train_clipped_loss": grad_norm,
        #     "train_mel_loss": mel_loss.item(),
        #     "train_gate_loss": mel_loss.item(),
        #     "total_prediction_loss": total_prediction_loss,
        # }
        #   self.tf_logger.log_validation(total_prediction_loss, model, y, y_pred, step=step)

        return total_prediction_loss

    def get_last_iterator(self, model_name: str, layer_name: str) -> int:
        """
         Return last iterator,  for a given model and layer.
         During a model save each model might have own iterator counter.
         thus, trainer has a dict that hold self.iters[mode_name][layer_name] = it

        :param model_name: model name that must already present.
        :param layer_name: model layer must be already created.
        :return: return last iterator counter.
        """
        it = 0
        if layer_name in self._steps:
            it = self._steps[layer_name]
        else:
            self._steps[layer_name] = 0
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
        Perform inference on input.
        :param plot:
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
            raise TrainerError("You need load model {}".format(model_name))

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
    def train_epoch(self, model, model_name: str, layer_name: str, optimizer, scheduler=None, save_callback=None):
        """

        :param model:
        :param model_name:
        :param layer_name: 
        :param optimizer:
        :param scheduler:
        :param save_callback:
        :return:
        """
        device = self.device
        take_batch = self._batch_loader[model_name][layer_name]
        tbar_update_rate = self.trainer_spec.console_log_rate()

        # if self.trainer_spec.is_distributed_run():
        #     device = torch.device(f"cuda:{dist.get_rank()}")
        # else:
        #     device = self.device

        total_accuracy = 0
        current_total_loss = 0
        step = self.get_last_iterator(model_name, layer_name)
        self._callback.on_loader_begin()
        for batch_idx, batch in enumerate(self._train_loader):
            self._callback.on_batch_begin()
            model.zero_grad(set_to_none=True)
            x, y = take_batch(batch, device)
            y_pred = model(x)
            # if self.trainer_spec.is_distributed_run():
            #     reduced_loss = dist.reduce_tensor(loss.data, n_gpus).item()
            criterion_out = self.criterion(y_pred, y)
            mel_loss = criterion_out['mel_loss']
            gate_loss = criterion_out['gate_loss']
            loss = criterion_out['loss']

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
            self._callback.on_batch_begin()
            # run lr_scheduler
            if scheduler is not None:
                scheduler.step()

            # self.log_if_needed(it, loss, grad_norm, duration)
            if self.rank == 0 and step != 0 and step % tbar_update_rate == 0:
                self.tqdm_iter.set_postfix({'step': step,
                                            'loss': normal_loss,
                                            'batch_loss': current_total_loss // max(1, batch_idx + 1),
                                            'avg loss': self.metric.total_mean_loss(),
                                            'mel_loss': mel_loss.item(),
                                            'gate_loss': gate_loss.item(),
                                            'clip_loss': grad_norm.item(),
                                            'batch': batch_idx,
                                            'lr': optimizer.param_groups[0]['lr'],
                                            'saved step': self.saved_run})
            # # run prediction if_needed
            # if step != 0 and step % self.trainer_spec.predict() == 0:
            #     prediction_accuracy = self.validate(model, model_name, step)
            #     total_accuracy += prediction_accuracy

            # save model checkpoint if needed
            if self.rank == 0 and self.save_if_need(model_name=model_name, layer_name=layer_name,
                                                    epoch=self.epoch, step=step):
                print("Updating saved")
                self.saved_run = step

            # dist.barrier()
            # hparam we want track.
            hparams = {
                'lr': optimizer.param_groups[0]['lr'],
                'batch_size': self.trainer_spec.batch_size,
            }

            metrics = {
                'hparam/loss': normal_loss,
                'hparam/grad_norm': grad_norm,
                'hparam/mel_los': mel_loss,
                'hparam/gate_loss': grad_norm,
            }

            criterions = {
                "train_normal_loss": normal_loss,
                "train_clipped_loss": grad_norm,
                "train_mel_loss": mel_loss,
                "train_gate_loss": gate_loss,
            }

            # self.tf_logger.log_training(criterions, step, optimizer.param_groups[0]['lr'],
            #                             hparams=hparams, metrics=metrics)
            step += 1

        self._callback.on_loader_end()
        self._steps[layer_name] = step
        return step

    def tacotron25_batch(self, batch, device):
        """

        :param device:
        :param batch:
        :return:
        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded, device).long()
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

        return (text_padded, input_lengths, mel_padded, max_len, output_lengths), \
               (mel_padded, gate_padded)

        # return (text_padded, input_lengths, mel_padded, max_len, output_lengths, spectral), \
        #        (mel_padded, gate_padded, spectral)

    def tacotron30_batch(self, batch, device):
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
        :return:
        """
        dist.destroy_process_group()
        logger.info(f"Rank {self.rank} is done.")

    @staticmethod
    def reduce_tensor(tensor: Tensor, n_gpus) -> Tensor:
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

    def prepare_trainer(self, model_name: str, layer_name: str):
        """

        Does all last sanity check and must provide to a trainer
        loop all what it required to train.

        :param model_name:
        :param layer_name:
        :return:
        """
        assert model_name in self._models
        assert layer_name in self._models[model_name]
        model = self._models[model_name][layer_name]
        if self.trainer_spec.is_distributed_run():
            device = self.device
        else:
            device = self.device

        assert self.criterion is not None
        self.criterion.to(device)
        self.tqdm_iter = self.trainer_iterator(model_name, layer_name)

        assert model_name in self._optimizers
        # assert layer_name in self._optimizers[model_name]

        optimizer = self._optimizers[model_name][layer_name]
        scheduler = None

        if layer_name in self._schedulers:
            logger.info("Model {} contains a scheduler".format(model_name))
            scheduler = self._schedulers[model_name]

        # if self.trainer_spec.is_distributed_run():
        #     logger.info("Running in distributed model. applying gradient reduce.".format(model_name))
        #     model = apply_gradient_allreduce(model)

        step = self.get_last_iterator(model_name, layer_name)
        if self._last_ckt_epochs[model_name][layer_name] == self.trainer_spec.epochs():
            prediction_accuracy = self.validate_epoch(model, model_name, layer_name, step)

        # TODO add option if epoch changed after save
        self.metric.set_num_iteration(self.trainer_spec.epochs() * self.total_batches)
        self.metric.init()
        logger.info("Staring training num epochs {}, epoch trained {} num batches {} expected total iteration {}",
                    self.trainer_spec.epochs(), self._last_ckt_epochs, len(self._train_loader),
                    self.trainer_spec.epochs() * len(self._train_loader))

        return model, optimizer, scheduler, step

    def sequential(self, model_name: str, layer_name: str, checkpoint_dir: str):
        """
        Sequential training loop.

        :param model_name:
        :param layer_name:
        :return:
        """
        assert model_name in self._models
        assert layer_name in self._models[model_name]

        model, optimizer, scheduler, step = self.prepare_trainer(model_name, layer_name)

        if layer_name in self._steps:
            logger.info(f"Epoch saved out of",
                        {self._last_ckt_epochs[model_name][layer_name]},
                        {self.trainer_spec.epochs()})
            logger.info(f"Last iteration saved {self._steps[layer_name]}")

        # if self.trainer_spec.is_distributed_run():
        #     model = dist
        model.train()
        self.tqdm_iter.set_postfix({'step': step})
        self._callback.on_begin()

        # if checkpoint_dir is None:
        #     checkpoint_dir =
        # checkpoint_dir = self.trainer_spec.model_files.get_model_dir()
        if checkpoint_dir is not None:
            model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        for epoch in self.tqdm_iter:
            self._callback.on_epoch_begin()
            if self.trainer_spec.is_distributed_run():
                dist.barrier()
            # update epoch
            self.epoch = epoch
            # train epoch's batch
            self.metric.start_epoch_timer(epoch)
            step = self.train_epoch(model, model_name, layer_name, optimizer)
            self.metric.update_epoch_timer(epoch)
            validation_loss = self.validate_epoch(model, model_name, layer_name, epoch)
            self._callback.on_epoch_end()
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=validation_loss)

        self._callback.on_end()

        # save
        if self.rank == 0 and self.save_if_need(model_name, layer_name, step,
                                                self.trainer_spec.epochs(),
                                                last_epoch=True):
            logger.info(f"Saving {self.trainer_spec.epochs()} last epoch.")
        self._callback.on_epoch_end()
        return step

    def train(self, model_name=None, config=None, checkpoint_dir=None):
        """
        :param config:
        :param model_name:
        :param  checkpoint_dir this mainly for ray
        :return:
        """
        # torch.manual_seed(self.model_spec.seed())
        # torch.cuda.manual_seed(self.model_spec.seed())
        #
        # if self.rank == 0:
        #     self.load_models()
        # if self.trainer_spec.is_distributed_run():
        #     #torch.cuda.set_device(self.device)
        print("################# Started")

        torch.cuda.empty_cache()
        model_name = self.trainer_spec.get_active_mode()
        model_layers = self.trainer_spec.get_active_sub_models()

        if self.is_trained(model_name):
            print("It looks like model already trained. \
                   Check file {}".format(self.trainer_spec.model_files.get_model_file_path(model_name)))
            return

        # last_step = 0
        strategy = self.trainer_spec.get_training_strategy(model_name)
        if strategy == 'sequential':
            for layer in model_layers:
                self.q.append(layer)
            while len(self.q) > 0:

                layer_name = self.q.pop()
                # update whatever we need
                if config is not None:
                    if 'batch_size' in config:
                        self._dataloaders.update(config['batch_size'])
                        # self.data_loader.update_batch(int(config["batch_size"]))
                        # self._train_loader[self._tkey].update_batch(config["batch_size"])
                        # self._validation_loader[self._tkey].update_batch(config["batch_size"])
                        # self._train_loader.up
                        # self.train_loader, self.validation_loader, self.collate_fn = self.data_loader.get_loader()
                    if 'lr' in config:
                        for param_group in self._optimizers[model_name][layer_name].param_groups:
                            param_group['lr'] = config["lr"]
                # run model
                last_step = self.sequential(model_name, layer_name, checkpoint_dir=checkpoint_dir)
                if self.rank == 0 and self.save_if_need(model_name=model_name,
                                                        layer_name=layer_name, epoch=self.trainer_spec.epochs(),
                                                        step=last_step,
                                                        last_epoch=True):
                    if self.verbose:
                        fmtl_print("Saved last epoch", self.trainer_spec.epochs())

        self.cleanup()

    def is_trained(self, model_name: str) -> bool:
        """
        Method check if specific model trained or not.
        # TODO refactor epochs per layer
        :param model_name:
        :return:
        """
        num_finished = 0
        if model_name not in self._last_ckt_epochs:
            return False

        layers = self.trainer_spec.get_active_sub_models()
        for layer in layers:
            if layer in self._last_ckt_epochs[model_name]:
                last = self._last_ckt_epochs[model_name][layer]
                if int(last) >= int(self.trainer_spec.epochs()):
                    num_finished += 1

        if num_finished == len(layers):
            return True

        return False

    def get_model(self, model_name):
        """
        :param model_name:
        :return:
        """
        if model_name in self._models:
            return self._models[model_name]

    @staticmethod
    def set_logger(is_enable: bool) -> None:
        """
        Sets logging level.
        :param is_enable:
        :return:
        """
        if is_enable:
            logger.enable(__name__)
        else:
            logger.disable(__name__)

    @staticmethod
    def log_param_if_needed(t_writer, epoch: int, model: nn.Module) -> None:
        """
        Log model hyper parameters if needed.

        :param t_writer:
        :param epoch:
        :param model:
        :return:
        """
        for name, weight in model.named_parameters():
            t_writer.add_histogram(name, weight, epoch)
            t_writer.add_histogram(f'{name}.grad', weight.grad, epoch)
        # tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

    # def backup_model(self, model_file):
    #     """
    #
    #     :param model_file:
    #     :return:
    #     """
    #     from datetime import datetime
    #     dateTimeObj = datetime.now()
    #     copyfile.copy(model_file, f"{dateTimeObj.year}{dateTimeObj.month}{dateTimeObj.day}_{model_file}_")
    #

#
# def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
#
#     data_dir = os.path.abspath("./data")
#
#     load_data(data_dir)
#     dataloader = SFTFDataloader(spec, rank=cmd_args.rank, world_size=cmd_args.world_size, verbose=args.verbose)
#
#     config = {
#         "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([2, 4, 8, 16])
#     }
#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2)
#
#     reporter = CLIReporter(
#         # parameter_columns=["l1", "l2", "lr", "batch_size"],
#         metric_columns=["loss", "accuracy", "training_iteration"])
#
#     result = tune.run(
#         partial(train_cifar, data_dir=data_dir),
#         resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter)
#
#     best_trial = result.get_best_trial("loss", "min", "last")
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(
#         best_trial.last_result["loss"]))
#     print("Best trial final validation accuracy: {}".format(
#         best_trial.last_result["accuracy"]))
#
#     best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if gpus_per_trial > 1:
#             best_trained_model = nn.DataParallel(best_trained_model)
#     best_trained_model.to(device)
#
#     best_checkpoint_dir = best_trial.checkpoint.value
#     model_state, optimizer_state = torch.load(os.path.join(
#         best_checkpoint_dir, "checkpoint"))
#     best_trained_model.load_state_dict(model_state)
#
#     test_acc = test_accuracy(best_trained_model, device)
#     print("Best trial test set accuracy: {}".format(test_acc))
