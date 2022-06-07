# Main trainer logic. ,
# The main idea we de-couple a trainer state,
# model specific trainer logic and model specification.
#
# In current implementation trainer has own dispatcher to factory
# method. later I'll move that out and trainer logic will completely
# de-couple.  I.e a factory method will create model, all optimizers.
# loss function , logic related to loging and metric.
# Each model specific.
#
#
# Mustafa. B
#
#
# import torch.distributed as dist
# import queue
import random
from abc import ABC
# from frozendict import frozendict
from collections import defaultdict
# from multiprocessing import Queue
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pylab as plt
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

from model_loader.stft_dataloader import SFTFDataloader
from model_trainer.distributed_wrapper import DistributedDataWrapper
from model_trainer.internal.abstract_trainer import TrainerError, AbstractTrainer
from model_trainer.internal.call_interface import BaseCallbacks
from model_trainer.internal.const import ReduceMode, MetricType
from model_trainer.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from model_trainer.trainer_logger import TensorboardTrainerLogger
from model_trainer.trainer_metrics import Metrics
from model_trainer.trainer_specs import ExperimentSpecs
from model_trainer.utils import fmtl_print, to_gpu
# from distributed import apply_gradient_allreduce
from models.loss_function import Tacotron2Loss
from models.dtc_loss_function import dtcLoss
from models.tacatronv30.model import Tacotron3
from models.tacotronv25.model import Tacotron25
from text import text_to_sequence


# from torch.nn.parallel import DistributedDataParallel
# from torch.autograd import Variable
# import numpy as np

# try:
#     import ray
#     from ray import tune
#     from ray.tune.schedulers import ASHAScheduler
# except ImportError:
#     logger.info("ray not found")
#     pass


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
                 callback: Optional[list[Callable]] = None,
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

        # unit testing flags.
        self._brake_epoch_loop = 1
        #
        self._brake_epoch_train = False

        # read do we run auto precision mode or not
        self.state.is_amp = trainer_spec.is_amp()

        if not self.state.is_inference:
            # set a batch size from spec.
            self.state.batch_size = data_loader.get_batch_size()
            assert self.state.batch_size > 0

        # this will be to model specific trainer.
        self.model_spec = trainer_spec.get_model_spec()
        self.spectogram_spec = self.model_spec.get_spectrogram()

        # end unit test.
        if config is not None:
            self.hyper_config = config

        # if we run hyperparameter tunner, ray need checkpoint dir.
        self.state.is_hyper_tunner = hp_tunner
        # ray config
        self.checkpoint_dir = checkpoint_dir

        self.set_logger(verbose)

        # dict that hold all schedulers that trainer need to use. TODO This will be moved.
        self._schedulers = {}

        # dict hold all optimizers. ( note trainer can train 2 model like in gan settings)
        self._optimizers = {}

        if not is_inference:
            if not trainer_spec.is_initialized():
                raise TrainerError("you need initialize trainer specs first.")

        # if self.state.trainer_spec.is_distributed_run():
        #     self.init_distributed()

        self._tkey = self.state.trainer_spec.get_default_train_set_key()
        self._vkey = self.state.trainer_spec.get_default_val_set_key()

        if not self.state.is_inference:
            if data_loader is None:
                raise TrainerError("Trainer need torch data loader.")
            self.state.data_loader = data_loader
            self.state.data_loaders, self.state.collate_fn = self.state.data_loader.get_all()

            self._train_loader = self.state.data_loaders[self._tkey]
            self._validation_loader = self.state.data_loaders[self._vkey]

            if len(self._train_loader) == 0:
                print("dataset size", data_loader.get_train_dataset_size())

        self.state.tbar_update_rate = self.state.trainer_spec.console_log_rate()
        assert self.state.tbar_update_rate > 0

        # initialized based on a model
        self.criterion = None

        # dict store all model
        self._models = {}

        # store last epoch
        self._last_ckt_epochs: dict[str, dict[str, int]] = {}

        # dict holds model name = last iterator value
        self._last_step: dict[str, int] = {}

        #  torch.cuda.amp.GradScaler for automatic mixed precision
        self.scaler = None

        # tqdm_iter, if we need fix post or pre or update state.
        # only rank 0 upate that.
        self.tqdm_iter = None

        self.clip_grad = False
        self.total_batches = 0
        if self.state.is_inference is False:
            # total batches
            self.total_batches = len(self.state.data_loaders[self._tkey])
            # clip or not grad
            self.clip_grad = trainer_spec.is_grad_clipped()

            if self.state.is_hyper_tunner is False:
                # by default, we log model name currently trainer and batch size.
                precision = "fp32" if not self.state.is_amp else "fp16"
                self.tf_logger = TensorboardTrainerLogger(trainer_spec=trainer_spec,
                                                          model_name=trainer_spec.get_active_model(),
                                                          batch_size=trainer_spec.batch_size(),
                                                          precision=precision,
                                                          comments=f"{trainer_spec.get_active_model()}_"
                                                                   f"{trainer_spec.batch_size()}")

            self.metric = Metrics(metric_step_file_path=trainer_spec.model_files.get_metric_file_path(),
                                  metric_batch_file_path=trainer_spec.model_files.get_metric_batch_file_path(),
                                  metric_perf_trace_path=trainer_spec.model_files.get_time_file_path(),
                                  num_epochs=self.state.trainer_spec.epochs(),
                                  num_batches=self.total_batches,
                                  batch_size=self.state.trainer_spec.batch_size(),
                                  metric_type=MetricType.MEAN,
                                  mode=ReduceMode.MIN,
                                  verbose=False)

            self._callback = BaseCallbacks(callbacks=callback)
            self._callback.register_trainer(self)
            self._callback.register_metric(self.metric)

            # self.callbacks.set
            logger.info("Device ID {}".format(self.state.cuda_device_id))
            # main queue note that train loop can re-add model back for training.

            # depend on a strategy how we train models.
            # in sequential case, train model my_model
            # than sub-model of model my_model
            self.q = deque()

        # late we will move this to factory
        # type class that will do all creation.
        self.model_creator, self.trainer_dispatcher, self._batch_loader = self.create_model_dispatch()

        # init trainer
        self.init_trainer()

    def get_models(self):
        return self._models

    def create_model_dispatch(self) -> tuple[dict[str, dict[str: Callable]],
                                             dict[str, Callable],
                                             dict[str, dict[str: Callable]]]:
        """
        @TODO for now we keep it here, it will move to Model Creator.

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
                'vocoder': self.create_tacotron25,
                'loss': Tacotron2Loss,
            },
            'dtc': {
                'spectrogram_layer': self.create_tacotron30,
                'vocoder': self.create_tacotron25,
                'loss': dtcLoss,
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
            'dtc': {
                'spectrogram_layer': self.tacotron30_batch,
            }
        }

        return model_dispatch, trainer_dispatch, batch_loader

    def init_trainer(self) -> None:
        """
        :return:
        """
        if self.state.is_inference:
            self.create_models()
            print("Model in inference state.")
            return

        if self.state.trainer_spec.is_distributed_run():
            torch.manual_seed(self.state.trainer_spec.seed())
            torch.cuda.manual_seed(self.state.trainer_spec.seed())
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(self.state.trainer_spec.seed())
            random.seed(self.state.trainer_spec.seed())

            logger.info("Starting distributed DDP training.")
            # self.init_distributed()
            dist.barrier()

        self.create_models()
        self.create_optimizers()
        self.create_lr_schedulers()

        # amp scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.state.is_amp)

    def create_tacotron25(self, is_set_cuda=False) -> nn.Module:
        """
        Factor method, creator for tacotron 2.5.
        TODO this one will be moved out of trainer.
        So we will have clear separation between trainer state , creation
        etc.
        :return:
        """
        logger.debug("Creating tacotron25 model.")
        if self.state.is_inference:
            model = Tacotron25(self.state.trainer_spec, self.state.device).to(self.state.device)
        elif self.state.trainer_spec.is_distributed_run():
            # device = torch.device(f"cuda:{dist.get_rank()}")
            device = self._loop_up_device(is_set_cuda)
            logger.info("Creating DDP on cuda device "
                        "{} torch device {} device received {}".format(self.cuda_device_id, device, self.state.device))
            model = Tacotron25(self.state.trainer_spec, device).cuda()
            model = DistributedDataWrapper(model,
                                           device_ids=[self.cuda_device_id],
                                           output_device=self.cuda_device_id).cuda()

        else:
            model = Tacotron25(self.state.trainer_spec, self.state.device).to(self.state.device)

        if self.state.trainer_spec.is_amp():
            model.decoder.attention_layer.score_mask_value = finfo('float16').min

        return model

    def create_tacotron30(self, is_set_cuda=False) -> nn.Module:
        """
        Factor method, creator for tacotron 3
        :return:
        """
        logger.debug("Creating tacotron30 model.")
        if self.state.is_inference:
            model = Tacotron3(self.state.trainer_spec, self.state.device).to(self.state.device)
        elif self.state.trainer_spec.is_distributed_run():
            device = self._loop_up_device(is_set_cuda)
            logger.info("Creating DDP on cuda device "
                        "{} torch device {} device received {}".format(self.cuda_device_id,
                                                                       device, self.state.device))
            model = Tacotron3(self.state.trainer_spec, device).cuda()
            model = DistributedDataWrapper(model,
                                           device_ids=[self.cuda_device_id],
                                           output_device=self.cuda_device_id).cuda()

        else:
            model = Tacotron3(self.state.trainer_spec, self.state.device).to(self.state.device)

        if self.state.trainer_spec.is_amp():
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
        logger.info(f"Creating a model {model_name} layer: {layer_name}.")
        if model_name not in self.model_creator:
            raise ValueError("Unknown {} model for a "
                             "trainer, supported"
                             " models are {}".format(model_name, list(self.model_creator.keys())))

        creator = self.model_creator[model_name][layer_name]
        if model_name not in self._models:
            self._models[model_name] = {}

        m = creator(is_set_cuda)
        if m is None:
            raise TrainerError("Failed create a model. {}.".format(creator.__name__))

        self._models[model_name][layer_name] = m
        logger.debug(f"Added model layer {layer_name} to a model {model_name}")
        if model_name not in self._last_ckt_epochs:
            self._last_ckt_epochs[model_name] = {}

        # update last epoch and step, initially it 0, during load this state update.
        self._last_ckt_epochs[model_name][layer_name] = 0
        self._last_step[layer_name] = 0

        self.criterion = self.model_creator[model_name]['loss']

    def create_models(self):
        """
        For each active model, which might consist two or more models.
        Each model store in internal dict.

        :return: nothing, create_model will add each model to a dict[model_name] = model
        """
        active_model = self.state.trainer_spec.get_active_model()
        _models = self.state.trainer_spec.get_active_sub_models()
        for m in _models:
            self._create_model_layers(active_model, m)

    # def load_models(self, to_device=True, ignore_layers=None):
    #     """
    #     Load al models.
    #     :return:
    #     """
    #     if self.state.rank > 0:
    #         print(f"Skipping loading. node rank {self.rank}")
    #
    #     for model in self._models:
    #         for layer in self._models[model]:
    #             self.load(model_name=model, layer_name=layer, to_device=to_device, ignore_layers=ignore_layers)

    def load_model_layer(self, layer_name: str, file_path: str):
        """
        Loads specific mode layer from model.
        :param layer_name:
        :param file_path:
        :return:
        """
        if self.state.rank > 0:
            print(f"Skipping loading. node rank {self.rank}")

        for model in self._models:
            for layer in self._models[model]:
                if layer_name == layer:
                    self._load_model(model_name=model, layer_name=layer,
                                     file_path=file_path, to_device=True, ignore_layers=None)

    def load(self):
        """
        Load all model and model layer to internal state.
        This method external hence caller can load and save
        a state of trainer.

        :return:
        """
        if self.state.rank > 0:
            print(f"Skipping loading. node rank {self.rank}")

        for m in self._models:
            for layer in self._models[m]:
                model_file = self.state.trainer_spec.model_files.get_model_file_path(layer)
                self._load_model(model_name=m, layer_name=layer, file_path=model_file)

    def _load_model(self, model_name: str, layer_name: str, to_device=True,
                    ignore_layers=None, file_path=None, skip_opt_state=False,
                    strict=False):
        """
        Method loads model from a checkpoint.  It will update internal state,
        that include model, last epoch, last step.

        :param model_name: - mode name that method need to load
        :param layer_name: - layer of model that we need load.
        :param file_path:  - path to a particular model file.
        :param ignore_layers: - a list contain of layers we skip
        :param to_device: - if true will load to device indicate as main device
        :param skip_opt_state:  by default we always load optimizer, if skip we skip
        :return: epoch, step
        """
        # setting in config, if set don't load trainer won't load model
        last_epoch = 0
        last_step = 0

        if not self.state.trainer_spec.is_load_model():
            print("Configuration set to skip loading models.")
            return last_epoch, last_step

        # by default, we use layer_name
        if file_path is None:
            p = Path(self.state.trainer_spec.model_files.get_model_file_path(layer_name))
        else:
            p = Path(file_path).expanduser()

        resolved_path = p.resolve()
        if not resolved_path.exists():
            self._last_ckt_epochs[model_name][layer_name] = last_epoch
            self._last_step[layer_name] = last_step
            return False
        if not resolved_path.is_file():
            raise TrainerError(f"Provided path is not a file {file_path}.")
        resolved_path = str(resolved_path)

        print(f"Loading model node rank {self.state.rank} '{model_name}' "
              f"model {layer_name} fro file {resolved_path}.")

        logger.info(f"Loading model node rank {self.state.rank} "
                    f"'{model_name}' model {layer_name} fro file {resolved_path}.")

        # original saved file with DataParallel
        # state_dict = torch.load('myfile.pth.tar')
        # create new OrderedDict that does not contain `module.`

        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict)

        # load trained optimizer state_dict
        try:
            if to_device:
                if self.state.trainer_spec.is_distributed_run():
                    self._models[model_name][layer_name].to(self.state.rank)
                else:
                    if {self.state.device} == "cpu":
                        print(f"Sending model to {self.state.device} device.")
                    else:
                        print(f"Sending model to {self.state.device} device,"
                              f" cuda device id {self.state.cuda_device_id}.")
                    self._models[model_name][layer_name].to(self.state.device)

            if to_device:
                print(f"Loading checkpoint to device and map location {self.state.device}")
                checkpoint = torch.load(resolved_path, map_location=self.state.device)
            else:
                print(f"Loading checkpoint from {resolved_path}.")
                checkpoint = torch.load(resolved_path)

            if 'model_state_dict' not in checkpoint:
                raise TrainerError(f"{model_name} layer {layer_name} has no state dict.")

            if model_name not in self._models:
                raise TrainerError(f"{model_name} not created. You need first create model.")

            if layer_name not in self._models[model_name]:
                raise TrainerError(f"{layer_name} not in model create.")

            self._models[model_name][layer_name].load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'model_state_dict' not in checkpoint:
                raise TrainerError("model has no state dict.")

            if not skip_opt_state:
                logger.debug(f"Loading optimizer state for model {model_name} layer {layer_name}.")
                if 'optimizer_state_dict' in checkpoint:
                    self._optimizers[model_name][layer_name].load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    if strict:
                        raise TrainerError(f"{model_name} has no optimizer_state_dict.")

            # if model has scheduler, re-create.
            if model_name in self._schedulers:
                if layer_name in self._schedulers[model_name]:
                    self._schedulers[model_name].load_state_dict(checkpoint['scheduler_state_dict'])
                    if 'scheduler_state_dict' not in checkpoint:
                        raise TrainerError("model has no scheduler_state_dict")

            # if run hyperparameter tunner we never do resume.
            if self.state.is_hyper_tunner:
                print(f"Hyperparameter tunner running, loading only states.")
                self._last_ckt_epochs[model_name][layer_name] = 0
                self._last_step[layer_name] = 0
                return 0, 0

            #  ignore_layers = 'reverse_decoder'
            #  ignore layers.
            if ignore_layers is not None and len(ignore_layers) > 0:
                model_dict = {k: v for k, v in self._models[model_name].items()
                              if k not in ignore_layers}
                new_state = self._models[model_name].state_dict()
                new_state.update(model_dict)
                self._models[model_name] = new_state

            # self.state.trainer_spec.set_lr(0.00001)
            if 'epoch' not in checkpoint:
                raise TrainerError("saved checkpoint has no epoch.")

            if model_name not in self._last_ckt_epochs:
                self._last_ckt_epochs[model_name] = {}
            self._last_ckt_epochs[model_name][layer_name] = checkpoint['epoch']

            # load last iterator.
            if 'it' not in checkpoint:
                raise TrainerError("saved checkpoint has no last iteration key.")

            if checkpoint['epoch'] is not None:
                last_epoch = checkpoint['epoch']

            if checkpoint['it'] is not None:
                last_step = checkpoint['it']

            self._last_ckt_epochs[model_name][layer_name] = last_epoch
            self._last_step[layer_name] = last_step

            print(f"Last checkpoint: {last_epoch}, step last step {last_step}.")
            logger.info(f"Last checkpoint: {last_epoch}, step last step {last_step}.")

            if last_epoch > 0 or last_step > 0:
                print(f"Resuming training from from epoch {last_epoch}, step {last_step}.")
                logger.info(f"Resuming training from epoch {last_epoch}, step {last_step}.")

            # load metric
            self.metric.load()
            return last_epoch, last_step

        except FileNotFoundError as e:
            print("Failed load model files {}. No saved model found.".format(file_path))
            logger.error(f"No model file to load model file, return default epoch 0, iteration 0 {e}")
            logger.error(e)

        return last_epoch, last_step

    def load_for_inference(self, model_name: str, layer_name: str, model_file=None, to_device=True, ignore_layers=None):
        """
        Method loads model from a checkpoint for inference.

        :param model_name: - mode name that method need to load.
        :param layer_name: - model layer we need load.
        :param model_file: - torch model file.
        :param ignore_layers: - a list contain of layers we skip
        :param to_device: - if true will load to device indicate as main device
        :return: epoch, step
        """

        logger.info("Loading model '{}' "
                    "model file name {} to device {}".format(model_name, model_file, self.state.device))

        try:
            self.state.current_layer = layer_name
            self.state.current_model = model_name

            if model_name not in self._models:
                self._models = {model_name: {}}

            checkpoint = torch.load(model_file, map_location=self.state.device)
            if 'model_state_dict' not in checkpoint:
                raise TrainerError("model has no state dict")

            creator = self.model_creator[model_name][layer_name]
            m = creator(False)

            # ignore_layers = ['embedding.weight']
            model_state_dict = checkpoint['model_state_dict']
            if ignore_layers is not None and len(ignore_layers) > 0:
                model_dict = {k: v for k, v in model_state_dict.items()
                              if k not in ignore_layers}
                new_state = m.state_dict()
                new_state.update(model_dict)
                model_state_dict = new_state

            self._models[model_name][layer_name].load_state_dict(model_state_dict, strict=False)
            if 'model_state_dict' not in checkpoint:
                raise TrainerError("model has no state dict.")

            # print(self._models[model_name][layer_name]['model'].item())

            # if ignore_layers is not None and len(ignore_layers) > 0:
            #     model_dict = {k: v for k, v in self._models[model_name][layer_name].items()
            #                   if k not in ignore_layers}
            #     new_state = self._models[model_name][layer_name].state_dict()
            #     new_state.update(model_dict)
            #     self._models[model_name][layer_name] = new_state

            if 'epoch' not in checkpoint:
                raise TrainerError("Saved checkpoint has no epoch key.")
            self._last_ckt_epochs[model_name][layer_name] = checkpoint['epoch']

            # load last iterator.
            if 'it' not in checkpoint:
                raise TrainerError("Saved checkpoint has no last step key.")

            self._last_step[layer_name] = checkpoint['it']
            logger.info("Last checkpoint. epoch {} step {}".format(checkpoint['epoch'], checkpoint['it']))

            self.state.current_model = self._models[model_name][layer_name]
            print(f"Last trained epoch {checkpoint['epoch']}, {checkpoint['it']}")

            #  m = torch.jit.script(m)
            # self._models[model_name][layer_name] = torch.jit.freeze(m, ["version"])
            self._models[model_name][layer_name] = m
            return checkpoint['epoch'], checkpoint['it']

        except FileNotFoundError as e:
            print("Failed load model files {}. No saved model found. {}".format(model_file, e))
            logger.info("No model file to load model file, return default epoch 0, iteration 0")

        return 0, 0

    def trainer_iterator(self, model_name: str, layer_name: str, last_epoch=0, max_epochs=0):
        """
        Method return iterator, i.e if you need compute, offset , update tqdm etc.
        :param model_name:
        :param layer_name:
        :param last_epoch:
        :param max_epochs:
        :return:
        """
        # for notebook, we need a bit of hack for tqdm_iter.
        if self.state.is_notebook:
            tqdm_iter = tnrange(last_epoch, max_epochs)
            return tqdm_iter

        # load last epoch in case we do re-summing.
        last_epoch = self._last_ckt_epochs[model_name][layer_name]
        # early stopping
        early_stopping = None

        max_epochs = self.state.trainer_spec.epochs()
        # what tqdm to use notebook for colab or normal one.
        logger.info(f"Creating tqdm last epoch {last_epoch} max epoch {max_epochs}")

        if self.state.is_notebook:
            tqdm_iter = tnrange(last_epoch, max_epochs)
        else:
            tqdm_iter = tqdm(range(last_epoch, max_epochs),
                             desc=f"Training in progress, device {self.state.device}, "
                                  f"batch size {self.state.batch_size}",
                             disable=self.state.disable_pbar)

        # tqdm_iter.set_postfix({'total_epoch_loss': 0})
        return tqdm_iter

    # def load_models(self) -> None:
    #     """
    #     For each active model, which might consist of more than one model.
    #     :return:
    #     """
    #     models = self.state.trainer_spec.get_active_sub_models()
    #     for m in models:
    #         self.load(m)

    def reduce_if_needed(self, loss):
        """
        TODO need proper test FP16
        :param loss:
        :return:
        """
        if self.state.trainer_spec.is_amp():
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
            raise TrainerError(f"{str(self.state.trainer_spec.config_file_name)} "
                               f"must contains valid binding for optimizer. "
                               f"Failed create '{model_name}' model")

        logger.info(f"Creating optimizer for {model_name} layer {layer_name} "
                    f"and optimizer type {self.state.trainer_spec.optimizer_type(alias_name)}")

        model_layer = self._models[model_name][layer_name]

        optimizer_type = self.state.trainer_spec.optimizer_type(alias_name)
        spec = self.state.trainer_spec

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
        elif self.state.trainer_spec.optimizer_type == 'none':
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
        model_name = self.state.trainer_spec.get_active_model()
        model_layers = self.state.trainer_spec.get_active_sub_models()
        if len(model_layers) == 0:
            logger.warning("Trainer spec has no model layer defined..")

        for model_layer_name in model_layers:
            logger.debug(f"Loading {model_layer_name} optimizer settings.")
            opt_spec_alias = self.state.trainer_spec.get_sub_model_optimizer(model_layer_name)
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
        alias_name = self.state.trainer_spec.get_sub_model_lr_scheduler(model_layer)
        if len(alias_name) == 0:
            logger.info(f"Model {model_layer} layer {model_layer} no scheduler attached.")
            return

        if optimizer is None:
            raise TrainerError("Can't create lr scheduler. Optimizer is None.")

        lr_scheduler_type = self.state.trainer_spec.lr_scheduler_type(alias_name)
        if lr_scheduler_type == 'cos':
            logger.info(f"Creating cos lr scheduler. model: {model_name} layer: {model_layer} spec: {alias_name}")
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=self.state.trainer_spec.t_max(alias_name),
                                                       eta_min=self.state.trainer_spec.eta_min(alias_name))
        elif lr_scheduler_type == 'multistep':
            logger.info(f"Creating multistep lr scheduler. model: {model_name} "
                        f"layer: {model_layer} spec: {alias_name}")
            logger.info(f"Creating {self.state.trainer_spec.milestones(alias_name)} milestone.")
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=self.state.trainer_spec.milestones(alias_name),
                                                 gamma=self.state.trainer_spec.gamma(alias_name))
        elif lr_scheduler_type == 'exp-warmup':
            logger.info(f"Creating exp-warmup lr scheduler. model: {model_name} "
                        f"layer: {model_layer} spec: {alias_name}")
            lr_lambdas = self.state.trainer_spec.lr_lambdas(alias_name)
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
        model_name = self.state.trainer_spec.get_active_model_name()
        if model_name not in self._optimizers:
            raise TrainerError(f"Model {model_name} must be created first.")

        # model_layers = self.state.trainer_spec.get_active_sub_models()
        # for layer in model_layers:
        #     if layer not in self._optimizers[model_name]:
        #         raise TrainerError(f"Model {model_name}, model layer {layer} must contain attached optimizer.")
        #     opt = self._optimizers[model_name][layer]
        #     logger.debug("Attaching lr scheduler.")
        #     self.create_lr_scheduler(model_name, layer, opt)

    def save(self) -> None:
        """
        :return:
        """
        if self.state.rank > 0:
            logger.info(f"Skipping saving, node rank {self.state.rank}")
            return

        if self.state.is_hyper_tunner:
            logger.info(f"Skipping saving, hyperparameter tunner.")
            return

        for m in self._models:
            for layer in self._models[m]:
                model_file = self.state.trainer_spec.model_files.get_model_file_path(layer)
                if not self.state.disable_pbar:
                    self.tqdm_iter.set_description(f"Saving in progress, {self.state.device}")
                self._save_model(model_name=m, layer_name=layer, file_path=model_file)

    def save_models(self, model_files: list[str], step=0):
        """
        Save's all models.
        :param model_files:
        :param step:
        :return:
        """
        for m in self._models:
            if len(self._models[m]) == len(model_files):
                for i, layer in enumerate(self._models[m]):
                    self.save_model(model_name=m, layer_name=layer,
                                    model_file=model_files[i], tep=step)
            else:
                raise TrainerError(f"Model contains {len(self._models[m])} layers, "
                                   f"each sub-layer must have must have one file per model.")

    def save_model_layer(self, layer_name: str, file_path: str):
        """
        Save's specific model's layer to a file_path.

        :param layer_name:
        :param file_path:
        :return:
        """
        for m in self._models:
            for layer in self._models[m]:
                if layer_name == layer:
                    self._save_model(model_name=m, layer_name=layer, file_path=file_path)

    def _save_model(self, model_name: str, layer_name: str, file_path: str,
                    epoch: Optional[int] = None,
                    step: Optional[int] = None) -> None:
        """

        :param model_name: a mode saving
        :param layer_name: a layer of model that , trainer saving
        :param file_path:  a full path to a model
        :param epoch: if we need save epoch
        :param step:  if we need save last step
        :return: nothing
        """

        last_epoch = self.state.epoch
        last_step = self.state.step

        if step is not None:
            last_step = step

        if model_name is None or len(model_name) == 0:
            raise TrainerError(f"Can't save model, model name argument is empty.")
        if layer_name is None or len(layer_name) == 0:
            raise TrainerError(f"Can't save model {model_name}, layer name argument is empty.")
        if file_path is None or len(file_path) == 0:
            raise TrainerError(f"Can't save model {model_name}, "
                               f"layer name {layer_name} argument name is empty.")

        if epoch is not None:
            last_epoch = epoch

        p = Path(file_path).expanduser()
        if p.is_dir():
            raise TrainerError("Invalid file_path.")

        resolved_path = str(p.resolve())
        logger.info('Saving node model {}'.format(resolved_path))
        self._callback.saving_start()

        is_saved = False
        # TODO for not dirty fix/ need check no ops for scaled save, because I need store per mode layer.
        if self.state.is_amp:
            # print(f"Saving model and auto precision state with scheduler state, "
            #       f"last epoch {last_epoch}, last step {last_step}, model file {resolved_path}.")
            logger.info(f"Saving model and auto precision state with, "
                        f"last epoch {last_epoch}, last step {last_step}, model file {resolved_path}.")
            if model_name in self._schedulers:
                logger.info("Model saving with optimizer and scheduler state.")
                torch.save({'epoch': last_epoch, 'it': last_step,
                            'model_state_dict': self._models[model_name][layer_name].state_dict(),
                            'optimizer_state_dict': self._optimizers[model_name][layer_name].state_dict(),
                            'scheduler_state_dict': self._schedulers[model_name].state_dict(),
                            'scaler': self.scaler.state_dict(),
                            'model_name': self.state.model_name,
                            }, resolved_path)
                is_saved = True

            else:
                # print(f"Saving model and auto precision state without scheduler state, "
                #       f"last epoch {last_epoch}, last step {last_step}, model file {resolved_path}.")
                logger.info(f"Saving model with without scheduler state, "
                            f"last epoch {last_epoch}, last step {last_step}, model file {resolved_path}.")
                logger.info("Model saving with optimizer without a scheduler state.")
                torch.save({'epoch': last_epoch, 'it': last_step,
                            'model_state_dict': self._models[model_name][layer_name].state_dict(),
                            'optimizer_state_dict': self._optimizers[model_name][layer_name].state_dict(),
                            'scaler': self.scaler.state_dict(),
                            'model_name': self.state.model_name,
                            }, resolved_path)
                is_saved = True

        else:
            if model_name in self._schedulers:
                # print(f"Saving model with scheduler state, "
                #       f"last epoch {last_epoch}, last step {last_step}, model file {resolved_path}.")
                logger.info(f"Saving model with scheduler state, "
                            f"last epoch {last_epoch}, last step {last_step}, model file {resolved_path}.")
                torch.save({'epoch': last_epoch, 'it': last_step,
                            'model_state_dict': self._models[model_name][layer_name].state_dict(),
                            'optimizer_state_dict': self._optimizers[model_name][layer_name].state_dict(),
                            'scheduler_state_dict': self._schedulers[model_name].state_dict()
                            }, resolved_path)
                is_saved = True

            else:

                # print(f"Saving without scheduler state, "
                #       f"last epoch {last_epoch}, last step {last_step}")
                # print(f"Model file {resolved_path}.")

                logger.info(f"Saving without scheduler state, "
                            f"last epoch {last_epoch}, last step {last_step}, model file {resolved_path}.")
                torch.save({'epoch': last_epoch, 'it': last_step,
                            'model_state_dict': self._models[model_name][layer_name].state_dict(),
                            'optimizer_state_dict': self._optimizers[model_name][layer_name].state_dict(),
                            }, resolved_path)
                is_saved = True

        if is_saved:
            self.state.saved_run = last_step
            self._last_ckt_epochs[model_name][layer_name] = last_epoch

    def save_if_need(self, step: int) -> bool:
        """
        Method called by trainer, to check if model 
        or model layer need to save or not.
        """
        if self.state.is_hyper_tunner:
            print("Hyperparameter tunner, skipping saving.")
            return False

        if self.state.trainer_spec.is_save_required() is False:
            return False

        # by default condition to save epoch , if save per iteration we check iteration.
        if step == 0:
            return False

        # in case we run distributed no need save.
        if self.state.trainer_spec.is_distributed_run() and self.rank > 0:
            return False

        # model save predicate condition, either iteration or epoch counter.
        # for large model it makes sense to track iteration vs epoch counter.
        save_condition = self.state.epoch if self.state.trainer_spec.is_save_iteration() else self.state.step
        if save_condition == 0 or save_condition == self.state.saved_run:
            return False

        # do nothing
        if save_condition % self.state.trainer_spec.epochs_save() == 0:
            self.save()
            self.state.saved_run = save_condition
            self._callback.saved()
            # save metrics
            self.metric.save()
            return True

        return False

    def validate_epoch(self, model: nn.Module, model_name: str,
                       layer_name: str, step: Optional[int] = None,
                       warmup: Optional[bool] = False):
        """
        Validation epoch

        :param model:  model we are training.
        :param model_name:  model_name a model we are training.
        :param layer_name:  layer in that model we are training.
        :param step: optional,  if we do validation in some n step before
                                end of epoch, we want log and track.
        :param warmup:
        :return:
        """
        # take a batch.
        self._callback.validation_start()
        take_batch = self._batch_loader[model_name][layer_name]

        model.eval()
        self.metric.update_bach_estimated(len(self._validation_loader))
        self.metric.on_prediction_batch_start()
        if warmup:
            self.tqdm_iter.set_description(f"Warm up in progress, {self.state.device}")
        else:
            self.tqdm_iter.set_description(f"Validation in progress, {self.state.device}")

        count = 0
        aggregated_loss_term = defaultdict(float)
        with torch.no_grad():
            current_batch_size = len(self._validation_loader)
            for batch_idx, batch in enumerate(self._validation_loader):
                x, y = take_batch(batch, self.state.device)
                y_pred = model(x)
                # our loss mel_loss + gate_loss
                criterion_out = self.criterion(y_pred, y)

                all_reduced_loss = {}
                for loss_term_key in criterion_out:
                    loss_tensor = criterion_out[loss_term_key]
                    if self.state.trainer_spec.is_distributed_run():
                        all_reduced_loss[loss_term_key] = self.split_tensor(loss_tensor.data, self.state.n_gpus).item()
                    else:
                        if isinstance(loss_tensor, float):
                            all_reduced_loss[loss_term_key] = loss_tensor
                        else:
                            all_reduced_loss[loss_term_key] = loss_tensor.item()

                self.metric.update(batch_idx, self.state.step, all_reduced_loss['loss'], validation=True)

                if not self.state.is_hyper_tunner:
                    if self.state.rank == 0 and batch_idx % self.state.tbar_update_rate == 0:
                        tqdm_update_dict = {'step': self.state.step,
                                            'loss': all_reduced_loss['loss'],
                                            'batch_loss': all_reduced_loss['loss'] // max(1, batch_idx + 1),
                                            'avg loss': self.metric.total_train_mean_loss(),
                                            'batch': f"{batch_idx}/{current_batch_size}",
                                            'diag': all_reduced_loss['diagonal_score'],
                                            'saved': self.state.saved_run}

                        for k in all_reduced_loss:
                            tqdm_update_dict[k] = all_reduced_loss[k]
                        self.tqdm_iter.set_postfix(tqdm_update_dict)

                # sum each loss term
                for k in all_reduced_loss:
                    aggregated_loss_term[k] += all_reduced_loss[k]
                count += 1

        # normalize at the update key
        for k in aggregated_loss_term:
            aggregated_loss_term[k] = (aggregated_loss_term[k] / (batch_idx + 1))

        self._callback.validation_end()
        self.metric.on_prediction_batch_end()

        model.train()
        # update tensorboard
        if self.state.is_hyper_tunner is False and self.state.rank == 0:
            # criterions = {
            #     "train_normal_loss": normal_loss,
            #     "train_clipped_loss": grad_norm,
            #     "train_mel_loss": mel_loss.item(),
            #     "train_gate_loss": mel_loss.item(),
            #     "total_prediction_loss": total_prediction_loss,
            # }

            if count > 0:
                criterions = {
                    "loss/validation": aggregated_loss_term['loss'],
                    "validation/validation_mel_loss": aggregated_loss_term['mel_loss'],
                    "validation/validation_gate_loss": aggregated_loss_term['gate_loss'],
                }
                self.tf_logger.log_validation(criterions,
                                              model, y, y_pred, step=self.state.step)

        return aggregated_loss_term

    def update_running_state(self, model_name: str, layer_name: str) -> tuple[int, int]:
        """
         Return last iterator,  for a given model and layer.
         During a model save each model might have own iterator counter
         Hence we save epoch and current step.

         During resuming we load back epoch and current step.

         Thus, step or epoch might be re-adjusted and hence last saved epoch and step
         are not the same.

         It will re adjust based on "some" strategy.
         trainer has a dict that hold self._last_step[mode_name][layer_name] = it

        :param model_name:
        :param layer_name: model layer must be already created.
        :return: return last iterator counter.
        """
        last_epoch = 0
        last_step = 0
        # model loader will load last step to last step.
        if model_name in self._last_ckt_epochs:
            if layer_name in self._last_ckt_epochs[model_name]:
                last_epoch = self._last_ckt_epochs[model_name][layer_name]
        else:
            print("Strange case")

        if layer_name in self._last_step:
            last_step = self._last_step[layer_name]
            self.state.step = last_step
        else:
            print("Strange case two")

        return last_epoch, last_step

    def inference(self, input_seq=None, model_name='dtc', plot=True,
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
        if model_name not in self._models:
            raise TrainerError("You need load model {}".format(model_name))

        model = self.state.current_model
        model.eval()

        #  model = getattr(models,)().eval()
        # Tracing the model with example input
   #     traced_model = torch.jit.trace(model, sample_input)
        # Invoking torch.jit.freeze
    #    traced_model = torch.jit.freeze(traced_model)

        if isinstance(input_seq, str):
            sequence = np.array(text_to_sequence(input_seq, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(
                    torch.from_numpy(sequence)).to(self.state.device).long()

        # sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        with torch.no_grad():
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

    # with amp.scale_loss(loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()
    # # Now it's safe to clip.  Replace
    # # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    # # with
    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
    # # or
    # torch.nn.utils.clip_grad_value_(amp.master_params(optimizer), max_)
    #
    def rescale_gradients(self) -> float:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.

        # See: https://nvidia.github.io/apex/advanced.html#gradient-clipping
        :return: the norm of the gradients.
        """
        if self._opt_level is not None:
            parameters_to_clip = [p for p in torch.amp.master_params(self.optimizer) if p.grad is not None]
        else:
            parameters_to_clip = [p for p in self.model.parameters() if p.grad is not None]

        if self.clip_grad:
            return clip_grad_norm_(parameters_to_clip, self.state.trainer_spec.grad_clip_thresh())
        else:
            return torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters_to_clip]))

    def current_running_state(self):
        """
        :return:
        """
        return self.state.epoch, self.state.step

    @staticmethod
    def adjust_learning_rate(epoch, optimizer, learning_rate, anneal_steps, anneal_factor):
        """
        Adjusts learning rate base on the initial setting anneal step and factor.

        :param epoch:
        :param optimizer:
        :param learning_rate:
        :param anneal_steps:
        :param anneal_factor:
        :return:
        """
        p = 0
        if anneal_steps is not None:
            for _, a_step in enumerate(anneal_steps):
                if epoch >= int(a_step):
                    p = p + 1

        if anneal_factor == 0.3:
            lr = learning_rate * ((0.1 ** (p // 2)) * (1.0 if p % 2 == 0 else 0.3))
        else:
            lr = learning_rate * (anneal_factor ** p)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def train_epoch(self, model, model_name: str, layer_name: str, optimizer,
                    schedulers: Optional[list[torch.optim.lr_scheduler._LRScheduler]] = None) -> None:
        """
        Train one step of epoch.
        :param model:  a model that we train.
        :param model_name: a model name that we train
        :param layer_name: a layer of model that we train.
        :param optimizer:  optimizer that bounded to this model.
        :param schedulers: a schedulers that we use.
        :return:
        """
        device = self.state.device
        take_batch = self._batch_loader[model_name][layer_name]
        self.state.tbar_update_rate = self.state.trainer_spec.console_log_rate()

        # if self.state.trainer_spec.is_distributed_run():
        #     device = torch.device(f"cuda:{dist.get_rank()}")
        # else:
        #     device = self.state.device

        current_total_loss = 0
        current_grad_norm_loss = 0
        self.tqdm_iter.set_description(f"Training in progress, {self.state.device}")

        current_epoch, current_step = self.current_running_state()
        loader_data_size = len(self._train_loader)
        self.metric.update_bach_estimated(loader_data_size)
        self._callback.on_loader_begin()
        self.metric.on_batch_start()
        # ths main for debug for ray if something went very wrong we can check here.
        if self.state.is_hyper_tunner:
            print(f"Entering train epoch loop: "
                  f"train loader length: {len(self._train_loader)}, "
                  f"train batch size: {self._train_loader.batch_size}, "
                  f"dataset size: {self.state.data_loader.get_train_dataset_size()} "
                  f"loader batch size: {self.state.data_loader.get_batch_size()}")

        grad_norm = None
        batch_counter = 0
        for batch_idx, batch in enumerate(self._train_loader):
            self._callback.on_batch_begin()
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.state.is_amp):

                x, y = take_batch(batch, device)
                y_pred = model(x)
                # print(len(y_pred))
                # for ret in y_pred:
                #     print(ret.dtype)
                # assert ret.dtype is torch.float16
                # if self.state.trainer_spec.is_distributed_run():
                #     reduced_loss = dist.reduce_tensor(loss.data, n_gpus).item()

                criterion_out = self.criterion(y_pred, y)
                mel_loss = criterion_out['mel_loss']
                gate_loss = criterion_out['gate_loss']
                # spectral_loss = criterion_out['spectral_loss']
                loss = criterion_out['loss']
                diag_score = criterion_out['diagonal_score']
                stft_err = criterion_out['abs_error']
                assert loss.dtype is torch.float32
                normal_loss = loss.item()

            # here we have scaled loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)

            # loss.backward()
            if self.clip_grad:
                grad_norm = clip_grad_norm_(model.parameters(), self.state.trainer_spec.grad_clip_thresh())
                grad_loss = grad_norm.item()
                self.metric.update(batch_idx, current_epoch, loss=normal_loss, grad_norm=grad_loss, validation=False)
                current_total_loss += normal_loss
                current_grad_norm_loss += grad_norm
            else:
                current_total_loss += normal_loss
                self.metric.update(batch_idx, current_epoch, normal_loss, grad_norm=None, validation=False)

            self.scaler.step(optimizer)
            self.scaler.update()

            # optimizer.step()
            self._callback.on_batch_begin()
            # run lr_scheduler
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.step()

            if not self.state.is_hyper_tunner:
                if self.state.rank == 0 and current_step != 0 and current_step % self.state.tbar_update_rate == 0:
                    self.state.step = current_step
                    self.tqdm_iter.set_postfix({'step': current_step,
                                                'grad': grad_norm.item(),
                                                'b_mean': self.metric.batch_grad_loss.mean(),
                                                'loss': normal_loss,
                                                'diag': diag_score,
                                                'stft_err': stft_err.item(),
                                                'epoch': self.metric.epoch_train_gn_loss.mean(),
                                                # 'spectral_loss': spectral_loss.item(),
                                                'mel': mel_loss.item(),
                                                'gate': gate_loss.item(),
                                                'batch': f"{batch_idx}/{loader_data_size}",
                                                'lr': optimizer.param_groups[0]['lr'],
                                                'saved': self.state.saved_run})
            # run prediction only if if need to.
            if self.state.trainer_spec.predict_per_iteration():
                if current_step != 0 and current_step % self.state.trainer_spec.predict() == 0:
                    self.state.step = current_step
                    self.validate_epoch(model, model_name, layer_name)

            # save model checkpoint if needed
            # ray has issue with torch and tensorboard.
            if self.state.rank == 0 and self.state.is_hyper_tunner is False:
                self.save_if_need(step=current_step)
                self.tqdm_iter.set_description(f"Training in progress, {self.state.device}")

                # dist.barrier()
                # hparam we want track.
                hparams = {
                    'lr': optimizer.param_groups[0]['lr'],
                    'batch_size': self.state.batch_size,
                }
                metrics = {
                    'loss/normal_loss': normal_loss,
                    'loss/grad_loss': grad_norm,
                }
                criterions = {
                    "loss/normal_loss": normal_loss,
                    "loss/grad_loss": grad_norm,
                    "loss/mel_loss": mel_loss,
                    "loss/gate_loss": gate_loss,
                    "loss/stft_err": stft_err,
                    "score/stft_err": diag_score,
                }

                if not self.state.is_hyper_tunner:
                    self.tf_logger.log_training(criterions, current_step,
                                                optimizer.param_groups[0]['lr'],
                                                hparams=hparams, metrics=metrics)
            current_step += 1
            batch_counter += 1

        self.metric.on_batch_end()
        self._callback.on_loader_end()

        # update last step trained for a current model.
        self._last_step[layer_name] = current_step
        self.state.step = current_step

    @staticmethod
    def tacotron25_batch(batch, device):
        """
        Batch parser for DTC.

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

        return (text_padded, input_lengths, mel_padded, max_len, output_lengths), \
               (mel_padded, gate_padded)

        # return (text_padded, input_lengths, mel_padded, max_len, output_lengths, spectral), \
        #        (mel_padded, gate_padded, spectral)

    def tacotron30_batch(self, batch, device):
        """
        Batch parse original output from data loader and upload to GPU.

        :param device:
        :param batch:
        :return:
        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, stft = batch
        text_padded = to_gpu(text_padded, device).long()
        input_lengths = to_gpu(input_lengths, device).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded, device).float()
        gate_padded = to_gpu(gate_padded, device).float()
        output_lengths = to_gpu(output_lengths, device).long()

        if self.spectogram_spec.is_stft_loss_enabled():
            mel_padded = mel_padded.contiguous().cuda(non_blocking=True)
            stft_padded = stft.contiguous().cuda(non_blocking=True)

            # sf = stft.contiguous()
            # if torch.cuda.is_available():
            #     sf = sf.cuda(non_blocking=True)
            #     sf.requires_grad = False

            return (text_padded, input_lengths, mel_padded, max_len, output_lengths), \
                   (mel_padded, gate_padded, stft_padded)

        else:
            return (text_padded, input_lengths, mel_padded, max_len, output_lengths), \
                   (mel_padded, gate_padded)

    def cleanup(self):
        """
        Cleanup call used only for DDP.
        :return:
        """
        if self.state.trainer_spec.is_distributed_run():
            dist.destroy_process_group()
        logger.info(f"Rank {self.state.rank} is done.")

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
        print(f"Prepare model {model_name} layer {layer_name} for training.")

        if self.state.trainer_spec.is_distributed_run():
            device = self.state.device
        else:
            device = self.state.device

        assert self.criterion is not None
        self.criterion = self.criterion(spec=self.spectogram_spec, device=device)
        self.criterion.to(device)

        self.tqdm_iter = self.trainer_iterator(model_name, layer_name)

        assert model_name in self._optimizers
        # assert layer_name in self._optimizers[model_name]

        optimizer = self._optimizers[model_name][layer_name]
        scheduler = None

        if model_name in self._schedulers:
            if layer_name in self._schedulers:
                logger.info("Model {} contains a scheduler".format(model_name))
                scheduler = self._schedulers[model_name][layer_name]

        # if self.state.trainer_spec.is_distributed_run():
        #     logger.info("Running in distributed model. applying gradient reduce.".format(model_name))
        #     model = apply_gradient_allreduce(model)

        last_epoch, last_step = self.update_running_state(model_name=model_name, layer_name=layer_name)
        if last_epoch == self.state.trainer_spec.epochs():
            print("Model trainer, warm up validation pass.")
            self.validate_epoch(model, model_name, layer_name, warmup=True)

        # TODO add option if epoch changed after save
        self.metric.set_num_iteration(self.state.trainer_spec.epochs() * self.total_batches)
        self.metric.init()
        logger.info("Staring training num epochs {}, epoch trained {} num batches {} expected total iteration {}",
                    self.state.trainer_spec.epochs(), self._last_ckt_epochs, len(self._train_loader),
                    self.state.trainer_spec.epochs() * len(self._train_loader))

        # re-adjust last step
        if not self.state.disable_pbar:
            self.tqdm_iter.set_postfix({'step': self.state.step})

        return model, optimizer, scheduler

    def trainer_sequential(self, model_name: str, layer_name: str) -> None:
        """
        Sequential training loop.

        :param model_name:
        :param layer_name:
        :return:
        """
        assert model_name in self._models
        assert layer_name in self._models[model_name]
        assert self.state.current_layer == layer_name
        assert self.state.current_model == model_name

        model, optimizer, scheduler = self.prepare_trainer(model_name, layer_name)
        self.state.current_model = model
        self.state.current_optimizer = optimizer
        self.state.current_schedulers = scheduler

        logger.info(f"Last epoch saved {self._last_ckt_epochs[model_name][layer_name]}")
        logger.info(f"Last iteration saved {self._last_step[layer_name]}")

        # if self.state.trainer_spec.is_distributed_run():
        #     model = dist

        model.train()
        self._callback.on_begin()
        self.metric.on_begin()

        if self.state.is_hyper_tunner:
            print(f"Training in hyperparameter tunner mode.")

        epochs_loss_terms = defaultdict(float)
        for epoch in self.tqdm_iter:
            if self.state.is_hyper_tunner:
                print(f"Training in hyperparameter tunner mode. {epoch} max epoch {len(self.tqdm_iter)}")
                for param_group in self._optimizers[model_name][layer_name].param_groups:
                    print(f"Training lr {param_group['lr']} batch_size {self.state.batch_size}")

            self.state.epoch = epoch
            self._callback.on_epoch_begin()
            if self.state.trainer_spec.is_distributed_run():
                dist.barrier()
            # train epoch's batch
            self.metric.on_epoch_begin()
            self.train_epoch(model, model_name, layer_name, optimizer)
            self.metric.on_epoch_end()
            # if self._brake_epoch_loop == epoch:
            #     sys.exit(1)            # if self._brake_epoch_loop == epoch:
            aggregate_loss = self.validate_epoch(model, model_name, layer_name, epoch)
            for k in aggregate_loss:
                epochs_loss_terms[k] += aggregate_loss[k]
            self._callback.on_epoch_end()
            if self.state.is_hyper_tunner and self.hyper_config['epoch'] == epoch:
                break

        self.state.epoch += 1
        self._callback.on_end()
        self.metric.on_end()

        self._callback.on_epoch_end()
        self.metric.total_train_mean_loss()
        return

    def train(self, model_name=None, config=None, checkpoint_dir=None):
        """
        Main routine for model training.

        :param  config:
        :param  model_name:
        :param  checkpoint_dir this mainly for ray
        :return:
        """
        # torch.manual_seed(self.model_spec.seed())
        # torch.cuda.manual_seed(self.model_spec.seed())
        # if self.state.trainer_spec.is_distributed_run():
        # torch.cuda.set_device(self.state.device)

        torch.cuda.empty_cache()
        model_name = self.state.trainer_spec.get_active_model()
        model_layers = self.state.trainer_spec.get_active_sub_models()

        self.load()
        if self.is_trained(model_name):
            print(f"It looks like model {model_name} already trained.")
            for layer in model_layers:
                print(f"Last saved file {self.state.trainer_spec.model_files.get_model_file_path(layer)}")
            return

        # last_step = 0
        strategy = self.state.trainer_spec.get_training_strategy(model_name)
        if strategy == 'sequential':
            for layer in model_layers:
                self.q.append(layer)
            while len(self.q) > 0:
                layer_name = self.q.pop()
                self.state.current_model = model_name
                self.state.current_layer = layer_name
                # update whatever we need
                # if config is not None:
                #     if 'batch_size' in config:
                #         self.state.data_loaders.update(config['batch_size'])
                #     if 'lr' in config:
                #         for param_group in self._optimizers[model_name][layer_name].param_groups:
                #             param_group['lr'] = config["lr"]
                # run model
                self.trainer_sequential(model_name=model_name, layer_name=layer_name)
                self.save()
                logger.info(f"Saved last epoch {self.state.trainer_spec.epochs()}")

        self.cleanup()

    def train_optimizer(self, config):
        """

        :return:
        """
        logger.info("Hyper parameters tunner invoked.")
        torch.cuda.empty_cache()
        model_name = self.state.trainer_spec.get_active_model()
        model_layers = self.state.trainer_spec.get_active_sub_models()

        # last_step = 0
        strategy = self.state.trainer_spec.get_training_strategy(model_name)
        if strategy == 'sequential':
            for layer in model_layers:
                self.q.append(layer)
            while len(self.q) > 0:
                layer_name = self.q.pop()
                self.state.current_model = model_name
                self.state.current_layer = layer_name
                # update whatever we need
                if config is not None:
                    if 'lr' in config:
                        for param_group in self._optimizers[model_name][layer_name].param_groups:
                            param_group['lr'] = config["lr"]
                # todo add aggregation
                return self.trainer_sequential(model_name, layer_name)

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

        layers = self.state.trainer_spec.get_active_sub_models()
        for layer in layers:
            if layer in self._last_ckt_epochs[model_name]:
                last = self._last_ckt_epochs[model_name][layer]
                if int(last) >= int(self.state.trainer_spec.epochs()):
                    num_finished += 1

        if num_finished == len(layers):
            return True

        return False

    def get_model(self, name: str):
        """

        :param name:
        :return:
        """
        if name in self._models:
            return self._models[name]

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
