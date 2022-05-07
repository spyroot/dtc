import time
from abc import ABC

import ray
import torch
import torch.distributed as dist
import math

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from TrainingLogger import Tacotron2Logger
from distributed import apply_gradient_allreduce
from model_loader.mel_dataloader import Mel_Dataloader
from model_loader.mel_dataset_loader import TextMelLoader
from model_trainer.model_trainer_specs import ExperimentSpecs
from models.model import Tacotron2
from tacotron2.loss_function import Tacotron2Loss
from tacotron2.utils import fmtl_print, fmt_print
from numpy import finfo
from torch.nn.utils import clip_grad_norm_
import argparse
from generator_trainer import GeneratorTrainer
from tqdm import tqdm, tnrange
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.autograd import Variable
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from text import sequence_to_text


class Trainer(GeneratorTrainer, ABC):
    """

    """

    def __init__(self, experiment_specs: ExperimentSpecs, data_loader,
                 verbose=False, is_notebook=False, rank=0, device=None):
        """

        :param experiment_specs:
        """
        super(Trainer, self).__init__(verbose=verbose, is_notebook=is_notebook)
        # self.model_spec = model_spec
        # self.n_gpus = model_spec['model_spec']
        # self.rank = model_spec['model_spec']
        # self.group_name = model_spec['group_name']

        self.n_gpus = 1
        self.device = device
        self.schedulers = {}
        self.optimizers = {}

        self.experiment_specs = experiment_specs
        if self.experiment_specs.is_distributed_run():
            self.init_distributed()

        self.dataloader = dataloader
        self.train_loader, self.validation_loader, self.collate_fn = data_loader.get_loader()
        self.rank = rank

        self.criterion = Tacotron2Loss()
        self.models = {}
        self.last_epochs = {}
        self.iters = {}

        self.init_trainer()
        self.scaler = None
        self.logger = Tacotron2Logger()

    def init_trainer(self):
        """

        Returns:

        """
        self.init_distributed()
        self.create_models()
        self.create_optimizers()
        self.create_lr_schedulers()
        if self.experiment_specs.fp16_run():
            self.scaler = torch.cuda.amp.GradScaler()

        torch.manual_seed(self.experiment_specs.seed())
        torch.cuda.manual_seed(self.experiment_specs.seed())

    def init_distributed(self):
        """

        :return:
        """
        assert torch.cuda.is_available(), "Distributed mode requires CUDA."
        fmtl_print("Distributed Available", torch.cuda.device_count())

        # Set cuda device so everything is done on the right GPU.
        torch.cuda.set_device(self.rank % torch.cuda.device_count())

        # Initialize distributed communication
        # dist.init_process_group(
        #     backend=self.model_spec.get_backend(),
        #     init_method=self.model_spec.dist_url(),
        #     world_size=self.n_gpus(),
        #     rank=self.rank(),
        #     group_name=self.group_name())

        fmtl_print("Done initializing distributed")

    def create_model(self, model_name):
        """
        Method create model.  Later will move to a separate dispatcher creator.
        Args:
            model_name:

        Returns:

        """
        if model_name == 'encoder':
            model = Tacotron2(self.experiment_specs, self.device)
            if self.experiment_specs.is_fp16_run():
                model.decoder.attention_layer.score_mask_value = finfo('float16').min

            if self.experiment_specs.is_distributed_run():
                model = apply_gradient_allreduce(model)

            self.models[model_name] = model
            self.last_epochs[model_name] = 0

    # def warm_start_model(checkpoint_path, model, ignore_layers):
    #     """
    #
    #     :param model:
    #     :param ignore_layers:
    #     :return:
    #     """
    #     assert os.path.isfile(checkpoint_path)
    #     print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    #     checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    #     model_dict = checkpoint_dict['state_dict']
    #     if len(ignore_layers) > 0:
    #         model_dict = {k: v for k, v in model_dict.items()
    #                       if k not in ignore_layers}
    #         dummy_dict = model.state_dict()
    #         dummy_dict.update(model_dict)
    #         model_dict = dummy_dict
    #     model.load_state_dict(model_dict)
    #     return model

    # def load_checkpoint(checkpoint_path, model, optimizer):
    #     assert os.path.isfile(checkpoint_path)
    #     print("Loading checkpoint '{}'".format(checkpoint_path))
    #     checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    #     model.load_state_dict(checkpoint_dict['state_dict'])
    #     optimizer.load_state_dict(checkpoint_dict['optimizer'])
    #     learning_rate = checkpoint_dict['learning_rate']
    #     iteration = checkpoint_dict['iteration']
    #     print("Loaded checkpoint '{}' from iteration {}".format(
    #         checkpoint_path, iteration))
    #     return model, optimizer, learning_rate, iteration
    #
    # def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    #     print("Saving model and optimizer state at iteration {} to {}".format(
    #         iteration, filepath))
    #     torch.save({'iteration': iteration,
    #                 'state_dict': model.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'learning_rate': learning_rate}, filepath)

    @staticmethod
    def split_tensor(tensor, n_gpus):
        """
        "
        Args:
            tensor:
            n_gpus:

        Returns:

        """
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= n_gpus
        return rt

    def load(self, model_name: str):
        """
        Method loads model from a checkpoint.
        Args:
            model_name:

        Returns:

        """
        if not self.experiment_specs.load():
            return 0

        model_file = self.experiment_specs.model_files.get_model_filename(model_name)
        fmtl_print('Loading model weights ', model_name, model_file)

        # load trained optimizer state_dict
        try:
            self.models[model_name].to(self.device)
            checkpoint = torch.load(model_file, map_location=self.device)
            if 'model_state_dict' not in checkpoint:
                raise Exception("model has no state dict")

            self.models[model_name].load_state_dict(checkpoint['model_state_dict'])
            if 'model_state_dict' not in checkpoint:
                raise Exception("model has no state dict")

            self.optimizers[model_name].load_state_dict(checkpoint['optimizer_state_dict'])
            if 'optimizer_state_dict' not in checkpoint:
                raise Exception("model has no optimizer_state_dict")

            # if model_name in self.schedulers:
            #     self.schedulers[model_name].load_state_dict(checkpoint['scheduler_state_dict'])
            #     if 'scheduler_state_dict' not in checkpoint:
            #         raise Exception("model has no scheduler_state_dict")

            # self.trainer_spec.set_lr(0.00001)
            # print('Model loaded, lr: {}'.format(self.trainer_spec.lr()))
            fmtl_print("Last checkpoint. ", checkpoint['epoch'], checkpoint['it'])
            self.last_epochs[model_name] = checkpoint['epoch']
            self.iters[model_name] = checkpoint['it']
            return checkpoint['epoch'], checkpoint['it']
        except FileNotFoundError as e:
            print("Failed load model files. No saved model found.")

        return 0

    def trainer_iterator(self, model_name: str, last_epoch=0, max_epochs=0):
        """

        Args:
            model_name:
            last_epoch:
            max_epochs:

        Returns:

        """
        if self.is_notebook:
            tqdm_iter = tnrange(last_epoch, max_epochs)
            return tqdm_iter

        # load last epoch in case we do re-summing.
        last_epoch = self.last_epochs[model_name]
        # early stopping
        early_stopping = None

        max_epochs = self.experiment_specs.epochs()
        # what tqdm to use notebook for colab or normal one.
        if self.verbose:
            fmtl_print("Creating tqdm", last_epoch, max_epochs)

        if self.is_notebook:
            tqdm_iter = tnrange(last_epoch, max_epochs)
        else:
            tqdm_iter = tqdm(range(last_epoch, max_epochs))

        tqdm_iter.set_postfix({'total_epoch_loss': 0})
        return tqdm_iter

    def create_models(self):
        """

        Returns:

        """
        models = self.experiment_specs.get_active_sub_models()
        for m in models:
            self.create_model(m)

    def load_models(self):
        """

        Returns:

        """
        models = self.experiment_specs.get_active_sub_models()
        for m in models:
            self.load(m)

    def reduce_if_needed(self, loss):
        """

        Args:
            loss:

        Returns:

        """
        if self.experiment_specs.is_fp16_run():
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

        if model_name not in self.models:
            raise Exception("config.yaml must contains valid active settings. "
                            "Failed create {} model".format(model_name))
        model = self.models[model_name]

        optimizer_type = self.experiment_specs.optimizer_type(alias_name)
        spec = self.experiment_specs

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
            opt = optim.opt = optim.SGD(list(self.models[model_name].parameters(alias_name)),
                                        lr=spec.optimizer_learning_rate(alias_name),
                                        momentum=spec.momentum(alias_name),
                                        dampening=spec.dampening(alias_name),
                                        weight_decay=spec.weight_decay(alias_name),
                                        nesterov=spec.nesterov(alias_name))
        elif self.experiment_specs.optimizer_type == 'none':
            opt = None
        else:
            raise ValueError("unknown optimizer: {}".format(optimizer_type))

        self.optimizers[model_name] = opt
        return opt

    def create_optimizers(self):
        """
        Create all required optimizers based on model specs.
        Each optimize in self.optimizers dict
        Returns: Nothing
        """

        models = self.experiment_specs.get_active_sub_models()
        for model_name in models:
            opt_spec_alias = experiment_specs.get_sub_model_optimizer(model_name)
            optimizer = self.create_optimizer(model_name, opt_spec_alias)
            self.optimizers[model_name] = optimizer

    def create_lr_scheduler(self, model_name: str, optimizer):
        """
        Creates lr scheduler based on specs and attach to optimizer
        Args:
            model_name:  a model name
            optimizer: target optimizer

        Returns: lr_scheduler

        """
        alias_name = experiment_specs.get_sub_model_lr_scheduler(model_name)
        if len(alias_name) == 0:
            if self.verbose:
                fmtl_print("Model {}".format(model_name), "no scheduler attached")
            return

        lr_scheduler_type = self.experiment_specs.lr_scheduler_type(alias_name)
        if lr_scheduler_type == 'cos':
            if self.verbose:
                fmtl_print("Creating {} lr scheduler.".format(alias_name), "cos")
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=self.experiment_specs.t_max(alias_name),
                                                       eta_min=self.experiment_specs.eta_min(alias_name))
        elif lr_scheduler_type == 'multistep':
            fmtl_print("Creating {} lr scheduler.".format(alias_name), "multistep")
            fmtl_print("Creating {} milestone.".format(alias_name), self.experiment_specs.milestones(alias_name))
            if self.verbose:
                fmtl_print("Creating {} lr scheduler.".format(alias_name), "multistep")
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=self.experiment_specs.milestones(alias_name),
                                                 gamma=self.experiment_specs.gamma(alias_name))
        elif lr_scheduler_type == 'exp-warmup':
            if self.verbose:
                fmtl_print("Creating {} lr_scheduler_type.".format(alias_name), "exp-warmup")
            lr_lambdas = self.experiment_specs.lr_lambdas(alias_name)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
        elif lr_scheduler_type == 'none' or lr_scheduler is None:
            if self.verbose:
                fmtl_print("Creating {} optimizer.".format(alias_name), "none")
            scheduler = None
        else:
            raise ValueError("unknown scheduler: {}".format(lr_scheduler_type))

        self.schedulers[model_name] = scheduler
        return scheduler

    def create_lr_schedulers(self):
        """

        Returns:

        """
        models = self.experiment_specs.get_active_sub_models()
        for model_name in models:
            if model_name not in self.optimizers:
                raise Exception("make sure optimizer created.")

            opt = self.optimizers[model_name]
            self.create_lr_scheduler(model_name, opt)

    def save_if_need(self, model_name, it, epoch, last_epoch=False):
        """
        Saves model checkpoint, when based on template settings.

        Args:
            it:
            model_name:
            epoch: current epoch
            last_epoch
        """
        # by default condition to save epoch , if save per iteration we check iteration.
        if self.experiment_specs.is_save() is False:
            return False

        if last_epoch is False and it == 0:
            return False

        save_condition = epoch
        model_file = self.experiment_specs.model_files.get_model_filename(model_name)
        if self.experiment_specs.is_save_per_iteration():
            save_condition = it

        if save_condition % self.experiment_specs.epochs_save() == 0 or last_epoch is True:
            if self.experiment_specs.is_train_verbose():
                fmt_print('Saving node model {}'.format(model_file))

            if model_name in self.schedulers:
                torch.save({
                    'epoch': epoch,
                    'it': it,
                    'model_state_dict': self.models[model_name].state_dict(),
                    'optimizer_state_dict': self.optimizers[model_name].state_dict(),
                    #      'scheduler_state_dict': self.schedulers[model_name].state_dict()
                }, model_file)
            else:
                torch.save({
                    'epoch': epoch,
                    'it': it,
                    'model_state_dict': self.models[model_name].state_dict(),
                    'optimizer_state_dict': self.optimizers[model_name].state_dict(),
                    #    'scheduler_state_dict': self.schedulers[model_name].state_dict()
                }, model_file)

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
        if self.rank == 0 and it % self.experiment_specs.epochs_log() == 0:
            print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(it,
                                                                            reduced_loss,
                                                                            grad_norm,
                                                                            duration))

    #     logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration)
    #
    # if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
    #     validate(model, criterion, valset, iteration, hparams.batch_size, n_gpus, collate_fn, logger,
    #              hparams.distributed_run, rank)
    #     if rank == 0:
    #         checkpoint_path = os.path.join(output_directory, "checkpoint_{}".format(iteration))
    #         save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)

    # if not is_overflow and (iteration % self.experiment_specs.save() == 0):
    #     validate(model, criterion, valset, iteration,
    #              hparams.batch_size, n_gpus, collate_fn, logger,
    #              hparams.distributed_run, rank)
    #     if rank == 0:
    #         checkpoint_path = os.path.join(
    #             output_directory, "checkpoint_{}".format(iteration))
    #         save_checkpoint(model, optimizer, learning_rate, iteration,
    #                         checkpoint_path)

    def warm_load(self, model_name: str, ignore_layers):
        """
        Method loads model from a checkpoint.
        Args:
            ignore_layers:
            model_name:

        Returns:

        """
        if not self.experiment_specs.load():
            return 0

        model_file = self.experiment_specs.model_files.get_model_filename(model_name)
        # load trained optimizer state_dict
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'model_state_dict' not in checkpoint:
                raise Exception("model has no state dict")

            model = self.models[model_name]
            model_dict = checkpoint['model_state_dict']
            if 'model_state_dict' not in checkpoint:
                raise Exception("model has no state dict")

            if len(ignore_layers) > 0:
                model_dict = {k: v for k, v in model_dict.items()
                              if k not in ignore_layers}
                dummy_dict = model.state_dict()
                dummy_dict.update(model_dict)
                model_dict = dummy_dict
                self.models[model_name].load_state_dict(model_dict)

            self.optimizers[model_name].load_state_dict(checkpoint['optimizer_state_dict'])
            if 'optimizer_state_dict' not in checkpoint:
                raise Exception("model has no optimizer_state_dict")

            # if model_name in self.schedulers:
            #     self.schedulers[model_name].load_state_dict(checkpoint['scheduler_state_dict'])
            #     if 'scheduler_state_dict' not in checkpoint:
            #         raise Exception("model has no scheduler_state_dict")

            fmtl_print('attempting to load {} model edge weights state_dict '
                       'loaded...'.format(self.experiment_specs.get_active_model()))

            # self.trainer_spec.set_lr(0.00001)
            # print('Model loaded, lr: {}'.format(self.trainer_spec.lr()))
            fmtl_print("Last checkpoint saved. ", checkpoint['epoch'], checkpoint['it'])
            return checkpoint['epoch']

        except FileNotFoundError as e:
            print("Failed load model files. No saved model found.")

        return 0

    def validate(self, model, model_name, it):
        """

        Args:
            model:
            it:
            n_gpus:

        Returns:

        """
        t_writer = self.experiment_specs.get_tensorboard_writer()

        model.eval()
        with torch.no_grad():
            # val_sampler = DistributedSampler(valset) if distributed_run else None
            # val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
            #                         shuffle=False, batch_size=batch_size,
            #                         pin_memory=False, collate_fn=collate_fn)

            total_prediction_loss = 0.0
            print("validation batch", len(self.validation_loader))
            for i, batch in enumerate(self.validation_loader):
                x, y = model.parse_batch(batch)
                y_pred = model(x)
                loss = self.criterion(y_pred, y)
                if self.experiment_specs.is_distributed_run():
                    reduced_val_loss = self.split_tensor(loss.data, self.n_gpus).item()
                else:
                    reduced_val_loss = loss.item()
                total_prediction_loss += reduced_val_loss
            total_prediction_loss = total_prediction_loss / (i + 1)

        model.train()
        if self.rank == 0:
            # t_writer.add_scalar('val_loss_' + model_name, loss.item(), it)
            print("Validation loss {}: {:9f}  ".format(it, total_prediction_loss))
            self.logger.log_validation(total_prediction_loss, model, y, y_pred, it)

        return total_prediction_loss

    def get_last_iterator(self, model_name):
        """

        Args:
            model_name:

        Returns:

        """
        it = 0
        if model_name in self.iters[model_name]:
            it = self.iters[model_name]

        return it

    def train(self, model_name='encoder'):
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
        t_writer = self.experiment_specs.get_tensorboard_writer()

        if model_name in self.iters:
            fmtl_print("Epoch saved out of", self.last_epochs[model_name], self.experiment_specs.epochs())
            fmtl_print("Last iter saved", self.iters[model_name])

        #
        model = self.models[model_name].to(self.device)
        tqdm_iter = self.trainer_iterator(model_name)
        optimizer = self.optimizers[model_name]
        scheduler = None
        if model_name in self.schedulers:
            print("Model has scheduler")
            scheduler = self.schedulers[model_name]

        #
        if self.experiment_specs.is_distributed_run():
            model = apply_gradient_allreduce(model)

        it = self.get_last_iterator(model_name)

        if self.last_epochs[model_name] == self.experiment_specs.epochs():
            total_prediction_loss = self.validate(model, model_name, it)

        model.train()
        tqdm_iter.set_postfix({'run': it})
        saved_run = it
        for epoch in tqdm_iter:
            total_epoch_loss = 0
            total_prediction_loss = 0
            total_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                start = time.perf_counter()
                # for param_group in self.optimizer[model_name].param_groups:
                #     param_group['lr'] = learning_rate

                model.zero_grad()
                x, y = model.parse_batch(batch)
                y_pred = model(x)

                loss = self.criterion(y_pred, y)
                total_epoch_loss += loss.item()

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), self.experiment_specs.grad_clip_thresh)
                optimizer.step()

                # run lr_scheduler
                if scheduler is not None:
                    scheduler.step()

                duration = time.perf_counter() - start
                # self.log_if_needed(it, loss, grad_norm, duration)
                if self.rank == 0 and it % self.experiment_specs.epochs_log() == 0:
                    tqdm_iter.set_postfix({'run': it,
                                           'total_loss': total_epoch_loss,
                                           'accuracy': total_prediction_loss,
                                           'grad_norm': grad_norm.item(),
                                           'device': str(grad_norm.device),
                                           'batch': batch_idx,
                                           'batches': total_batches,
                                           'saved run': saved_run})

                if it != 0 and it % self.experiment_specs.predict() == 0:
                    total_prediction_loss = self.validate(model, model_name, it)

                # save model checkpoint
                if self.save_if_need(model_name, it, epoch):
                    tqdm_iter.set_postfix({'run': it,
                                           'total_loss': total_epoch_loss,
                                           'accuracy': total_prediction_loss,
                                           'grad_norm': grad_norm.item(),
                                           'device': str(grad_norm.device),
                                           'batch': batch_idx,
                                           'batches': total_batches,
                                           'saved run': it})
                    saved_run = it

                self.logger.log_training(loss.item(), grad_norm, optimizer.param_groups[0]['lr'], it)
                metrics = {'accuracy/accuracy': None, 'loss/loss': None}

                # hp.hparams_config(
                #     hparams=[HP_OPTIMIZER],
                #     metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
                # )
                #
                # h_params = {'This': 1, 'is': 2, 'a': 3, 'test': 4}
                # t_writer.add_hparams(h_params, metrics)
                # with SummaryWriter() as w:
                #     for i in range(5):

                t_writer.add_hparams(
                    {'lr': optimizer.param_groups[0]['lr'], 'batch_size': self.experiment_specs.batch_size},
                    {'hparam/accuracy': total_epoch_loss,
                     'hparam/loss': float(loss.item())})
                # model logs
                for name, weight in model.named_parameters():
                    t_writer.add_histogram(name, weight, epoch)
                    t_writer.add_histogram(f'{name}.grad', weight.grad, epoch)

                # self.detach_hidden()
                it += 1
                t_writer.flush()
            #  tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

        if self.save_if_need(model_name, it, self.experiment_specs.epochs(), last_epoch=True):
            if self.verbose:
                fmtl_print("Saved last epoch", self.experiment_specs.epochs())

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
            model = self.models[model_name].to(self.device)
            tqdm_iter = self.trainer_iterator(model_name)
            learning_rate = self.experiment_specs.learning_rate

            optimizer = self.optimizers[model_name]
            #
            if self.experiment_specs.is_fp16_run():
                from apex import amp
                model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
            #
            if self.experiment_specs.is_distributed_run():
                model = apply_gradient_allreduce(model)
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
                    x, y = model.parse_batch(batch)
                    with torch.cuda.amp.autocast():
                        y_pred = model(x)

                    loss = self.criterion(y_pred, y)
                    total_epoch_loss += loss.item()

                    # reduced_loss = self.split_tensor(loss.data).item()
                    #     else:
                    #         reduced_loss = loss.item()
                    #

                    self.scaler.scale(loss).backward()
                    grad_norm = clip_grad_norm_(model.parameters(), self.experiment_specs.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm)

                    self.scaler.step(optimizer)
                    self.scaler.update()

                    # save model checkpoint
                    if self.save_if_need(model_name, iteration, epoch):
                        tqdm_iter.set_postfix({'total_epoch_loss': total_epoch_loss, 'saved': True})
                    iteration += 1


def print_optimizer(opt_name):
    """

    Returns:

    """
    optimizer_name = experiment_specs.get_sub_model_optimizer(opt_name)
    fmtl_print("Model optimizer", optimizer_name)
    fmtl_print("optimizer type", experiment_specs.optimizer_type(optimizer_name))
    fmtl_print("optimizer lr rate", experiment_specs.optimizer_learning_rate(optimizer_name))
    fmtl_print("optimizer weight_decay type", experiment_specs.weight_decay(optimizer_name))


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_specs = ExperimentSpecs(verbose=False)
    dataloader = Mel_Dataloader(experiment_specs, verbose=False)

    trainer = Trainer(experiment_specs, dataloader, verbose=True, device=device)

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    result = tune.run(
        tune.with_parameters(trainer.train()),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    if ray.util.client.ray.is_connected():
        # If using Ray Client, we want to make sure checkpoint access
        # happens on the server. So we wrap `test_best_model` in a Ray task.
        # We have to make sure it gets executed on the same node that
        # ``tune.run`` is called on.
        from ray.util.ml_utils.node import force_on_current_node


#     remote_fn = force_on_current_node(ray.remote(test_best_model))
#  ray.get(remote_fn.remote(best_trial))
# else:
# test_best_model(best_trial)

import dill
from pathlib import Path


def convert_mel_to_data(encoder_spec, target_dir: str, dataset,
                        dataset_name: str, data_type: str, post_check=True, verbose=True):
    """

    Args:
        data_type:
        verbose:
        target_dir:
        encoder_spec:
        dataset:
        dataset_name:
        post_check:

    Returns:

    """
    meta = dict(filter_length=encoder_spec.filter_length(), hop_length=encoder_spec.hop_length(),
                win_length=encoder_spec.win_length(), n_mel_channels=encoder_spec.n_mel_channels(),
                sampling_rate=encoder_spec.sampling_rate(), mel_fmin=encoder_spec.mel_fmin(),
                mel_fmax=encoder_spec.mel_fmax())

    data = []
    for i in range(0, len(dataset)):
        txt, mel = dataset[i]
        data.append((txt, mel))

    meta['data'] = data
    file_name = Path(target_dir) / f'{dataset_name}_{data_type}_{encoder_spec.n_mel_channels()}.pt'
    torch.save(meta, file_name)
    ds = torch.load(file_name)
    if verbose:
        fmtl_print("Dataset saved", file_name)
        fmtl_print("Dataset filter length", ds['filter_length'])
        fmtl_print("Dataset mel channels", ds['n_mel_channels'])
        fmtl_print("Dataset contains records", len(ds['data']))

    if post_check:
        d = ds['data']
        for i, (one_hot, mel) in enumerate(d):
            txt_original, mel_from_ds = dataset[i]
            if not torch.equal(mel, mel_from_ds):
                raise Exception("data mismatched.")
            if not torch.equal(one_hot, txt_original):
                raise Exception("data mismatched.")


def convert(verbose=True):
    """

    Returns:

    """
    trainer_spec = ExperimentSpecs(verbose=False)
    training_set, validation_set, test_set = trainer_spec.get_audio_ds_files()
    model_spec = trainer_spec.get_model_spec()
    encoder_spec = model_spec.get_encoder()

    #
    train_dataset = TextMelLoader(encoder_spec, list(training_set.values()))
    validation_dataset = TextMelLoader(encoder_spec, list(validation_set.values()))
    test_dataset = TextMelLoader(encoder_spec, list(test_set.values()))

    if verbose:
        fmtl_print("filter_length", encoder_spec.filter_length())
        fmtl_print("hop_length", encoder_spec.hop_length())
        fmtl_print("win_length", encoder_spec.win_length())
        fmtl_print("n_mel_channels", encoder_spec.n_mel_channels())
        fmtl_print("sampling_rate", encoder_spec.sampling_rate())
        fmtl_print("mel_fmin", encoder_spec.mel_fmin())
        fmtl_print("mel_fmax", encoder_spec.mel_fmax())

    convert_mel_to_data(encoder_spec, trainer_spec.get_dataset_dir(),
                        train_dataset, trainer_spec.use_dataset, "train")
    convert_mel_to_data(encoder_spec, trainer_spec.get_dataset_dir(),
                        validation_dataset, trainer_spec.use_dataset, "validate")
    convert_mel_to_data(encoder_spec, trainer_spec.get_dataset_dir(),
                        test_dataset, trainer_spec.use_dataset, "test")


def create_loader():
    """

    Returns:

    """
    experiment_specs = ExperimentSpecs(verbose=False)
    pk_dataset = experiment_specs.get_audio_dataset()
    print(pk_dataset.keys())
    # experiment_specs.model_files.build_dir()
    data_loader = Mel_Dataloader(experiment_specs, verbose=True)

    # for i, batch in enumerate(data_loader):
    #     print(type(batch))
    #     break


if __name__ == '__main__':
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    is_benchmark_read = False
    is_train = True
    is_convert = False

    experiment_specs = ExperimentSpecs(verbose=False)
    experiment_specs.model_files.build_dir()

    dataloader = Mel_Dataloader(experiment_specs, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_benchmark_read:
        dataloader.create()
        dataloader.benchmark_read()

    if is_convert:
        convert()

    create_loader()

    if is_train:
        trainer = Trainer(experiment_specs, dataloader, verbose=True, device=device)
        trainer.train()

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
