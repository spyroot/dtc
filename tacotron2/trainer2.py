import os
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from hparams import create_hparams
from tacotron2.model_specs import ModelSpecs
from tacotron2.utils import fmtl_print


class Trainer:
    """

    """
    def __init__(self, model_spec: ModelSpecs):
        """

        :param model_spec:
        """
        # self.model_spec = model_spec
        # self.n_gpus = model_spec['model_spec']
        # self.rank = model_spec['model_spec']
        # self.group_name = model_spec['group_name']

        if self.model_spec.is_distributed_run():
            self.init_distributed()

    def init_distributed(self):
        """

        :return:
        """
        assert torch.cuda.is_available(), "Distributed mode requires CUDA."
        print("Initializing Distributed")

        # Set cuda device so everything is done on the right GPU.
        torch.cuda.set_device(self.rank() % torch.cuda.device_count())

        # Initialize distributed communication
        dist.init_process_group(
            backend=self.model_spec.get_backend(),
            init_method=self.model_spec.dist_url(),
            world_size=self.n_gpus(),
            rank=self.rank(),
            group_name=self.group_name())

        print("Done initializing distributed")

    def prepare_dataloaders(self):
        """

        :return:
        """
        # Get data, data loaders and collate function ready
        train_set = TextMelLoader(hparams.training_files, hparams)
        val_set = TextMelLoader(hparams.validation_files, hparams)
        collate_fn = TextMelCollate(hparams.n_frames_per_step)

        if self.model_spec.is_distributed_run():
            train_sampler = DistributedSampler(train_set)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        train_loader = DataLoader(train_set, num_workers=1, shuffle=shuffle,
                                  sampler=train_sampler,
                                  batch_size=self.model_spec.batch_size, pin_memory=False,
                                  drop_last=True, collate_fn=collate_fn)
        return train_loader, valset, collate_fn

    def load_model(self, hparams):
        """

        :param hparams:
        :return:
        """
        model = Tacotron2(hparams).cuda()
        if hparams.fp16_run():
            model.decoder.attention_layer.score_mask_value = finfo('float16').min

        if hparams.distributed_run():
            model = apply_gradient_allreduce(model)

        return model

    def warm_start_model(checkpoint_path, model, ignore_layers):
        """

        :param model:
        :param ignore_layers:
        :return:
        """
        assert os.path.isfile(checkpoint_path)
        print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        model_dict = checkpoint_dict['state_dict']
        if len(ignore_layers) > 0:
            model_dict = {k: v for k, v in model_dict.items()
                          if k not in ignore_layers}
            dummy_dict = model.state_dict()
            dummy_dict.update(model_dict)
            model_dict = dummy_dict
        model.load_state_dict(model_dict)
        return model

    def load_checkpoint(checkpoint_path, model, optimizer):
        assert os.path.isfile(checkpoint_path)
        print("Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint_dict['state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        learning_rate = checkpoint_dict['learning_rate']
        iteration = checkpoint_dict['iteration']
        print("Loaded checkpoint '{}' from iteration {}".format(
            checkpoint_path, iteration))
        return model, optimizer, learning_rate, iteration

    def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
        print("Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath))
        torch.save({'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': learning_rate}, filepath)

    def reduce_tensor(self, tensor, n_gpus):
        """

        :param n_gpus:
        :return:
        """
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= n_gpus
        return rt

    def train(self, checkpoint_path, warm_start):
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

        torch.manual_seed(self.model_spec.seed())
        torch.cuda.manual_seed(self.model_spec.seed())

        model = self.load_model()
        learning_rate = self.model_spec.learning_rate()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=self.model_spec.weight_decay)

        if self.model_spec.is_fp16_run():
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

        if self.model_spec.is_distributed_run():
            model = apply_gradient_allreduce(model)

        criterion = Tacotron2Loss()

        # logger = prepare_directories_and_logger(
        #     output_directory, log_directory, rank)

        train_loader, valset, collate_fn = self.prepare_dataloaders()

        # Load checkpoint if one exists
        iteration = 0
        epoch_offset = 0
        # if checkpoint_path is not None:
        #     if warm_start:
        #         # model = warm_start_model(
        #         #     checkpoint_path, model, hparams.ignore_layers)
        #     else:
        #         model, optimizer, _learning_rate, iteration = self.load_checkpoint(
        #             checkpoint_path, model, optimizer)
        #         if self.model_spec.use_saved_learning_rate:
        #             learning_rate = _learning_rate
        #         iteration += 1  # next iteration is iteration + 1
        #         epoch_offset = max(0, int(iteration / len(train_loader)))

        model.train()
        is_overflow = False
        # ================ MAIN TRAINNIG LOOP! ===================
        for epoch in range(epoch_offset, self.model_spec.epochs):
            print("Epoch: {}".format(epoch))
            for i, batch in enumerate(train_loader):
                start = time.perf_counter()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

                model.zero_grad()
                x, y = model.parse_batch(batch)
                y_pred = model(x)

                loss = criterion(y_pred, y)

                if self.model_spec.is_fp16_run():
                    reduced_loss = self.reduce_tensor(loss.data).item()
                else:
                    reduced_loss = loss.item()

                if self.model_spec.is_fp16_run():
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.model_spec.is_fp16_run:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), self.model_spec.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.model_spec.grad_clip_thresh)

                optimizer.step()

                # if not is_overflow and rank == 0:
                #     duration = time.perf_counter() - start
                #     print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                #         iteration, reduced_loss, grad_norm, duration))
                #     logger.log_training(
                #         reduced_loss, grad_norm, learning_rate, duration, iteration)
                #
                # if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                #     validate(model, criterion, valset, iteration,
                #              hparams.batch_size, n_gpus, collate_fn, logger,
                #              hparams.distributed_run, rank)
                #     if rank == 0:
                #         checkpoint_path = os.path.join(
                #             output_directory, "checkpoint_{}".format(iteration))
                #         save_checkpoint(model, optimizer, learning_rate, iteration,
                #                         checkpoint_path)

                iteration += 1


if __name__ == '__main__':
    """
    """
    model_trainer_spec = ModelSpecs()
    fmtl_print("active model", model_trainer_spec.active_model)
    fmtl_print("active dataset", model_trainer_spec.use_dataset)

    print(model_trainer_spec.model_spec)

    # model_trainer_spec.build_training_set_from_files()
    # print(model_trainer_spec.dataset_specs['dir'])
    # training_set, validation_set, test_set = model_trainer_spec.get_audio_ds_files()
    # trainer = Trainer(model_trainer_spec)

    # train_set = TextMelLoader(model_trainer_spec, list(training_set.values()))

    # val_set = TextMelLoader(hparams.validation_files, hparams)
    # collate_fn = TextMelCollate(hparams.n_frames_per_step)

