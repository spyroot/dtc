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
from tacotron2.model_specs.model_specs import ModelSpecs
from tacotron2.utils import fmtl_print

if __name__ == '__main__':
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

    args = parser.parse_args()
    model_trainer_spec = ModelSpecs()
    fmtl_print("active model", model_trainer_spec.active_model)
    fmtl_print("active dataset", model_trainer_spec.use_dataset)

    # model_trainer_spec.build_training_set_from_files()
    # print(model_trainer_spec.dataset_specs['dir'])
    # training_set, validation_set, test_set = model_trainer_spec.get_audio_ds_files()

    # build_file_list(model_trainer_spec.dataset_specs)
    # hparams = create_hparams(args.hparams)
    # torch.backends.cudnn.enabled = hparams.cudnn_enabled
    # torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    #
    # print("FP16 Run:", hparams.fp16_run)
    # print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    # print("Distributed Run:", hparams.distributed_run)
    # print("cuDNN Enabled:", hparams.cudnn_enabled)
    # print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    #
    # # train(args.output_directory, args.log_directory, args.checkpoint_path,
    # #       args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)

    # print(model.model_files.filename_metrics)
    # print(model.model_files.filename_train)
    # print(model.model_files.filename_test)
    # print(model.model_files.filename_prediction)
    # print(model.model_files.root_dir)
    # print(model.model_files.results_dir())
    # model.model_files.build_dir()
