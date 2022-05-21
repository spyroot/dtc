import argparse
import logging
import os
import random
import signal
import sys
from pathlib import Path
import socket

import numpy as np
import torch
from loguru import logger

from model_loader.mel_dataloader import Mel_Dataloader
from model_loader.mel_dataset_loader import TextMelLoader
from model_trainer.trainer_specs import ExperimentSpecs
from model_trainer.trainer import Trainer
import torch.distributed as dist

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"


# docker network create -d macvlan --subnet=192.168.254.0/24 --ip-range=192.168.254.64/29 --gateway=192.168.254.100 -o parent=eth0 macvlan macvlan_mode=bridge
# docker run --gpus=all --rm --network macvlan -v ${PWD}:/datasets --workdir=/datasets dtc_rt:v1
# docker run --gpus=all --rm --network mynet -v ${PWD}:/datasets --workdir=/datasets dtc_rt:v1
# docker network create -d macvlan --subnet=192.168.254.0/24 --gateway=192.168.254.100 -o ipvlan_mode=l2 -o parent=eth0 ipvlan_net
# docker run --gpus=all --rm --network ipvlan_net --ip 192.168.254.232 -v ${PWD}:/datasets --workdir=/datasets dtc_rt:v1
# docker run --gpus=all --rm --network bridge --ip 192.168.254.232 -v ${PWD}:/datasets --workdir=/datasets dtc_rt:v1


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
        logger.info("Dataset saved", file_name)
        logger.info("Dataset filter length", ds['filter_length'])
        logger.info("Dataset mel channels", ds['n_mel_channels'])
        logger.info("Dataset contains records", len(ds['data']))

    if post_check:
        d = ds['data']
        for i, (one_hot, mel) in enumerate(d):
            txt_original, mel_from_ds = dataset[i]
            if not torch.equal(mel, mel_from_ds):
                raise Exception("data mismatched.")
            if not torch.equal(one_hot, txt_original):
                raise Exception("data mismatched.")


def convert(trainer_spec, verbose=True):
    """
    Convert dataset to native torch tensor representation.

    :param trainer_spec:
    :param verbose:
    :return:
    """
    trainer_spec = ExperimentSpecs(verbose=verbose)
    training_set, validation_set, test_set = trainer_spec.get_audio_ds_files()
    model_spec = trainer_spec.get_model_spec()
    encoder_spec = model_spec.get_encoder()

    #
    train_dataset = TextMelLoader(encoder_spec, list(training_set.values()), "audio_raw")
    validation_dataset = TextMelLoader(encoder_spec, list(validation_set.values()), "audio_raw")
    test_dataset = TextMelLoader(encoder_spec, list(test_set.values()), "audio_raw")

    if verbose:
        logging.info("filter_length", encoder_spec.filter_length())
        logging.info("hop_length", encoder_spec.hop_length())
        logging.info("win_length", encoder_spec.win_length())
        logging.info("n_mel_channels", encoder_spec.n_mel_channels())
        logging.info("sampling_rate", encoder_spec.sampling_rate())
        logging.info("mel_fmin", encoder_spec.mel_fmin())
        logging.info("mel_fmax", encoder_spec.mel_fmax())

    convert_mel_to_data(encoder_spec, trainer_spec.get_dataset_dir(),
                        train_dataset, trainer_spec.use_dataset, "train")
    convert_mel_to_data(encoder_spec, trainer_spec.get_dataset_dir(),
                        validation_dataset, trainer_spec.use_dataset, "validate")
    convert_mel_to_data(encoder_spec, trainer_spec.get_dataset_dir(),
                        test_dataset, trainer_spec.use_dataset, "test")


# def handler(a,b=None):
#     """
#
#     :param a:
#     :param b:
#     :return:
#     """
#     sys.exit(1)
#
# def install_handler():
#     """
#
#     :return:
#     """
#     if sys.platform == "win32":
#         import win32api
#         win32api.SetConsoleCtrlHandler(handler, True)


def cleanup():
    """

    :return:
    """
    dist.destroy_process_group()


def signal_handler(sig, frame):
    dist.destroy_process_group()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
if os.name != 'nt':
    signal.signal(signal.SIGTSTP, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def init_distributed(spec=None, rank=0, world_size=0):
    """

    :return:
    """
    if spec is None:
        print("Empty trainer spec.")
        sys.exit()

    #  if self.rank != 0:
    os.environ['MASTER_ADDR'] = spec.get_master_address()
    os.environ['MASTER_PORT'] = spec.get_master_port()
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    logger.info("Distributed Available".format(torch.cuda.device_count()))
    logger.info("Distribute protocol nccl available {}".format(torch.distributed.is_nccl_available()))
    logger.info("Distribute protocol mpi available {}".format(torch.distributed.is_mpi_available()))
    logger.info("Distribute protocol glow available {}".format(torch.distributed.is_gloo_available()))
    logger.info("Distribute endpoint {} my rank {}".format(spec.get_backend(), rank))

    # Set cuda device so everything is done on the right GPU.
    # torch.cuda.set_device(self.rank % torch.cuda.device_count())
    logger.info("Set cuda device".format(rank % torch.cuda.device_count()))
    # Initialize distributed communication
    if rank == 0:
        host = socket.gethostname()
        address = socket.gethostbyname(host)
        logger.info("resolve hostname {}".format(host))
        logger.info("resolve hostname {}".format(address))

    torch.distributed.init_process_group(
            backend=spec.get_backend(),
            init_method=spec.dist_url(),
            world_size=world_size,
            rank=rank)
    print("Done init")
    logger.debug("Done initializing distributed")


def train(spec=None, cmd_args=None, device=None, verbose=True, cudnn_bench=False):
    """

    :param cmd_args:
    :param spec: trainer spec, a config
    :param cudnn_bench: if we need run cudnn bench
    :param device: device where run
    :param verbose: if we need verbose output
    :return:
    """
    if int(cmd_args.rank) == 0:
        logger.info("Staring rank zero node.")

    if spec.is_distributed_run():
        logger.info("Staring training in distributed settings. rank {} world size {}".format(args.rank, args.world_size))
        init_distributed(spec, int(args.rank), int(args.world_size))
        device = torch.device(f"cuda:{int(args.rank)}")
        torch.cuda.set_device(int(args.rank))
        dist.barrier()

    dataloader = Mel_Dataloader(spec, rank=cmd_args.rank, world_size=cmd_args.world_size, verbose=True)
    torch.backends.cudnn.enabled = True
    if cudnn_bench:
        torch.backends.cudnn.benchmark = True

    logger.debug("Torch allow matmul fp16 {}", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    logger.debug("Torch cudnn version, {}", torch.backends.cudnn.version())
    logger.debug("Torch backend openmp", torch.backends.openmp)

    Trainer(spec,
            dataloader,
            rank=int(args.rank),
            world_size=int(cmd_args.world_size),
            verbose=args.verbose, device=device).train()


def dataloader_dry(cmd_args, trainer_specs, verbose=False):
    """

    :return:
    """
    data_loader = Mel_Dataloader(trainer_specs, verbose=verbose)
    if cmd_args.benchmark:
        data_loader.create()
        data_loader.benchmark_read()


def set_random_seeds(random_seed=0):
    """

    :param random_seed:
    :return:
    """
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# CUDA_VISIBLE_DEVICES=0
# python -c "import torch;print(torch.version.cuda)"
# NCCL_IB_DISABLE=1
# os.environ["NCCL_DEBUG"] = "INFO"

def main(cmd_args):
    """

    :param cmd_args:
    :return:
    """
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer_spec = ExperimentSpecs(spec_config=cmd_args.config, verbose=False)
    if cmd_args.mode.strip().upper().lower() == 'standalone':
        trainer_spec.set_distributed(False)
        sys.exit(1)
    elif cmd_args.mode.strip().upper().lower() == 'distributed':
        trainer_spec.set_distributed(True)

    if trainer_spec.is_distributed_run():
        set_random_seeds(trainer_spec.seed())

    trainer_spec.model_files.build_dir()
    if cmd_args.train:
        train(spec=trainer_spec, cmd_args=cmd_args, device=_device)

    if trainer_spec.is_distributed_run():
        dist.destroy_process_group()


if __name__ == '__main__':
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--world_size', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--verbose', type=bool, default='store_true',
                        required=False, help='set verbose output')
    parser.add_argument('--train', type=bool, default=True,
                        required=False, help='set verbose output')
    parser.add_argument('--convert', type=bool, default=False,
                        required=False, help='set verbose output')
    parser.add_argument('--inference', type=bool, default=False,
                        required=False, help='set verbose output')
    parser.add_argument('--benchmark', type=bool, default=False,
                        required=False, help='set verbose output')
    parser.add_argument('--config', type=str, help='set config file',
                        default='config.yaml',
                        required=False)
    parser.add_argument('--mode', type=str, default="",
                        help='run trainer in distributed or standalone',
                        required=False)

    # parser.add_argument('--load', type=bool, default=False,
    #                     required=False, help='set verbose output')
    # level = logger.level("ERROR")
    # logger.info(f"LOGURU_LEVEL: {os.environ['LOGURU_LEVEL']}")
    # logger.remove()
    # logger.enable("__main__")
    # logger.add(sys.stderr, level=config.LOG_LEVEL)
    # logger.enable()
    args = parser.parse_args()
    cuda_device_count = torch.cuda.device_count()

    try:
        main(args)
    except FileNotFoundError as file_error:
        print("File not found ", str(file_error))
        sys.exit(2)
