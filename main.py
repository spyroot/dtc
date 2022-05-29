import argparse
import logging
import os
import pickle
import random
import signal
import sys
from functools import partial
from pathlib import Path
import socket

import numpy as np
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch
from loguru import logger

from model_loader.dataset_stft25 import SFTF2Dataset
from model_loader.ds_util import md5_checksum
from model_loader.stft_dataloader import SFTFDataloader
from model_loader.dataset_stft30 import SFTF3Dataset
from model_trainer.internal.call_interface import Callback
from model_trainer.internal.time_meter import TimeMeter
from model_trainer.internal.time_tracer import BatchTimer
from model_trainer.specs.dtc_spec import TacotronSpec, ModelSpecDTC
from model_trainer.trainer_specs import ExperimentSpecs, TrainerSpecError
from model_trainer.trainer import Trainer, TrainerError
import torch.distributed as dist
from tqdm import tqdm

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "2"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_SYNCER"] = "1"

import warnings

warnings.filterwarnings("ignore")


class ConverterError(Exception):
    """Base class for other exceptions"""
    pass


def convert_mel_to_data(encoder_spec: TacotronSpec,
                        dataset: SFTF2Dataset,
                        target_dir="",
                        meta_file="",
                        dataset_name="default",
                        data_type="all",
                        post_check=True,
                        verbose=True):
    """

    :param encoder_spec:  all parameter for SFTS encoder
    :param dataset: list.
    :param meta_file: a meta file used to generate a dataset.
    :param target_dir:  where we will put our final file.
    :param dataset_name:  a name of dataset that will be in file name.
    :param data_type:  just a name train, validate, test , dev etc.
    :param post_check: if we do sanity check post.
    :param verbose:
    :return:
    """
    # all data how SFT's generated in meta
    meta = dict(filter_length=encoder_spec.filter_length(),
                hop_length=encoder_spec.hop_length(),
                win_length=encoder_spec.win_length(),
                n_mel_channels=encoder_spec.n_mel_channels(),
                sampling_rate=encoder_spec.sampling_rate(),
                mel_fmin=encoder_spec.mel_fmin(),
                mel_fmax=encoder_spec.mel_fmax())

    ds_size = len(dataset)

    data = []
    for i in tqdm(range(0, ds_size), desc="Converting"):
        data.append(dataset[i])

    meta['data'] = data
    meta['meta_file'] = meta_file
    file_name = Path(target_dir) / f'{dataset_name}_{data_type}_num_sam_' \
                                   f'{len(dataset)}_filter_{encoder_spec.n_mel_channels()}.pt'
    print("Saving ", file_name)
    torch.save(meta, str(file_name))
    print("MD5 checksum", md5_checksum(str(file_name)))

    if not post_check:
        return

    ds = torch.load(file_name)
    print("Loading back and checking.")
    if verbose:
        logger.info(f"Dataset saved to a file: {file_name}")
        logger.info(f"Dataset filter length: {ds['filter_length']}")
        logger.info(f"Dataset mel channels: {ds['n_mel_channels']}")
        logger.info(f"Dataset contains records: {len(ds['data'])}")

    d = ds['data']
    for i, (one_hot, mel) in tqdm(enumerate(d), total=len(ds['data']), desc="Validating"):
        txt_original, mel_from_ds = dataset[i]
        if not torch.equal(mel, mel_from_ds):
            raise ConverterError("data mismatched.")
        if not torch.equal(one_hot, txt_original):
            raise ConverterError("data mismatched.")

    print("Done.")


def convert(trainer_spec, version=2, dataset_name=None, merge=True, verbose=True, target_dir=None):
    """
    Routine convert dataset to native torch tensor representation.

    :param target_dir:
    :param version: a dataset version.  since we're serializing a tensor or numpy we need know what feature
                    on top of MEL we extract.
    :param trainer_spec: a trainer spec object.
    :param verbose: verbose output
    :param dataset_name: if empty will use current active one. whatever in config use_dataset: 'mydataset'
    :param merge:  if true merge all datasets to single one.
    :return:
    """

    # trainer_spec = ExperimentSpecs(verbose=verbose)
    if dataset_name is None or len(dataset_name) == 0:
        ds_spec = trainer_spec.get_active_dataset_spec()
        if not trainer_spec.is_audio_raw(ds_spec):
            print("Please check configuration, the active dataset not raw audio.")
            return

    if dataset_name is None or len(dataset_name) == 0:
        data = trainer_spec.get_audio_dataset()
    else:
        dataset_names = trainer_spec.get_dataset_names()
        ds_name = dataset_name.strip()
        if ds_name in dataset_names:
            data = trainer_spec.get_audio_dataset(dataset_name=ds_name)

    if data is None:
        raise ConverterError("Dataset not found.")

    if 'data' in data:
        raise ConverterError("Data has no key ds_type active dataset must be raw audio.")

    # 'train_meta', 'validation_meta', ['test_meta'] ['train_set'] ['validation_set'] ['test_set']

    training_set = data['train_set']
    validation_set = data['validation_set']
    test_set = data['test_set']

    model_spec: ModelSpecDTC = trainer_spec.get_model_spec()
    encoder_spec = model_spec.get_encoder()

    train_listified = list(training_set.values())
    val_listified = list(validation_set.values())
    test_listified = list(test_set.values())

    if merge:
        final_list = [*train_listified, *val_listified, *test_listified]
    #
    if version == 2:
        train_dataset = SFTF2Dataset(model_spec=encoder_spec,
                                     data=train_listified,
                                     data_format="audio_raw",
                                     verbose=verbose)
        validation_dataset = SFTF2Dataset(model_spec=encoder_spec,
                                          data=val_listified,
                                          data_format="audio_raw",
                                          verbose=verbose)
        test_dataset = SFTF2Dataset(model_spec=encoder_spec,
                                    data=test_listified,
                                    data_format="audio_raw",
                                    verbose=verbose)
    else:
        train_dataset = SFTF3Dataset(model_spec=encoder_spec,
                                     data=train_listified,
                                     data_format="audio_raw",
                                     verbose=verbose)
        validation_dataset = SFTF3Dataset(model_spec=encoder_spec,
                                          data=val_listified,
                                          data_format="audio_raw",
                                          verbose=verbose)
        test_dataset = SFTF3Dataset(model_spec=encoder_spec,
                                    data=test_listified,
                                    data_format="audio_raw",
                                    verbose=verbose)
    #
    if verbose:
        logging.info("filter_length", encoder_spec.filter_length())
        logging.info("hop_length", encoder_spec.hop_length())
        logging.info("win_length", encoder_spec.win_length())
        logging.info("n_mel_channels", encoder_spec.n_mel_channels())
        logging.info("sampling_rate", encoder_spec.sampling_rate())
        logging.info("mel_fmin", encoder_spec.mel_fmin())
        logging.info("mel_fmax", encoder_spec.mel_fmax())

    # by default target we read form specs
    if target_dir is not None:
        p = Path(target_dir)
        expanded = p.expanduser()
        resolved = expanded.resolve()
        if target_dir.exists() and target_dir.is_dir():
            final_dir = resolved
        else:
            raise ConverterError("can't resolve target dir.")
    else:
        final_dir = trainer_spec.get_dataset_dir()

    print("Going to write to dir", final_dir)

    ds_spec = trainer_spec.get_dataset_spec(dataset_name=dataset_name)
    files = [
        Path(ds_spec['dir']).expanduser() / ds_spec['training_meta'],
        Path(ds_spec['dir']).expanduser() / ds_spec['validation_meta'],
        Path(ds_spec['dir']).expanduser() / ds_spec['test_meta']
    ]

    # print(files)
    # print(data['train_meta'])
    # sys.exit(1)

    convert_mel_to_data(encoder_spec, train_dataset,
                        dataset_name=dataset_name,
                        meta_file=data['train_meta'],
                        target_dir=final_dir,
                        data_type="train")

    convert_mel_to_data(encoder_spec, validation_dataset,
                        dataset_name=dataset_name,
                        target_dir=final_dir,
                        meta_file=data['validation_meta'],
                        data_type="validate")

    convert_mel_to_data(encoder_spec, test_dataset,
                        dataset_name=dataset_name,
                        target_dir=final_dir,
                        meta_file=data['test_meta'],
                        data_type="test")


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


def cleanup(is_distributed: bool) -> None:
    """

    :param is_distributed:
    :return:
    """
    if is_distributed:
        dist.destroy_process_group()


def signal_handler(sig, frame) -> None:
    if is_distributed:
        dist.destroy_process_group()
    print("handling signal")
    sys.exit(0)


def setup_handler(handler):
    """

    :return:
    """
    if sys.platform == "win32":
        import win32api
        win32api.SetConsoleCtrlHandler(handler, True)

    signal.signal(signal.SIGINT, signal_handler)
    if os.name != 'nt':
        signal.signal(signal.SIGTSTP, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def init_distributed(spec=None, rank=0, world_size=0) -> None:
    """
    Routine for distributed training.

    :param spec:
    :param rank:
    :param world_size:
    :return:
    """

    if spec is None:
        print("Empty trainer spec.")
        sys.exit()

    # if rank != 0:
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


class Trainable(tune.Trainable, Callback):
    def setup(self, config):

        self.config = config
        spec = config['spec']

        batch_size = 0
        if 'batch_size' in config:
            spec.set_batch_size(int(config['batch_size']))
            assert spec.batch_size() == int(config['batch_size'])
        else:
            batch_size = spec.batch_size()

        SFTFDataloader.set_logger(False)
        data = SFTFDataloader(config['spec'],
                              batch_size=batch_size,
                              rank=int(args.rank),
                              world_size=config['world_size'],
                              verbose=False)

        Trainer.set_logger(False)
        self.trainer = Trainer(config['spec'],
                               data_loader=data,
                               rank=int(args.rank),
                               world_size=config['world_size'],
                               verbose=False,
                               device=config['device'],
                               hp_tunner=True,
                               disable_pbar=True)

        self.trainer.set_logger(False)

    def on_epoch_begin(self):
        # loss = validation_loss)
        pass

    def on_epoch_end(self):
        pass

    def save_checkpoint(self, tmp_checkpoint_dir):
        print("called save_checkpoint with tmp dir ", tmp_checkpoint_dir)
        for model in self._models:
            for model_layer in self.trainer._models[model]:
                checkpoint_path = os.path.join(tmp_checkpoint_dir, f"{model}{model_layer}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):

        print("called save_checkpoint with tmp dir ", tmp_checkpoint_dir)
        for model in self._models:
            for model_layer in self.trainer._models[model]:
                checkpoint_path = os.path.join(tmp_checkpoint_dir, f"{model}{model_layer}.pth")
                self.model.load_state_dict(torch.load(checkpoint_path))

    def step(self):
        self.trainer.train_optimizer(self.config)
        self.trainer.metric.batch_val_loss.mean()
        print(f"###### returned {self.trainer.metric.val_loss.mean()}")
        return {"mean_val_loss": self.trainer.metric.val_loss.mean()}

    # def reset_config(self, new_config):
    #     self.trainer.update_optimizer(new_config)
    #     for param_group in self.optimizer.param_groups:
    #         if "lr" in new_config:
    #             param_group["lr"] = new_config["lr"]
    #         if "momentum" in new_config:
    #             param_group["momentum"] = new_config["momentum"]
    #
    #     self.model = ConvNet()
    #     self.config = new_config
    #     return True


def tune_hyperparam(spec=None, cmd_args=None, device=None, cudnn_bench=False):
    """

    :param spec:
    :param cmd_args:
    :param device:
    :param cudnn_bench:
    :return:
    """
    spec.set_logger(False)
    if int(cmd_args.rank) == 0:
        logger.info("Staring rank zero node.")

    if spec.is_distributed_run():
        logger.info("Staring training in distributed settings. "
                    "rank {} world size {}".format(args.rank, args.world_size))
        init_distributed(spec, int(args.rank), int(args.world_size))
        # device = torch.device(f"cuda:{int(0)}")
        # device = torch.device(f"cuda:{dist.get_rank()}")
        # device = torch.device(device)
        dist.barrier()

    if cmd_args.overfit:
        spec.set_overfit()

    torch.backends.cudnn.enabled = True
    if cudnn_bench:
        torch.backends.cudnn.benchmark = True

    if args.verbose:
        logger.debug("Torch allow matmul fp16 {}", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        logger.debug("Torch cudnn version, {}", torch.backends.cudnn.version())
        logger.debug("Torch backend openmp", torch.backends.openmp)

    try:
        metric = 'mean_val_loss'

        scheduler = ASHAScheduler(
                metric=metric,
                mode="min",
                max_t=spec.epochs(),
                grace_period=1,
                reduction_factor=2)

        reporter = CLIReporter(
                # parameter_columns=["l1", "l2", "lr", "batch_size"],
                metric_columns=[metric, "training_iteration"])

        config = {
            # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            "spec": spec,
            "world_size": int(0),
            "device": device,
            "lr": ray.tune.loguniform(1e-4, 1e-1),
            "batch_size": ray.tune.choice([2, 4, 8, 16, 32, 64])
        }

        tuner_result = ray.tune.run(Trainable,
                                    resources_per_trial={"cpu": 4, "gpu": 1},
                                    config=config,
                                    num_samples=10,
                                    scheduler=scheduler,
                                    checkpoint_freq=2,
                                    local_dir=spec.model_files.get_tuner_log_dir(),
                                    stop={
                                        # "mean_accuracy": 0.95,
                                        "training_iteration": 3 if cmd_args.smoke_test else 20,
                                    },
                                    max_concurrent_trials=1,
                                    progress_reporter=reporter)

        best_trial = tuner_result.get_best_trial(metric, "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result[metric]))
        # print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))
        # print('best config: ', analysis.get_best_config(metric="score", mode="max"))
        # tuner_result = ray.tune.run(
        #         tune.with_parameters(
        #                 Trainer(spec,
        #                         SFTFDataloader(spec, rank=cmd_args.rank,
        #                                        world_size=cmd_args.world_size,
        #                                        verbose=args.verbose),
        #                         rank=int(args.rank),
        #                         world_size=int(cmd_args.world_size),
        #                         verbose=args.verbose, device=device,
        #                         callback=[BatchTimer()],
        #                         config=config,
        #                         checkpoint_dir=spec.model_files.get_tuner_dir())
        #         ),
        #         resources_per_trial={"gpu": 1},
        #         config=config,
        #         num_samples=10,
        #         scheduler=scheduler,
        #         stop={"training_iteration": 5},
        #         progress_reporter=reporter)

    except TrainerError as e:
        print("Error: trainer error: ", e)
        cleanup(spec.is_distributed_run())
        sys.exit(10)
    except Exception as other:
        print(other)
        raise other


def train(spec=None, cmd_args=None, device=None, cudnn_bench=False):
    """
    Main routine for to train a models.

    :param cmd_args:
    :param spec: trainer spec, a config
    :param cudnn_bench: if we need run cudnn bench
    :param device: device where run
    :return:
    """
    if int(cmd_args.rank) == 0:
        logger.info("Staring rank zero node.")

    if args.tune:
        tune_hyperparam(spec, cmd_args, device, cudnn_bench)

    if spec.is_distributed_run():
        logger.info("Staring training in distributed settings. "
                    "rank {} world size {}".format(args.rank, args.world_size))
        init_distributed(spec, int(args.rank), int(args.world_size))
        # device = torch.device(f"cuda:{int(0)}")
        # device = torch.device(f"cuda:{dist.get_rank()}")
        # device = torch.device(device)
        dist.barrier()

    if cmd_args.overfit:
        spec.set_overfit()

    torch.backends.cudnn.enabled = True
    if cudnn_bench:
        torch.backends.cudnn.benchmark = True

    if args.verbose:
        logger.debug(f"Torch allow matmul fp16 {torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")
        logger.debug(f"Torch cudnn version, {torch.backends.cudnn.version()}")
        logger.debug(f"Torch backend openmp {torch.backends.openmp}")
    try:

        dataloader = SFTFDataloader(spec, rank=cmd_args.rank,
                                    world_size=cmd_args.world_size,
                                    verbose=args.verbose)
        dataloader.set_logger(is_enable=True)

        # trainer = Trainer(spec,
        #                   dataloader,
        #                   rank=int(args.rank),
        #                   world_size=int(cmd_args.world_size),
        #                   verbose=args.verbose, device=device,
        #                   callback=[BatchTimer()])

        trainer = Trainer(spec,
                          dataloader,
                          rank=int(args.rank),
                          world_size=int(cmd_args.world_size),
                          verbose=args.verbose, device=device)

        trainer.set_logger(is_enable=True)
        trainer.metric.set_logger(is_enable=True)
        trainer.train()

    except TrainerError as e:
        print("Error: trainer error: ", e)
        cleanup(spec.is_distributed_run())
        sys.exit(10)
    except Exception as other:
        print(other)
        raise other


def dataloader_dry(cmd_args, trainer_specs):
    """
    Routine pass dry run over dataset and read time.
    # TODO add batch size.
    :return:
    """
    data_loader = SFTFDataloader(trainer_specs, verbose=cmd_args.verbose)
    if cmd_args.benchmark:
        data_loader._create()
        data_loader.benchmark_read()


def set_random_seeds(random_seed=1234):
    """
    Routine called when we run in DDP mode.
    It fixes all seed values.

    :param random_seed:
    :return:
    """
    logger.info("Setting random seed for torch.")
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def inference(spec: ExperimentSpecs, cmd_args, device):
    """
    Main routine for inference.

    :param spec:
    :param cmd_args:
    :param device:
    :return:
    """
    if cmd_args.model_file:
        model_path = Path(cmd_args.model_file)
        if model_path.exists() and model_path.is_file():
            trainer = Trainer(spec, rank=int(args.rank),
                              world_size=int(cmd_args.world_size),
                              verbose=args.verbose, device=device, is_inference=True)
            trainer.load_for_inference(model_name="encoder", model_file=str(model_path.resolve()))
            logger.info("Model loaded.")
            text = "Hello world, I missed you so much."
            trainer.inference(input_seq=text,
                              model_name="encoder",
                              mel_output_path=str(spec.model_files.get_figure_dir() / "mel_out.png"),
                              mel_post_path=str(spec.model_files.get_figure_dir() / "mel_post.png"),
                              mel_alignment_path=str(spec.model_files.get_figure_dir() / "mel_alignment.png"),
                              plot=True)
            # tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32')
            # tacotron2 = tacotron2.to(device)
            # tacotron2.eval()

        else:
            print("Error: File does not exist {}".format(cmd_args.model_file))
            sys.exit()


def main(cmd_args):
    """

    :param cmd_args:
    :return:
    """
    if cmd_args.device_id >= 0:
        logger.info("Manually setting cuda device.")
        _device = torch.device(f"cuda:{int(cmd_args.device_id)}" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer_spec = ExperimentSpecs(spec_config=cmd_args.config, verbose=cmd_args.verbose)
    trainer_spec.set_logger(cmd_args.verbose)

    if cmd_args.batch_size is not None:
        trainer_spec.set_batch_size(int(cmd_args.batch_size))
    if cmd_args.batch_size is not None:
        trainer_spec.set_epochs(int(cmd_args.epochs))

    if cmd_args.dataset_name is not None and len(cmd_args.dataset_name) > 0:
        spec = trainer_spec.get_dataset_spec(cmd_args.dataset_name)
        trainer_spec.set_active_dataset(dataset_name=str(cmd_args.dataset_name))

    logger.add(trainer_spec.model_files.get_model_log_file_path(), rotation=None,
               format="{elapsed} {level} {message}",
               filter="model_trainer.trainer_metrics", level="INFO")

    logger.add(trainer_spec.model_files.get_trace_log_file("loader"),
               format="{elapsed} {level} {message}",
               filter="model_loader.stft_dataloader", level="INFO")

    logger.add(trainer_spec.model_files.get_trace_log_file("trainer"), rotation=None,
               format="{elapsed} {level} {message}",
               filter="model_trainer.trainer", level="INFO")

    if cmd_args.mode.strip().upper().lower() == 'standalone':
        trainer_spec.set_distributed(False)
    elif cmd_args.mode.strip().upper().lower() == 'distributed':
        trainer_spec.set_distributed(True)
    else:
        print("Unknown option supported standalone | distributed, default standalone.")

    if trainer_spec.is_distributed_run():
        set_random_seeds(trainer_spec.seed())

    if cmd_args.convert:
        convert(trainer_spec, dataset_name=cmd_args.dataset_name, verbose=cmd_args.verbose)
        return

    trainer_spec.model_files.build_dir()
    if cmd_args.train:
        train(spec=trainer_spec, cmd_args=cmd_args, device=_device)
        return

    if cmd_args.train and trainer_spec.is_distributed_run():
        logger.info("Number cuda devices {}".format(torch.cuda.device_count()))
        dist.destroy_process_group()

    if args.inference:
        inference(spec=trainer_spec, cmd_args=cmd_args, device=_device)
        return


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


if __name__ == '__main__':
    """
    """
    logger.remove()

    set_logger(False)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', type=str,
                        help='Path to a pre-trained model.')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints.')
    parser.add_argument('--debug', action="store_true",
                        required=False, help='set debug output.')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('--warm', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--world_size', type=int, default=0,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='run trainer in distributed or standalone',
                        required=False)
    parser.add_argument('--device_id', type=int, default=0,
                        help='run trainer in distributed or standalone',
                        required=False)
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--verbose', action="store_true",
                        required=False, help='set verbose output')
    parser.add_argument('--overfit', action='store_true',
                        required=False, help='if set will reduce dataset and set batch 1 and overfit.')
    parser.add_argument('--tune', action='store_true',
                        required=False, help='run ray hyperparameter optimization.')
    parser.add_argument('--smoke_test', action='store_true',
                        required=False, help='run ray hyperparameter optimization in smoke test.')
    parser.add_argument('--train', type=bool, default=True,
                        required=False, help='set verbose output.')
    parser.add_argument('--convert', action="store_true",
                        required=False, help='convert dataset.')
    parser.add_argument('--dataset_name', type=str,
                        required=False, help='by default convert will take active one or we can overwrite.')
    parser.add_argument('--inference', action="store_true",
                        required=False, help='set model in inference model.')
    parser.add_argument('--benchmark', type=bool, default=False,
                        required=False, help='set verbose output')
    parser.add_argument('--config', type=str, help='set config file',
                        default='config.yaml', required=False)
    parser.add_argument('--epochs', type=int, help='overwrites epoch in config file', required=False)
    parser.add_argument('--batch_size', type=int, help='overwrites batch_size in config file', required=False)
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

    # we set all CUDA and NCCL debug flags.
    if args.debug is True:
        logger.info("Switching nccl and cuda debug flags")
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.local_rank >= 0:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    if args.rank >= 0:
        os.environ["RANK"] = str(args.rank)
    if args.world_size >= 0:
        os.environ["WORLD_SIZE"] = str(args.world_size)
    if args.device_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    logger.info("Training setting  "
                "rank: {} local rank: {} world_size {} cuda device {}".format(os.environ["RANK"],
                                                                              os.environ["LOCAL_RANK"],
                                                                              os.environ["WORLD_SIZE"],
                                                                              os.environ["CUDA_VISIBLE_DEVICES"]))
    if args.inference:
        logger.info("Model in inference mode, switching training off.")
        args.train = False

    if args.train:
        logger.info("Model in training mode.")

    is_distributed = False
    if args.mode.strip().upper().lower() == 'standalone':
        is_distributed = True

    try:
        set_logger(args.verbose)
        # trainer_spec = ExperimentSpecs(spec_config=args.config, verbose=args.verbose)
        main(args)
        # setup_handler(cleanup(is_distributed))
    except FileNotFoundError as file_error:
        print("File not found ", str(file_error))
        logger.error(f"File not found: {str(file_error)}")
    except ConverterError as convert_err:
        print("Dataset converter error ", str(convert_err))
        logger.error(f"Dataset converter error: {str(convert_err)}")
    except TrainerSpecError as spec_error:
        print("Invalid spec:", str(spec_error))
        logger.error(f"Invalid spec: {str(spec_error)}")
        sys.exit(2)
