# all main command line tooling
#
# All trainer logic is abstracted in the generic trainer.
# During the initial start, the trainer takes specs i.e., the yaml file invokes
# the factory method. That creates a model, a model-specific optimizer, and a scheduler.
#
# Note my assumption that the model can be stacked. Hence internally, it queues.
# So, for example, if you have two models and you train, you can define two sub-layers.
#
#  A good example is if you want to train DTC with a different Vocoder,
# or, for instance, Tacotron 2 and WaveGlow
#
# Note right trainer is logically executed in a sequence generally backward
# in torch implementation and can not be executed in parallel anyway.
#
# Hence queue is just FIFO.
#
# Ray loaded optionally.
# Mus
import argparse
import logging
import os
import random
import signal
import socket
import sys
import warnings
from pathlib import Path
from typing import Optional

import scipy
import torch
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from loguru import logger
import soundfile as sf

try:
    import ray
    from ray import tune
    from tune import CLIReporter
    from tune.schedulers import ASHAScheduler
    from tunner import Trainable
except ImportError:
    pass

from tqdm import tqdm

from inference_tools import plot_spectrogram
from model_loader.dataset_stft25 import SFTF2Dataset
from model_loader.dataset_stft30 import SFTF3Dataset
from model_loader.ds_util import md5_checksum
from model_loader.stft_dataloader import SFTFDataloader
from model_trainer.internal.save_best import CheckpointBest
from model_trainer.plotting_utils import plot_spectrogram_to_numpy
from model_trainer.specs.model_tacotron25_spec import SpectrogramLayerSpec, ModelSpecTacotron25
from model_trainer.trainer import Trainer, TrainerError
from model_trainer.trainer_specs import ExperimentSpecs, TrainerSpecError

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "2"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_SYNCER"] = "1"

warnings.filterwarnings("ignore")


class ConverterError(Exception):
    """Base class for other exceptions"""
    pass


def convert_mel_to_data(encoder_spec: SpectrogramLayerSpec,
                        dataset: SFTF2Dataset,
                        target_dir: Optional[str] = "",
                        meta_file: Optional[str] = "",
                        dataset_name: Optional[str] = "default",
                        data_type: Optional[str] = "all",
                        version: Optional[int] = 3,
                        post_check: Optional[bool] = True,
                        verbose: Optional[bool] = True):
    """
    Convert audio dataset to MEL tensor representation.

    TODO add meta v2/v3 and detect during data loader creation.

    :param version:
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
    meta['version'] = version
    file_name = Path(target_dir) / f'{dataset_name}_{data_type}_num_sam_' \
                                   f'{len(dataset)}_filter_' \
                                   f'{encoder_spec.n_mel_channels()}_{version}.pt'
    md5_sig = Path(target_dir) / f'{dataset_name}_{data_type}_num_sam_' \
                                 f'{len(dataset)}_filter_' \
                                 f'{encoder_spec.n_mel_channels()}_{version}.sig'

    print("Saving ", file_name)
    torch.save(meta, str(file_name))
    md5checksum = md5_checksum(str(file_name))
    print("MD5 checksum", md5checksum)
    with open(md5_sig, 'w', encoding='utf-8') as f:
        f.write(md5checksum)

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
    for i, (data_tuple) in tqdm(enumerate(d), total=len(ds['data']), desc="Validating"):
        if version == 2:
            txt_original, mel_from_ds = dataset[i]
            one_hot, mel = data_tuple
        elif version == 3:
            txt_original, mel_from_ds, sft_ds = dataset[i]
            one_hot, mel, stft = data_tuple
            if not torch.equal(stft, sft_ds):
                raise ConverterError("data mismatched.")
        else:
            raise ConverterError("Unknown version.")

        if not torch.equal(mel, mel_from_ds):
            raise ConverterError("data mismatched.")
        if not torch.equal(one_hot, txt_original):
            raise ConverterError("data mismatched.")

    print("Done.")


def plot_example(trainer_spec, version=3, dataset_name=None, verbose=True, target_dir=None):
    """
    Routine plot dataset example for inspection.
    Each MEl and STFT generated in result folder.  SampleS TFT reconstructed by inverse
    function and serialized in result directory.

    :param target_dir: where to write plots
    :param version: a dataset version.  since we're serializing a tensor or numpy we need know what feature
                    on top of MEL we extract.
    :param trainer_spec: a trainer spec object.
    :param verbose: verbose output
    :param dataset_name: if empty will use current active one. whatever in config use_dataset: 'mydataset'
    :param merge: if true merge all datasets to single one.
    :return:
    """

    # trainer_spec = ExperimentSpecs(verbose=verbose)
    if dataset_name is None or len(dataset_name) == 0:
        ds_spec = trainer_spec.get_active_dataset_spec()
        # if not trainer_spec.is_audio_raw(ds_spec):
        #     print("Please check configuration, the active dataset not raw audio.")
        #     return

    if dataset_name is None or len(dataset_name) == 0:
        data = trainer_spec.get_audio_dataset()
    else:
        dataset_names = trainer_spec.get_dataset_names()
        ds_name = dataset_name.strip()
        if ds_name in dataset_names:
            data = trainer_spec.get_audio_dataset(dataset_name=ds_name)

    if data is None:
        raise ConverterError("Dataset not found.")

    dataloader = SFTFDataloader(trainer_spec, batch_size=2, verbose=True)
    loaders, collate = dataloader.get_loader()

    # get all
    start_time = time.time()
    data_loaders, collate_fn = dataloader.get_all()
    _train_loader = data_loaders['train_set']
    print(f"Train datasize {dataloader.get_train_dataset_size()}")
    print("--- %s load time, seconds ---" % (time.time() - start_time))
    fig = plt.figure()

    start_time = time.time()
    # take one batch exampe.
    for bidx, batch in enumerate(_train_loader):
        text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, stft_padded = batch

        plot_spectrogram_to_numpy(mel_padded[0], file_name="results/default_mel.png")
        plot_spectrogram(stft_padded[0],
                         y_axis_label="mel freq",
                         file_name="results/default_stft.png")

        # MEL
        S = librosa.feature.inverse.mel_to_stft(mel_padded[0].numpy())
        y = librosa.griffinlim(S)
        sf.write('results/default.wav', y, 22050, 'PCM_24')

        # iSTFT
        y_numpy = stft_padded.numpy()
        for i in range(0, y_numpy.shape[0]):
            n = (y_numpy[bidx].shape[i] * y_numpy[i].shape[1])
            y_out = librosa.istft(y_numpy[i], length=n)
            sf.write(f'results/default_stft_{i}.wav', y_out, 22050, 'PCM_24')
        break

    fig.show()
    print("--- %s Single batch memory load, load time, seconds ---" % (time.time() - start_time))


def convert(trainer_spec, version=3,
            dataset_name=None, merge=True, verbose=True, target_dir=None,
            ds_ratio=10, exclude=None):
    """
    Routine convert dataset to native torch tensor representation.  It takes entire audio dataset.
    and create single dat file. It supports both Tacotron2 format and DTC.

    Each file serialized to disk and md5 hash provided.  In case we need provide download option.
    BaseDataset class provide list of url that provide option to fetch url and dataset.

    Each entries require mirror and md5 hash.

    I used this method in my training procedure It significantly increases
    batch load to GPU time.

    :param exclude:
    :param ds_ratio:
    :param target_dir:
    :param version: a dataset version.  since we're serializing a tensor or numpy
                    we need know what feature on top of MEL we extract.
    :param trainer_spec: a trainer spec object.
    :param verbose: verbose output
    :param dataset_name: if empty will use current active one. whatever in config use_dataset: 'mydataset'
    :param merge:  if true merge all datasets to single one.
    :return:
    """
    # trainer_spec = ExperimentSpecs(verbose=verbose)
    if exclude is None:
        exclude = ['validation_set']

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

    training_set = data['train_set']
    validation_set = data['validation_set']
    test_set = data['test_set']

    model_spec: ModelSpecTacotron25 = trainer_spec.get_model_spec()
    encoder_spec = model_spec.get_spectrogram()

    train_listified = list(training_set.values())
    val_listified = list(validation_set.values())
    test_listified = list(test_set.values())
    if ds_ratio > 0:
        if 'train_set' not in exclude:
            old_sz = len(train_listified)
            train_listified = train_listified[0: int(len(train_listified) * (ds_ratio / 100))]
            print(f"Train set reduced from {old_sz} to {len(train_listified)}.")
        if 'validation_set' not in exclude:
            old_sz = len(val_listified)
            val_listified = val_listified[0: int(len(val_listified) * (ds_ratio / 100))]
            print(f"Train set reduced from {old_sz} to {len(val_listified)}.")
        if 'test_set' not in exclude:
            old_sz = len(test_listified)
            test_listified = test_listified[0: int(len(test_listified) * (ds_ratio / 100))]
            print(f"Test set reduced from {old_sz} to {len(test_listified)}.")

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
        logging.info(f"filter_length {encoder_spec.filter_length()}")
        logging.info(f"hop_length {encoder_spec.hop_length()}")
        logging.info(f"win_length {encoder_spec.win_length()}")
        logging.info(f"n_mel_channels {encoder_spec.n_mel_channels()}")
        logging.info(f"sampling_rate {encoder_spec.sampling_rate()}")
        logging.info(f"mel_fmin {encoder_spec.mel_fmin()}")
        logging.info(f"mel_fmax {encoder_spec.mel_fmax()}")

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

    # by default we generate 3 seperate files.
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


def cleanup(is_dist: bool) -> None:
    """

    :param is_dist:
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


def tune_hyperparam(spec=None, cmd_args=None, device=None, cudnn_bench=False):
    """

    Note Ray tunned pulled to a separate class and conditionally include
    since ray has an issue with Python 3.10, I test code 3.9 and 3.10.

    Specs for ray in same trainer spec.

    Example will test batch size variation,  lr , grad clip rate.
    ray:
      batch_size: [32, 64]
      lr_min: 1e-4
      lr_max: 1e-1
      num_samples: 10
      checkpoint_freq: 4
      resources:
        cpu: 4
        gpu: 1
      attention_location_filters: 32
      attention_kernel_size: 31
      grad_clip:
        min: 0.5
        max: 1.0

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
        dist.barrier()

    if cmd_args.overfit:
        spec.set_overfit()

    torch.backends.cudnn.enabled = True
    if cudnn_bench:
        torch.backends.cudnn.benchmark = True

    if args.verbose:
        logger.debug(f"Torch allow matmul fp16 "
                     f"{torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")
        logger.debug(f"Torch cudnn version, "
                     f"{torch.backends.cudnn.version}")
        logger.debug(f"Torch backend openmp "
                     f"{torch.backends.openmp}")
        logger.debug(f"Torch backend openmp "
                     f"{torch.version.cuda}")
        logger.debug(f"Torch backend openmp"
                     f"{torch.backends.cudnn.version()}")
        logger.debug(f"Torch backend openmp "
                     f"{torch.__version__}")
        logger.debug(f"Torch backend openmp "
                     f"{torch.cuda.get_device_name(0)}")
        logger.debug(f"Torch backend openmp "
                     f"{torch.cuda.get_device_properties(0)}")

    try:

        metric = 'mean_train_loss'
        scheduler = ASHAScheduler(
                metric=metric,
                mode="min",
                max_t=100,
                grace_period=1,
                reduction_factor=2)

        reporter = CLIReporter(
                metric_columns=[metric, "training_iteration"])

        ray_spec = spec.get_tuner_spec()

        config = {
            "spec": spec,
            "world_size": int(0),
            "device": device,
        }

        if 'batch_size' in ray_spec:
            config['batch_size'] = tune.choice(ray_spec['batch_size'])

        if 'lr_min' in ray_spec and 'lr_max' in ray_spec:
            config['lr'] = tune.loguniform(ray_spec['lr_min'], ray_spec['lr_max'])

        if 'num_samples' not in ray_spec:
            print("You need indicate num_samples in spec. It mandatory for Ray to function. ")
            return

        if 'checkpoint_freq' not in ray_spec:
            print("Please checkpoint_freq, It essential for Ray to checkpoint.")
            return

        if 'resources' not in ray_spec:
            print("Please indicate resources ray can use.")
            return

        resources = ray_spec['resources']

        # optional
        if 'grad_clip' in ray_spec:
            grad_clip_spec = ray_spec['grad_clip']
            if 'min' in grad_clip_spec and 'max' in grad_clip_spec:
                config['grad_clip'] = tune.loguniform(float(grad_clip_spec['min']),
                                                      float(grad_clip_spec['max']))

        tuner_result = tune.run(Trainable,
                                resources_per_trial={"cpu": int(resources['cpu']),
                                                     "gpu": int(resources['gpu'])},
                                config=config,
                                num_samples=ray_spec['num_samples'],
                                scheduler=scheduler,
                                checkpoint_freq=ray_spec['checkpoint_freq'],
                                local_dir=spec.model_files.get_tuner_log_dir(),
                                stop={
                                    # "mean_accuracy": 0.95,
                                    "training_iteration": 2 if cmd_args.smoke_test else 20,
                                },
                                max_concurrent_trials=1,
                                progress_reporter=reporter)

        best_trial = tuner_result.get_best_trial(metric, "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final train loss: {}".format(best_trial.last_result[metric]))

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

        trainer = Trainer(spec,
                          dataloader,
                          rank=int(args.rank),
                          world_size=int(cmd_args.world_size),
                          verbose=args.verbose, device=device,
                          callback=[CheckpointBest()],
                          hp_tunner=False)

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
    data_loader = SFTFDataloader(trainer_specs, verbose=cmd_args._verbose)
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


def inference(spec: ExperimentSpecs, cmd_args, device, sigma=1.0, sampling_rate=22050, denoiser_strength=0.2,
              max_wav_values=32768.0):
    """
    Main routine for inference.

    :param max_wav_values:
    :param denoiser_strength:
    :param sampling_rate:
    :param sigma:
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
            trainer.load_for_inference(model_name="dtc",
                                       layer_name="spectrogram_layer",
                                       model_file=str(model_path.resolve()))

            logger.info("Model loaded.")
            # text = "Hello world, I missed you so much."
            text = "Meat was no longer issued raw, to be imperfectly cooked before a ward fire and bolted gluttonously, the whole two pounds at one sitting."
            mel_outputs, mel_outputs_postnet, alignments = \
                trainer.inference(input_seq=text,
                                  model_name="dtc",
                                  mel_output_path=str(spec.model_files.get_figure_dir() / "mel_out.png"),
                                  mel_post_path=str(spec.model_files.get_figure_dir() / "mel_post.png"),
                                  mel_alignment_path=str(spec.model_files.get_figure_dir() / "mel_alignment.png"),
                                  plot=True)

            glow_path = f"{spec.model_files.get_model_dir()}/waveglow_256channels_universal_v5.pt"
            model_file = Path(glow_path)
            if not model_file.exists():
                print("Please download glow model and put to a results/model dir.")
                print("url: https://drive.google.com/u/0/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF&export=download")
                return

            from waveglow.denoiser import Denoiser
            sys.path.insert(0, './waveglow')
            waveglow = torch.load(str(model_file))['model']
            waveglow = waveglow.remove_weightnorm(waveglow)
            waveglow.cuda().eval()

            if denoiser_strength > 0:
                denoiser = Denoiser(waveglow).cuda()

            mel = torch.autograd.Variable(mel_outputs_postnet.cuda())
            # mel = torch.unsqueeze(mel_outputs, 0)
            with torch.no_grad():
                audio = waveglow.infer(mel, sigma=sigma)
                if denoiser_strength > 0:
                    audio = denoiser(audio, denoiser_strength)
                audio = audio * max_wav_values

            audio = audio.squeeze()
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')

            audio_path = os.path.join(spec.model_files.get_results_dir(), "{}_synthesis.wav".format("test.wav"))
            scipy.io.wavfile.write(audio_path, sampling_rate, audio)

            # from scipy.io import wavfile
            # from pesq import pesq
            #
            # rate, ref = wavfile.read(audio_path)
            # rate, deg = wavfile.read("./audio/speech_bab_0dB.wav")

            #  print(pesq(rate, ref, deg, 'wb'))
            # print(pesq(rate, ref, deg, 'nb'))

            print(audio_path)
        else:
            print("Error: File does not exist {}".format(cmd_args.model_file))
            sys.exit()


def load_glow(spec: ExperimentSpecs):
    """

    :param spec:
    :return:
    """
    glow_path = f"{spec.model_files.get_model_dir()}/waveglow_256channels_universal_v5.pt"
    model_file = Path(glow_path)
    if not model_file.exists():
        print("Please download glow model and put to a results/model dir.")
        print("url: https://drive.google.com/u/0/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF&export=download")
        return

    from waveglow.denoiser import Denoiser
    sys.path.insert(0, './waveglow')
    waveglow = torch.load(str(model_file))['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    return waveglow


def glow_inference(waveglow, mel, sigma=1.0, denoiser_strength=0.2, max_wav_valuest=32768.0):
    """

    :param max_wav_valuest:
    :param denoiser_strength:
    :param mel:
    :param sigma:
    :param waveglow:
    :return:
    """
    from waveglow.denoiser import Denoiser
    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    mel = torch.autograd.Variable(mel.cuda())
    # mel = torch.unsqueeze(mel_outputs, 0)
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)
        if denoiser_strength > 0:
            audio = denoiser(audio, denoiser_strength)
        audio = audio * max_wav_valuest

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')

    return audio


def load_dtc(spec: ExperimentSpecs, device,
             model_path: Optional[str],
             model_name: Optional[str] = "dtc",
             layer_name: Optional[str] = "spectrogram_layer"):
    """
    :param model_path:
    :param device:
    :param model_name:
    :param layer_name:
    :param spec:
    :return:
    """
    trainer = Trainer(spec, verbose=False, device=device, is_inference=True)
    trainer.load_for_inference(model_name=model_name,
                               layer_name=layer_name,
                               model_file=model_path)
    return trainer


def metric(spec: ExperimentSpecs, cmd_args, device,
           model_name: Optional[str] = "dtc",
           layer_name: Optional[str] = "spectrogram_layer",
           num_sample: Optional[int] = 10,
           glow_file_name: Optional[str] = "waveglow_256channels_universal_v5.pt",
           sampling_rate: Optional[int] = 22050):
    """
    Main routine for PESQ computation.

    :param layer_name:
    :param model_name:
    :param glow_file_name:
    :param num_sample:
    :param sampling_rate:
    :param spec:
    :param cmd_args:
    :param device:
    :return:
    """
    if not cmd_args.model_file:
        print("Please indicate model file in argument --model_file.")
        return

    model_path = Path(cmd_args.model_file)
    if model_path.exists() and model_path.is_file():

        trainer = load_dtc(spec=spec, model_name=model_name,
                           layer_name=layer_name,
                           device=device, model_path=str(model_path))

        glow_path = f"{spec.model_files.get_model_dir()}/{glow_file_name}"
        model_file = Path(glow_path)
        if not model_file.exists():
            print("Please download glow model and put to a results/model dir.")
            print("url: https://drive.google.com/u/0/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF&export=download")
            return

        waveglow = None
        try:
            waveglow = load_glow(spec)
        except Exception as err:
            print(f"Failed to load glow model. err {err}")

        dataset_files = spec.get_audio_dataset()
        ds_keys = spec.get_audio_dataset_keys()

        assert len(ds_keys) > 0
        assert len(dataset_files) > 0

        # for keys ['train_set', 'validation_set', 'test_set'] we sample K times
        for _, k in enumerate(ds_keys):
            if k not in dataset_files:
                continue
            ds = list(dataset_files[k].values())
            pesq_wbs = np.zeros((num_sample, 1))
            pesq_nss = np.zeros((num_sample, 1))
            for i, dataset_rec in enumerate(ds):
                if i == num_sample:
                    break
                print(f"Sampled for {k} {i}-th example")

                text_seq = dataset_rec['meta']
                print(f"Generating inference for {text_seq.strip()} "
                      f"original path {dataset_rec['path']}")

                file_name = Path(dataset_rec['path'])
                mel_outputs, mel_outputs_post, alignments = \
                    trainer.inference(input_seq=text_seq, model_name=model_name, plot=False)
                audio = glow_inference(waveglow, mel_outputs_post)
                audio_path = os.path.join(spec.model_files.get_generated_dir(),
                                          "{}_synthesis.wav".format(file_name.name))
                scipy.io.wavfile.write(audio_path, sampling_rate, audio)
                print(f"Saving audio {audio_path}")

                # down sample since peqq need either 8k or 16k
                ref, s = librosa.load(dataset_rec['path'], sr=16000)
                deg, s = librosa.load(audio_path, sr=16000)

                # now we read both files.
                from pesq import pesq
                pesq_wb = pesq(16000, ref, deg, 'wb')
                pesw_nb = pesq(16000, ref, deg, 'nb')
                pesq_wbs[i] = pesq_wb
                pesq_nss[i] = pesw_nb
                print(f"{file_name.name} : {pesq_wb}")
                print(f"{file_name.name} : {pesw_nb}")

            print(f"Average pesq for {k} : {pesq_wbs.mean()} {pesq_wbs.mean()}")

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

    _start_time = time.time()
    trainer_spec = ExperimentSpecs(spec_config=cmd_args.config, verbose=cmd_args.verbose)
    trainer_spec.set_logger(cmd_args.verbose)
    print("--- %s Parser load time, seconds ---" % (time.time() - _start_time))

    # cmd overwrite batch size from spec.
    if cmd_args.batch_size is not None:
        trainer_spec.set_batch_size(int(cmd_args.batch_size))
    if cmd_args.batch_size is not None:
        trainer_spec.set_epochs(int(cmd_args.epochs))

    # similarly active dataset
    if cmd_args.dataset_name is not None and len(cmd_args.dataset_name) > 0:
        spec = trainer_spec.get_dataset_spec(cmd_args.dataset_name)
        trainer_spec.set_active_dataset(dataset_name=str(cmd_args.dataset_name))

    logger.add(trainer_spec.model_files.get_model_log_file_path(remove_old=True),
               format="{elapsed} {level} {message}",
               filter="model_trainer.trainer_metrics", level="INFO", rotation="1h")

    logger.add(trainer_spec.model_files.get_trace_log_file("loader"),
               format="{elapsed} {level} {message}",
               filter="model_loader.stft_dataloader", level="INFO", rotation="1h")

    logger.add(trainer_spec.model_files.get_trace_log_file("trainer"),
               format="{elapsed} {level} {message}",
               filter="model_trainer.trainer", level="INFO", rotation="1h")

    if cmd_args.mode.strip().upper().lower() == 'standalone':
        trainer_spec.set_distributed(False)
    elif cmd_args.mode.strip().upper().lower() == 'distributed':
        trainer_spec.set_distributed(True)
    elif len(cmd_args.mode) == 0:
        trainer_spec.set_distributed(False)
    else:
        print("Unknown option supported standalone | distributed, default standalone.")

    if trainer_spec.is_distributed_run():
        set_random_seeds(trainer_spec.seed())

    if cmd_args.plot:
        plot_example(trainer_spec, dataset_name=cmd_args.dataset_name,
                     version=3, verbose=cmd_args.verbose)
        return

    if len(cmd_args.load) > 0:
        resolved = Path(cmd_args.load).expanduser().resolve()
        if not resolved.exists():
            print("Error, File not found.")
            return
        if cmd_args.model is None or len(cmd_args.model) == 0:
            print(cmd_args.model)
            print("Error. If you want load model from a file, Please indicate a model name. --model")
            return
        if cmd_args.remove_opt is True:
            model = torch.load(cmd_args.load)
            del model["optimizer_state_dict"]
            torch.save(model, f"{cmd_args.load}.new.dat")
            return
        if cmd_args.freeze is True:
            model = torch.load(cmd_args.load)
            print(model.keys())
            return
        if cmd_args.show is True:
            model = torch.load(cmd_args.load)
            if 'epoch' in model:
                print(f"Model trained {model['epoch']} epochs.")
            if 'it' in model:
                print(f"Model trained {model['it']} steps.")
            print(model.keys())
            return
        print(f"Loading model {cmd_args.model} from {cmd_args.load}")
        trainer_spec.set_active_model(model_name=cmd_args.model)
        if not trainer_spec.model_files.update_model_file(cmd_args.load):
            print(f"Error; Please check path to a file.")
            return

    if cmd_args.convert:
        if 0 > cmd_args.convert_size > 100:
            print("Please indicated value between 1...100.")
            return
        convert(trainer_spec, dataset_name=cmd_args.dataset_name,
                version=3, verbose=cmd_args.verbose, ds_ratio=cmd_args.convert_size)
        return

    if cmd_args.tune:
        try:
            import ray
            return tune_hyperparam(spec=trainer_spec, cmd_args=cmd_args, device=_device)
        except ImportError:
            print("You need install ray first.")
            return

    trainer_spec.model_files.build_dir()
    if cmd_args.metric:
        metric(spec=trainer_spec, cmd_args=cmd_args, device=_device)
        return

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



# import numpy as np
#
# def v_pesq2mos(p):
#     a, b, c, d = p
#     if np.isempty(a):
#         a= 0.999
#         b= 4.999 - a
#         c= -1.4945
#         d = 4.6607
#      if nargout > 0:
#         m= a + b . / (1 + exp (c * p + d))
#      else
#         if nargin < 1 or np.isempty(p):
#             pp=np.linspace(-0.5,4.5,100);
#         else:
#             pp=p

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
                        required=False, help='set verbose output.')
    parser.add_argument('--remove_opt', action="store_true",
                        required=False, help='remove optimizer from checkpoint.')
    parser.add_argument('--show', action="store_true",
                        required=False, help='show loaded model.')
    parser.add_argument('--freeze', action="store_true",
                        required=False, help='freeze a model.')
    parser.add_argument('--plot', action="store_true",
                        required=False, help='plot examples from a dataset.')
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
    parser.add_argument('--convert_size', type=int, default=0,
                        required=False, help='convert dataset with specific ration.')
    parser.add_argument('--dataset_name', type=str,
                        required=False, help='by default convert will take active one or we can overwrite.')
    parser.add_argument('--dataset_version', type=int, default=3,
                        required=False, help='by default convert will save in version 3. '
                                             'note that it much bigger since contain sfts.')
    parser.add_argument('--inference', action="store_true",
                        required=False, help='set model in inference model.')

    parser.add_argument('--metric', action="store_true",
                        required=False, help='set model in metric model.')

    parser.add_argument('--benchmark', type=bool, default=False,
                        required=False, help='set verbose output')
    parser.add_argument('--config', type=str, help='set config file',
                        default='config.yaml', required=False)
    parser.add_argument('--epochs', type=int, help='overwrites epoch in config file', required=False)
    parser.add_argument('--batch_size', type=int, help='overwrites batch_size in config file', required=False)
    parser.add_argument('--mode', type=str, default="", help='run trainer in distributed or standalone',
                        required=False)
    parser.add_argument('--load', type=str, default="",
                        help='load model from a file. argument path to a file.', required=False)
    parser.add_argument('--model', type=str, default="dtc", help='model name. note load and model '
                                                                 'name mainly used if we need explicit load from file',
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
    if args.metric:
        args.train = False

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
