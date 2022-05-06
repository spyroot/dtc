import os

from tensorboard.plugins.hparams import api as hp
import yaml
from torch.utils.tensorboard import SummaryWriter

import shutil
import logging
from pathlib import Path

from tacotron2.text.symbols import symbols
from tacotron2.utils import fmtl_print, fmt_print

import argparse
import random
import sys
import time
from datetime import time
from datetime import timedelta
from typing import Final
from datetime import time
from datetime import timedelta

class ModelFiles:
    def __init__(self, root_dir=""):
        self.root_dir = root_dir
        self.dir_input = root_dir
        self.dir_result = Path(self.dir_input) / Path(self.results_dir())
        self.dir_log = Path(self.dir_result) / Path(self.log_dir())
        self.model_save_path = self.dir_result / Path(self.model_save_dir())

        self.dir_graph_save = self.dir_result / Path(self.graph_dir())
        self.dir_figure = self.dir_result / Path(self.figures_dir())
        self.dir_timing = self.dir_result / Path(self.timing_dir())
        # default dir where we store serialized prediction graph as image
        self.dir_model_prediction = self.dir_result / Path(self.prediction_dir())

        self.filename = None
        self.filename_prediction = None
        self.filename_train = None
        self.filename_test = None
        self.filename_metrics = None
        self.filename_time_traces = None


class ModelSpecs:
    """

    """

    def __init__(self, template_file_name='config.yaml', verbose=False):
        """

        :param template_file_name:
        :param verbose:
        """
        if isinstance(template_file_name, str):
            fmtl_print("Loading", template_file_name)

        # store poitne to config , after spec read serialize yaml to it.
        self.config = None
        # device
        self.device = 'cuda'
        fmtl_print("Device", self.device)

        # a file name or io.string
        self.config_file_name = template_file_name

        # if clean tensorboard
        self.clean_tensorboard = False

        self.epochs = 500
        self.iters_per_checkpoint = 1000
        self.seed = 1234
        self.fp16_run = False
        self.distributed_run = False
        self.dist_backend = "nccl"
        self.dist_url = "tcp://localhost:54321"
        self.cudnn_enabled = True
        self.cudnn_benchmark = False
        self.ignore_layers = ['embedding.weight']

        self.load_mel_from_disk = False,
        self.training_files = 'filelists/ljs_audio_text_train_filelist.txt',
        self.validation_files = 'filelists/ljs_audio_text_val_filelist.txt',
        self.text_cleaners = ['english_cleaners'],

        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

        ################################
        # Model Parameters             #
        ################################
        self.n_symbols = len(symbols)
        self.symbols_embedding_dim = 512

        # Encoder parameters
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = 512

        # Decoder parameters
        self.n_frames_per_step = 1  # currently only 1 is supported
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention parameters
        self.attention_rnn_dim = 1024
        self.attention_dim = 128

        # Location Layer parameters
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Mel-post processing network parameters
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate = False
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.grad_clip_thresh = 1.0
        self.batch_size = 64
        self.mask_padding = True  # set model's padded outputs to padded values
        self.dynamic_loss_scaling = True

        self.cudnn_enabled = True
        self.cudnn_benchmark = True

        # self.optimizer = HParam('optimizer', hp.Discrete(['adam', 'sgd']))

        def setup_tensorflow(self):
            """
            Setup tensorflow dir
            """
            time = time.strftime("%Y-%m-%d-%H", time.gmtime())
            fmt_print("tensorboard log dir", self.log_dir())
            logging.basicConfig(filename=str(self.dir_log / Path('train' + time + '.log')), level=logging.DEBUG)

            if bool(self.config['regenerate']):
                if os.path.isdir("tensorboard"):
                    shutil.rmtree("tensorboard")
            self.writer = SummaryWriter()
            # with SummaryWriter() as w:
            #     for i in range(5):
            #         w.add_hparams({'lr': 0.1 * i, 'bsize': i}, {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i})

            return self.writer
