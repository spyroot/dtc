import os

from tensorboard.plugins.hparams import api as hp
import yaml
from torch.utils.tensorboard import SummaryWriter

import shutil
import logging
from pathlib import Path

from .model_files import ModelFiles
from .dtc_spec import DTC
from text.symbols import symbols
from tacotron2.utils import fmtl_print, fmt_print
import torch

import argparse
import random
import sys
import time
from datetime import time
from datetime import timedelta
from typing import Final, List
from datetime import time
from datetime import timedelta
from .model_spec import ModelSpec


class ExperimentSpecs:
    """

    """

    def __init__(self, template_file_name='config.yaml', verbose=False):
        """

        :param template_file_name:
        :param verbose:
        """

        # a file name or io.string
        self.config_file_name = None
        current_dir = os.path.dirname(os.path.realpath(__file__))

        if isinstance(template_file_name, str):
            fmtl_print("Loading", template_file_name)
            # a file name or io.string
            self.config_file_name = Path(template_file_name)

        self._verbose = None

        #
        self._setting = None

        #
        self._model_spec = None

        #
        self.inited = None

        #
        self.writer = None

        #
        self._active_setting = None

        # list of models
        self.models_specs = None

        # active model
        self.active_model = None

        # dataset specs
        self.dataset_specs = None

        # active dataset
        self.use_dataset = None

        # store pointer to config, after spec read serialize yaml to it.
        self.config = None

        # device
        self.device = 'cuda'

        fmtl_print("Device", self.device)

        # model files
        self.model_files = ModelFiles()

        # if clean tensorboard
        self.clean_tensorboard = False

        self.ignore_layers = ['embedding.weight']

        self.load_mel_from_disk = False,
        self.text_cleaners = ['english_cleaners'],

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

        self.read_from_file()
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

    def models_list(self):
        """
        List of network types and sub-network models used for a given model.
        For example GraphRNN has graph edge and graph node model.
        @return: list of models.
        """
        models_types = []
        for k in self._model_spec:
            if k.find('model') != -1:
                models_types.append(k)

    def set_active_dataset(self):
        """

        :return:
        """
        if 'use_dataset' not in self.config:
            raise Exception("config.yaml must contains valid active settings.")
        self.use_dataset = self.config['use_dataset']

    def set_dataset_specs(self):
        """

        :return:
        """
        if 'datasets' not in self.config:
            raise Exception("config.yaml must contains corresponding "
                            "datasets settings for {}".format(self.use_dataset))

        dataset_list = self.config['datasets']
        if self.use_dataset not in dataset_list:
            raise Exception("config.yaml doesn't contain {} template, check config.".format(self.use_dataset))

        self.dataset_specs = self.config['datasets'][self.use_dataset]

    def set_active_model(self):
        """

        :return:
        """
        if 'models' not in self.config:
            raise Exception("config.yaml must contain at least one models list and one model.")
        if 'use_model' not in self.config:
            raise Exception("config.yaml must contain use_model and it must defined.")
        #
        self.active_model = self.config['use_model']
        #
        self.models_specs = self.config['models']
        #
        if self.active_model == 'dts':
            self._model_spec = DTC(self.models_specs[self.active_model], self.dataset_specs)

        if self.active_model not in self.models_specs:
            raise Exception("config.yaml doesn't contain model {}.".format(self.active_model))

        # set model spec
        #self.model_spec = self.models_specs[self.active_model]

    def set_active_settings(self, debug=False):
        """

        :param debug:
        :return:
        """
        self._active_setting = self.config['active_setting']
        _settings = self.config['settings']
        if debug:
            fmt_print("Settings list", _settings)

        if self._active_setting not in _settings:
            raise Exception("config.yaml use undefined variable {} ".format(self._active_setting))

        self._setting = _settings[self._active_setting].copy()
        if debug:
            fmt_print("Active settings", self._setting)

    def read_config(self, debug=False):
        """
        Parse config file and initialize trainer
        :param debug: will output debug into
        :return: nothing
        """
        if debug:
            fmtl_print("Parsing... ", self.config)

        self.set_active_dataset()
        self.set_dataset_specs()
        self.set_active_model()

        # settings stored internally
        if debug:
            fmt_print("active setting", self.config['active_setting'])

        self.set_active_settings()
        self.inited = True

    def read_from_file(self, debug=False):
        """
        Read config file and initialize trainer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(self.config_file_name, "r") as stream:
            try:
                fmtl_print("Reading... ", self.config_file_name)
                self.config = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

        self.read_config()

    def read_from_stream(self, buffer, debug=False):
        """
        Read config file from a stream
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            if self._verbose:
                print("Reading from io buffer")
            self.config = yaml.load(buffer, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit("Failed parse yaml")

        self.read_config()

    def get_model_sub_models(self) -> List[str]:
        """
        @return:  Return list of all sub models.
        """
        keys = self._model_spec.keys()
        return [k for k in keys]

    def text_parser(message):
        """Get the normalized list of words from a message string.

        This function should split a message into words, normalize them, and return
        the resulting list. For splitting, you should split on spaces. For normalization,
        you should convert everything to lowercase.

        Args:
            message: A string containing an SMS message

        Returns:
           The list of normalized words from the message.
        """
        # *** START CODE HERE ***
        mess = message.lower()
        return mess.split()
        # *** END CODE HERE ***

    def load_text_file(self, file_name, delim="|", filter='DUMMY/'):
        """Load from text file name and metadata (text)

        Args:
             file_name: file name.

        Returns:
            dict where key is file name and value text.
            :param filter:
            :param file_name:
            :param delim:
        """

        target_dir = self.dataset_specs['dir']
        file_path = Path(target_dir) / file_name
        print(str(file_path))

        file_meta_kv = {}
        with open(file_path, 'r', newline='', encoding='utf8') as meta_file:
            lines = meta_file.readlines()
            for line in lines:
                tokens = line.split(delim)
                if len(tokens) == 2:
                    file_name = tokens[0].replace(filter, '')
                    file_meta_kv[file_name] = tokens[1]

        return file_meta_kv

    def update_meta(self, metadata_file):
        """
        Build a dict that hold each file as key, a nested dict contains
        full path to a file,  metadata such as label , text , translation etc.
        :return:
        """
        file_meta_kv = self.load_text_file(metadata_file)
        files = self.model_files.make_file_dict(self.dataset_specs['dir'], "wav")

        for k in file_meta_kv:
            if k in files:
                files[k]['meta'] = file_meta_kv[k]

        return files

    def build_training_set(self):
        """
        :return:
        """
        if 'training_meta' in self.dataset_specs:
            return self.update_meta(self.dataset_specs['training_meta'])

        return {}

    def build_validation_set(self):
        """
        :return:
        """
        if 'training_meta' in self.dataset_specs:
            return self.update_meta(self.dataset_specs['validation_meta'])

        return {}

    def build_test_set(self):
        """
        :return:
        """
        if 'training_meta' in self.dataset_specs:
            return self.update_meta(self.dataset_specs['test_meta'])

        return {}

    def get_audio_ds_files(self):
        """
        :return:
        """
        return self.build_training_set(), self.build_validation_set(), self.build_test_set()

    def is_distributed_run(self):
        """

        :return:
        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'distributed' in self._setting:
            return self._setting['distributed']

        return False

    def tensorboard_sample_update(self):
        """
        Return true if early stopping enabled.
        :return:  default value False
        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'early_stopping' in self._setting:
            return True

    def seed(self):
        return 1234

    def is_fp16_run(self):
        return False

    def get_backend(self):
        pass

    def dist_url(self):
        pass

    def get_model_spec(self) -> ModelSpec:
        return self._model_spec

    def fp16_run(self):
        pass
