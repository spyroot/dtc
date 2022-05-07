import os
from os import listdir
from os.path import isfile, join

from tensorboard.plugins.hparams import api as hp
import yaml
from torch.utils.tensorboard import SummaryWriter

import shutil
import logging
from pathlib import Path

# from tacotron2.text.symbols import symbols
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
import torch


class ModelFiles:
    """

    """

    def __init__(self, config, root_dir=".", name_generator=None):
        """

        :param root_dir:
        """
        self._verbose = None
        self.config = config
        self.name_generator = None

        self.root_dir = root_dir
        self._dir_input = root_dir
        self._dir_result = Path(self._dir_input) / Path(self.spec_results_dir())
        self._dir_log = Path(self._dir_result) / Path(self.spec_log_dir())
        self._model_save_path = self._dir_result / Path(self.model_save_dir())

        self._dir_graph_save = self._dir_result / Path(self.graph_dir())
        self._dir_figure = self._dir_result / Path(self.figures_dir())
        self._dir_timing = self._dir_result / Path(self.timing_dir())

        # default dir where we store serialized prediction graph as image
        self._dir_model_prediction = self._dir_result / Path(self.prediction_dir())

        self.filename = None
        self.filename_prediction = None
        self.filename_train = None
        self.filename_test = None
        self.filename_metrics = None
        self.filename_time_traces = None

        self._active_model = self.config['use_model']
        self._active_setting = self.config['active_setting']

        self._models = self.config['models']
        if self._active_model not in self._models:
            raise Exception("config.yaml doesn't contain model {}.".format(self._active_model))

        self._model = self._models[self._active_model]
        _settings = self.config['settings']
        self._setting = _settings[self._active_setting].copy()

        self.generate_file_name_template()

    def generate_file_name_template(self):
        """
         Generates file name templates.
        """
        if self.name_generator is not None:
            self.filename = self.name_generator()
        else:
            self.filename = self._active_model

        self.filename = "{}_{}".format(self._active_model, self.config['active_setting'])

        self.filename_prediction = self.filename + 'predictions_'
        self.filename_train = self.filename + 'train_'
        self.filename_test = self.filename + 'test_'
        self.filename_metrics = self.filename + 'metric_'
        self.filename_time_traces = self.filename + 'timetrace_'

        self.training_files = None

    def spec_results_dir(self) -> str:
        """
        Return main directory where all results stored.
        """
        if self.config is not None and 'results_dir' in self.config:
            return self.config['results_dir']
        return 'results'

    def spec_log_dir(self) -> str:
        """
        Return directory that used to store logs.
        """
        if self.config is not None and 'log_dir' in self.config:
            return self.config['log_dir']
        return 'logs'

    def graph_dir(self) -> str:
        """
        Return directory where store original graphs
        """
        if self.config is not None and 'graph_dir' in self.config:
            return self.config['graph_dir']
        return 'graphs'

    def timing_dir(self) -> str:
        """
        Return directory we use to store time traces
        """
        if self.config is not None and 'timing_dir' in self.config:
            return self.config['timing_dir']
        return 'timing'

    def model_save_dir(self) -> str:
        """
        Default directory where model checkpoint stored.
        """
        if self.config is not None and 'model_save_dir' in self.config:
            return self.config['model_save_dir']

        return 'model_save'

    def prediction_dir(self) -> str:
        """
        Default directory where model prediction serialized.
        """
        if self.config is not None and 'figures_prediction_dir' in self.config:
            return self.config['figures_prediction_dir']

        return 'prediction'

    def prediction_figure_dir(self) -> str:
        """
        Default directory where model prediction serialized.
        """
        if self.config is not None and 'figures_prediction_dir' in self.config:
            return self.config['prediction_figures']
        return 'prediction'

    def figures_dir(self) -> str:
        """
        Default directory where test figures serialized.
        """
        if self.config is not None and 'figures_dir' in self.config:
            return self.config['figures_dir']
        return 'figures'

    def build_dir(self):
        """
        Creates all directories required for trainer.
        """
        if not os.path.isdir(self._model_save_path):
            os.makedirs(self._model_save_path)

        if not os.path.isdir(self._dir_log):
            os.makedirs(self._dir_log)

        if not os.path.isdir(self._dir_graph_save):
            os.makedirs(self._dir_graph_save)

        if not os.path.isdir(self._dir_figure):
            os.makedirs(self._dir_figure)

        if not os.path.isdir(self._dir_timing):
            os.makedirs(self._dir_timing)

        if not os.path.isdir(self._dir_model_prediction):
            os.makedirs(self._dir_model_prediction)

    def make_file_dict(self, target_dir, file_ext=None, filter_dict=None):
        """
        Recursively walk and build a dict where key is file name,
        value is dict that store path and metadata.

        :param target_dir:
        :return:
        """
        target_files = {}
        for dir_path, dir_names, filenames in os.walk(target_dir):
            for a_file in filenames:
                if file_ext is not None and a_file.find(file_ext) != -1:
                    if filter_dict is None:
                        target_files[a_file] = {'path': join(dir_path, a_file), 'meta': '', 'label': '0'}
                    else:
                        if a_file in filter_dict:
                            target_files[a_file] = {'path': join(dir_path, a_file), 'meta': '', 'label': '0'}

        return target_files

    def get_model_log_dir(self):
        """

        Returns:

        """
        return self._dir_log

    def get_model_filename(self, model_name, file_type='.dat'):
        """
        Returns dict that hold sub-model name and respected checkpoint filename.
        Args:
            model_name:
            file_type:

        Returns:

        """
        batch_size = 0
        if 'batch_size' in self._setting:
            batch_size = int(self._setting['batch_size'])

        for k in self._model:
            if k == model_name:
                return str(self._model_save_path / Path(self.filename + '_' + k + '_batch_' +
                                                        str(batch_size) +
                                                        '_' + str(self.load_epoch())
                                                        + file_type))

    def model_filenames(self, file_type='.dat'):
        """

        Returns dict that hold sub-model name and
        respected checkpoint filename.

        @param file_type:
        @return:
        """
        models_filenames = {}

        batch_size = 0
        if 'batch_size' in self._setting:
            batch_size = int(self._setting['batch_size'])

        print("######", batch_size)
        for k in self._model:
            models_filenames[k] = str(self._model_save_path / Path(self.filename +
                                                                   '_' + k + '_batch_' +
                                                                   str(batch_size) +
                                                                   '_' +
                                                                   str(self.load_epoch())
                                                                   + file_type))

        return models_filenames

    def load_epoch(self) -> int:
        """
        Setting dictates whether load model or not.
        """
        return int(self.config['load_epoch'])

    def is_trained(self) -> bool:
        """
        Return true if model trained, it mainly checks if dat file created or not.

        :return: True if trainer
        """
        models_filenames = self.model_filenames()
        if self._verbose:
            print("Model filenames", models_filenames)

        for k in models_filenames:
            if not os.path.isfile(models_filenames[k]):
                return False

        return True

    def get_last_saved_epoc(self):
        """
         Return last checkpoint saved as dict where key sub-model: last checkpoint
         If model un trained will raise exception
        """
        checkpoints = {}
        if self._verbose:
            fmtl_print('Trying load models last checkpoint...', self._active_model)

        if not self.is_trained():
            raise Exception("Untrained model")

        models_filenames = self.model_filenames()
        if self._verbose:
            print("Model filenames", models_filenames)

        for m in models_filenames:
            if self._verbose:
                print("Trying to load checkpoint file", models_filenames[m])

            check = torch.load(models_filenames[m])
            if self._verbose:
                print(check.keys())

            if 'epoch' in check:
                checkpoints[m] = check['epoch']

        return checkpoints
