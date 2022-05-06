import os
from os import listdir
from os.path import isfile, join

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
    """

    """

    def __init__(self, root_dir=".", name_generator=None):
        """

        :param root_dir:
        """
        self.config = None
        self.name_generator = None

        self.root_dir = root_dir
        self._dir_input = root_dir
        self._dir_result = Path(self._dir_input) / Path(self.results_dir())
        self._dir_log = Path(self._dir_result) / Path(self.log_dir())
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

        self.generate_file_name_template()

    def generate_file_name_template(self):
        """
         Generates file name templates.
        """
        if self.name_generator is not None:
            self.filename = self.name_generator()
        else:
            self.filename = "default_"

        # self.filename = "{}_{}_{}_layers_{}_hidden_{}_".format(self.active,
        #                                                        self.active_model,
        #                                                        self.graph_type,
        #                                                        str(self._num_layers()),
        #                                                        str(self._hidden_size_rnn))
        self.filename_prediction = self.filename + 'predictions_'
        self.filename_train = self.filename + 'train_'
        self.filename_test = self.filename + 'test_'
        self.filename_metrics = self.filename + 'metric_'
        self.filename_time_traces = self.filename + 'timetrace_'

        self.training_files = None

    def results_dir(self) -> str:
        """
        Return main directory where all results stored.
        """
        if self.config is not None and 'results_dir' in self.config:
            return self.config['results_dir']
        return 'results'

    def log_dir(self) -> str:
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

    def make_file_dict(self, target_dir, ext=None):
        """
        Recursively walk and build a dict where key is file name,
        value is dict that store path and metadata.

        :param target_dir:
        :return:
        """
        target_files = {}
        for dir_path, dir_names, filenames in os.walk(target_dir):
            for a_file in filenames:
                if ext is not None and a_file.find(ext) != -1:
                    target_files[a_file] = {'path': join(dir_path, a_file), 'meta': '27', 'label': '0'}

        return target_files
