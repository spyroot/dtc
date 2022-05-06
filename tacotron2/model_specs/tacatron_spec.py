import os

from tensorboard.plugins.hparams import api as hp
import yaml
from torch.utils.tensorboard import SummaryWriter

import shutil
import logging
from pathlib import Path

from tacotron2.model_files import ModelFiles
from tacotron2.text.symbols import symbols
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


class TacotronSpec:
    """

    """
    def __init__(self, model_dict, dataset_dict):
        """

        :param model_dict:
        :param dataset_dict:
        """
        self.model_dict = model_dict
        self.dataset_dict = dataset_dict

    def filter_length(self):
        pass

    def hop_length(self):
        pass

    def win_length(self):
        pass

    def n_mel_channels(self):
        pass

    def sampling_rate(self):
        pass

    def mel_fmin(self):
        pass

    def mel_fmax(self):
        pass

    def max_wav_value(self):
        pass