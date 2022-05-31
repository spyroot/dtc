import os
import sys
import warnings
from pathlib import Path
from typing import List, Callable, Optional
from typing import Type

import torch
import yaml
from loguru import logger

import model_loader
from model_trainer.utils import fmt_print
from text.symbols import symbols
from .model_files import ModelFiles
from .specs.model_tacotron25_spec import ModelSpecTacotron25
from .specs.model_dts_spec import ModelSpecDTS
from .specs.model_spec import ModelSpec
from model_loader.ds_util import check_integrity


class SpecsDispatcher:
    """
    Specs dispatcher main router for factory methods.
    """
    def __init__(self, verbose=False):
        """
        :param verbose:
        """
        self._verbose: bool = False
        self.dispatchers = self._create_spec_dispatch()

    def _create_spec_dispatch(self) -> dict[str, Callable]:
        """
        Create spec dispatch.
        each model might contain many hparam and settings.
        Method creates  a separate dispatcher, key is string and value to factory callable.
        we just dispatch entire spec.

        During spec parsing,  we dispatch to respected class.

        :return: model creator Callable , and trainer creator Callable.
        """
        model_dispatch = {
            'tacotron25': self.tacotron25_creator,
            'dts': self.dts_creator,
        }
        return model_dispatch

    @staticmethod
    def tacotron25_creator(model_spec, dataset_spec, verbose: bool):
        """
        Create spec, for dts.
        :return:
        """
        return ModelSpecTacotron25(model_spec, dataset_spec, verbose=verbose)

    @staticmethod
    def dts_creator(model_spec, dataset_spec, verbose: bool):
        """
        Create spec, for dts.
        :return:
        """
        return ModelSpecDTS(model_spec, dataset_spec, verbose=verbose)

    def get_model(self, model_name: str):
        """
        :param model_name:
        :return:
        """
        if model_name in self.dispatchers:
            return self.dispatchers[model_name]
        else:
            raise ValueError("Unknown model name")

    def register(self, model_name, creator: Callable):
        """
        Register a model and respected callable factory
        callable.

        :param model_name:
        :param creator:
        :return:
        """
        self.dispatchers[model_name] = creator

    def has_creator(self, model_name: str) -> bool:
        """
        This mainly externally we can check if dispatcher has a
        factory method.

        :param model_name:
        :return:
        """
        if model_name not in self.dispatchers:
            return False
        return True
