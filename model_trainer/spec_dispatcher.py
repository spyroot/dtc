from typing import Callable

from .specs.model_dtc_spec import ModelSpecDTC
from .specs.model_tacotron25_spec import ModelSpecTacotron25


class SpecsDispatcher:
    """
    Specs dispatcher main router for factory methods.
    """
    def __init__(self, verbose=False):
        """
        :param verbose:
        """
        self._verbose: bool = verbose
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
            'dtc': self.dtc_creator,
        }
        return model_dispatch

    @staticmethod
    def tacotron25_creator(model_spec, dataset_spec, verbose: bool):
        """
        Create spec, for dtc.
        :return:
        """
        return ModelSpecTacotron25(model_spec, dataset_spec, verbose=verbose)

    @staticmethod
    def dtc_creator(model_spec, dataset_spec, verbose: bool):
        """
        Create spec, for dtc.
        :return:
        """
        return ModelSpecDTC(model_spec, dataset_spec, verbose=verbose)

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
