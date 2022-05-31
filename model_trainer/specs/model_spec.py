from abc import ABC, abstractmethod, ABCMeta
from loguru import logger


class ModelSpec(ABC, metaclass=ABCMeta):
    """

    """
    @abstractmethod
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        self.set_logger(verbose)

    @abstractmethod
    def get_model_param(self):
        pass

    @abstractmethod
    def sub_models(self):
        pass

    @abstractmethod
    def get_sub_models_names(self):
        pass

    @abstractmethod
    def get_spec(self, name: str):
        pass

    @abstractmethod
    def get_spectrogram(self):
        pass

    @staticmethod
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

