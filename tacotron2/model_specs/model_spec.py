import pickle
from abc import ABC, abstractmethod, ABCMeta

from tacotron2.utils import fmt_print


class ModelSpec(ABC, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, verbose=False):
        self.verbose = verbose
