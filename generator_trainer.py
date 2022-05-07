import pickle
from abc import ABC, abstractmethod, ABCMeta

from tacotron2.utils import fmt_print


class GeneratorTrainer(ABC, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, verbose=False, is_notebook=True):
        self.verbose = verbose
        self.is_notebook = is_notebook

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load(self, model_name):
        pass

    @staticmethod
    def save_graphs(g, file_name, verbose=False):

        if file_name is None:
            raise Exception("File name is none.")

        if len(file_name) == 0:
            raise Exception("empty file name")

        if verbose:
            fmt_print("Saving graph to a file ", file_name)

        with open(file_name, "wb") as f:
            pickle.dump(g, f)

    def set_notebook(self, param):
        self.is_notebook = param

    def set_verbose(self, param):
        self.verbose = param