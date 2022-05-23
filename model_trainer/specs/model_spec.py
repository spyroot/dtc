from abc import ABC, abstractmethod, ABCMeta


class ModelSpec(ABC, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, verbose=False):
        self.verbose = verbose

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
