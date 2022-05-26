from abc import ABC
from .model_spec import ModelSpec
from .tacatron_spec import TacotronSpec
from loguru import logger


class InvalidModelSpec(Exception):
    """Base class for other exceptions"""
    pass


class ModelSpecDTC(ModelSpec, ABC):
    """

    """
    def __init__(self, model_spec, dataset_spec, verbose=False):
        """
        :param model_spec:
        :param dataset_spec:
        """
        super(ModelSpecDTC, self).__init__(verbose=verbose)
        self.set_logger(verbose)

        logger.debug("Creating model spec dts", model_spec, dataset_spec)

        self._model_dict = model_spec
        self._generator_param_dict = dataset_spec
        # list of sub-models
        self._sub_models = {}

        # todo refactor this
        self._encoder_spec = TacotronSpec(model_spec['spectrogram_layer'])
        if 'spectrogram_layer' not in model_spec:
            raise InvalidModelSpec("Model must contains spectrogram_layer.")
        self._sub_models['spectrogram_layer'] = TacotronSpec(model_spec['spectrogram_layer'])

    def get_model_param(self):
        """
        Return model hyperparameter.
        :return:
        """
        return self._model_dict, self._generator_param_dict

    def get_encoder(self) -> TacotronSpec:
        """

        :return:
        """
        return self._encoder_spec

    def sub_models(self):
        """
        Method return dict that hold all specs for all sub_models.
        :return:
        """
        return self._model_dict

    def get_sub_models_names(self):
        """
        Return all model layer names.
        :return:
        """
        return list(self._model_dict.keys())

    def get_spec(self, name):
        """

        :param name:
        :return:
        """
        return self._sub_models[name]

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
