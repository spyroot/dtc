from abc import ABC
from .model_spec import ModelSpec
from .spectrogram_layer_spec import SpectrogramLayerSpec
from loguru import logger
import attr


class InvalidModelSpec(Exception):
    """Base class for other exceptions"""
    pass


# @attr.s(frozen = True)
class ModelSpecDTS(ModelSpec, ABC):
    """
    MODEL SPEC {'spectrogram_layer': {'model': 'tacotron25', 'optimizer': 'tacotron2_optimizer', 'has_input': True,
     'has_output': True, 'max_wav_value': 32768.0, 'frames_per_step': 1, 'sampling_rate': 22050, 'filter_length': 1024,
     'win_length': 1024, 'hop_length': 256, 'n_mel_channels': 80, 'mel_fmin': 0.0, 'mel_fmax': 8000.0},
     'vocoder': {'state': 'disabled', 'name': 'Test', 'model': 'GraphLSTM',
     'optimizer': 'edge_optimizer', 'lr_scheduler': 'main_lr_scheduler', 'input_size': 1}}

    """
    def __init__(self, model_spec, dataset_spec, verbose=False) -> None:
        """
        :param model_spec:
        :param dataset_spec:
        """
        super(ModelSpecDTS, self).__init__(verbose=verbose)
        self.set_logger(verbose)

        logger.debug("Creating model spec dts", model_spec, dataset_spec)

        self._model_dict = model_spec

        self._generator_param_dict = dataset_spec

        # list of sub-models
        self._sub_models = {}

        # todo refactor this
        if 'spectrogram_layer' not in model_spec:
            raise InvalidModelSpec("Model must contains spectrogram_layer.")

        self._spectrogram_spec = SpectrogramLayerSpec(model_spec['spectrogram_layer'])
        self._sub_models['spectrogram_layer'] = SpectrogramLayerSpec(model_spec['spectrogram_layer'])

    def get_model_param(self):
        """
        Return model hyperparameter.
        :return:
        """
        return self._model_dict, self._generator_param_dict

    def get_spectrogram(self) -> SpectrogramLayerSpec:
        """

        :return:
        """
        return self._spectrogram_spec

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

    def get_spec(self, name: str):
        """
        :param name:
        :return:
        """
        if name not in self._sub_models:
            raise InvalidModelSpec(f"{name} not found.")

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
