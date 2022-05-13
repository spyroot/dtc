from abc import ABC

from .model_spec import ModelSpec
from .tacatron_spec import TacotronSpec


class DTC(ModelSpec, ABC):
    """

    """

    def __init__(self, transcoder_spec, generator_spec, verbose=False):
        """

        :param transcoder_spec:
        :param generator_spec:
        """
        super(DTC, self).__init__(verbose=verbose)
        self._model_dict = transcoder_spec
        self._generator_param_dict = generator_spec
        self._encoder_spec = TacotronSpec(transcoder_spec['encoder'])

    def get_model_param(self):
        return self._model_dict, self._generator_param_dict

    def get_encoder(self) -> TacotronSpec:
        return self._encoder_spec

    def sub_models(self):
        return self._model_dict

    def get_sub_models_names(self):
        return list(self._model_dict.keys())
