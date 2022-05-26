class TacotronSpecError(Exception):
    """Base class for other exceptions"""
    pass


class TacotronSpec:
    """

models:
  # this pure model specific, single model can describe both edges and nodes
  # in case we need use single model for edge and node prediction task ,
  # use keyword single_model: model_name
  tacotron25:
    spectrogram_layer:
      model: tacotron25
      optimizer: tacotron2_optimizer
#      lr_scheduler: main_lr_scheduler
      has_input: True
      has_output: True
      max_wav_value: 32768.0
      frames_per_step: 1
      sampling_rate: 22050
      filter_length: 1024   # length of the FFT window
      win_length: 1024      # each frame of audio is windowed by
      hop_length: 256
      n_mel_channels: 80
      mel_fmin: 0.0
      mel_fmax: 8000.0
      symbols_embedding_dim: 512

    """

    def __init__(self, model_dict):
        """

        :param model_dict:
        """
        self._model_dict = model_dict
        self.n_frames_per_step = 1

    def filter_length(self):
        if 'filter_length' not in self._model_dict:
            raise TacotronSpecError("Model has no filter_length defined.")
        return self._model_dict['filter_length']

    def frames_per_step(self):
        if 'frames_per_step' not in self._model_dict:
            raise TacotronSpecError("Model has no frames_per_step defined.")
        return self._model_dict['frames_per_step']

    def hop_length(self):
        if 'hop_length' not in self._model_dict:
            raise TacotronSpecError("Model has no sampling_rate defined.")
        return self._model_dict['hop_length']

    def win_length(self):
        if 'win_length' not in self._model_dict:
            raise TacotronSpecError("Model has no win_length defined.")
        return self._model_dict['win_length']

    def n_mel_channels(self):
        if 'n_mel_channels' not in self._model_dict:
            raise TacotronSpecError("Model has no n_mel_channels defined.")
        return self._model_dict['n_mel_channels']

    def sampling_rate(self) -> int:
        if 'sampling_rate' not in self._model_dict:
            raise TacotronSpecError("Model has no sampling_rate defined.")
        return self._model_dict['sampling_rate']

    def mel_fmin(self) -> float:
        if 'mel_fmin' not in self._model_dict:
            raise TacotronSpecError("Model has no mel_fmin defined.")
        return self._model_dict['mel_fmin']

    def mel_fmax(self) -> float:
        if 'mel_fmax' not in self._model_dict:
            raise TacotronSpecError("Model has no mel_fmax defined.")
        return self._model_dict['mel_fmax']

    def max_wav_value(self) -> float:
        if 'max_wav_value' not in self._model_dict:
            raise TacotronSpecError("Model has no max_wav_value defined.")
        return self._model_dict['max_wav_value']

    def get_text_cleaner(self):
        """
        :return:
        """
        if 'sampling_rate' not in self._model_dict:
            raise TacotronSpecError("Model has no sampling_rate defined.")
        return ['english_cleaners']

    def get_seed(self):
        """

        :return:
        """
        return 1234

    def __str__(self):
        return str(self._model_dict)

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
