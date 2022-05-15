class TacotronSpec:
    """

    """
    def __init__(self, model_dict):
        """

        :param model_dict:
        """
        self._model_dict = model_dict
        self.n_frames_per_step = 1

    def filter_length(self):
        return self._model_dict['filter_length']

    def frames_per_step(self):
        return self._model_dict['frames_per_step']

    def hop_length(self):
        return self._model_dict['hop_length']

    def win_length(self):
        return self._model_dict['win_length']

    def n_mel_channels(self):
        return self._model_dict['n_mel_channels']

    def sampling_rate(self):
        return self._model_dict['sampling_rate']

    def mel_fmin(self):
        return self._model_dict['mel_fmin']

    def mel_fmax(self):
        return self._model_dict['mel_fmax']

    def max_wav_value(self):
        return self._model_dict['max_wav_value']

    def load_mel_from_disk(self):
        if 'is_load_mel' in self._model_dict:
            return self._model_dict['is_load_mel']
        return False

    def get_text_cleaner(self):
        """

        :return:
        """
        list = ['english_cleaners']
        return list

    def get_seed(self):
        """

        :return:
        """
        return 1234

    def __str__(self):
        return str(self._model_dict)
