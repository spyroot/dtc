import random
from abc import abstractmethod

import torch
import torch.utils.data

from model_trainer.specs.tacatron_spec import TacotronSpec
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """

    """

    def __init__(self, model_spec: TacotronSpec, data, data_format,
                 fixed_seed=True, shuffle=False, is_trace_time=False):
        """

        :param model_spec:
        :param data:  data is dict must hold key data
        :param data_format: tensor_mel, numpy_mel, audio_raw
        :param is_trace_time:
        :param fixed_seed: if we want shuffle dataset
        :param shuffle: shuffle or not,  in case DDP you must not shuffle
        """
        # type tensor . numpy , file etc
        self.is_trace_time = is_trace_time
        self.shuffle = shuffle
        self.is_a_tensor = False
        self.is_a_numpy = False
        self.is_a_raw = False

        if data_format is None or len(data_format) == 0:
            raise Exception("Dataset file type format is none or empty")

        if 'tensor_mel' in data_format:
            self.is_a_tensor = True
        elif 'numpy_mel' in data_format:
            self.is_a_numpy = True
        elif 'audio_raw' in data_format:
            self.is_a_raw = True
        else:
            raise Exception("Dataset file type format is unsupported.")

        self._model_spec = model_spec
        # check dataset contain key
        if self.is_a_raw is False and 'data' not in data:
            raise Exception("Dataset dict doesn't contain key 'data'")

        if self.is_a_tensor:
            self._data = data['data']
        elif self.is_a_raw:
            self._data = data
        else:
            raise Exception("Unknown format.")

        self.text_cleaners = model_spec.get_text_cleaner()
        self.max_wav_value = model_spec.max_wav_value()
        self.sampling_rate = model_spec.sampling_rate()
        self.load_mel_from_disk = model_spec.load_mel_from_disk()

        if fixed_seed:
            random.seed(model_spec.get_seed())

        if shuffle:
            # TOD
            random.shuffle(self.audiopaths_and_text)

        if fixed_seed:
            self.is_trace_time = False

    @abstractmethod
    def file_to_mel(self, filename: str):
        """
        Should return mel and other data
        :param filename:
        :return:
        """
        pass

    @abstractmethod
    def numpy_to_mel(self, filename: str):
        """
        :param filename:
        :return:
        """
        pass

    def text_to_tensor(self, input_seq):
        """
        One hot encoder for text seq
        :param input_seq:
        :return:
        """
        text_norm = torch.IntTensor(text_to_sequence(input_seq, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        """
        Return tensor from t the index.
        :param index:
        :return:
        """
        if self.is_a_tensor:
            text, mel, = self._data[index]
            return text, mel
        if self.is_a_raw:
            if 'meta' not in self._data[index]:
                raise Exception("data must contain meta key")
            if 'path' not in self._data[index]:
                raise Exception("data must contain path key")
            text = self.text_to_tensor(self._data[index]['meta'])
            mel, spectral_flatness = self.file_to_mel(self._data[index]['path'])
            return text, mel, spectral_flatness

        return None, None

    def __len__(self):
        """
        :return:
        """
        return len(self._data)
