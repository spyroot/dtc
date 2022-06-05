import warnings
from loguru import logger


class TacotronSpecError(Exception):
    """Base class for other exceptions"""
    pass


class SpectrogramLayerSpec:
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

    def is_reverse_decoder(self) -> bool:
        """
        Return true if spec contains reverse decoder enabled
        by default it false.
        :return:
        """
        if 'reverse_decoder' in self._model_dict:
            return bool(self._model_dict['reverse_decoder'])

        return False

    def is_vae_enabled(self) -> bool:
        """
        :return:
        """
        if 'enable_vae' in self._model_dict:
            return bool(self._model_dict['enable_vae'])

        return False

    def is_stft_loss_enabled(self) -> bool:
        """
        Return true if model need compute loss based on stft.
        It also dictates what actually uploaded to GPU.
        :return:
        """
        if 'enable_stft_loss' in self._model_dict:
            return bool(self._model_dict['enable_stft_loss'])

        return False

    def encoder_spec(self):
        """
        :return:
        """
        return ['encoder']

    def decoder(self):
        return ['decoder']

    def attention(self):
        if 'attention' not in self._model_dict:
            return ['attention']

    def attention(self):
        return ['attention']

    def attention_location(self):
        return ['attention_location']

    def post_net(self):
        return ['post_net']

    def filter_length(self):
        if 'filter_length' not in self._model_dict:
            raise TacotronSpecError("Model has no filter_length defined.")
        return self._model_dict['filter_length']

    def frames_per_step(self):
        if 'frames_per_step' not in self._model_dict:
            warnings.warn("frame per step not defined in model spec. return default 1")
            return 1
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
        """
        :return:
        """
        if 'max_wav_value' not in self._model_dict:
            raise TacotronSpecError("Model has no max_wav_value defined.")

        return self._model_dict['max_wav_value']

    def symbols_embedding_dim(self) -> int:
        """
        :return:
        """
        if 'symbols_embedding_dim' in self._model_dict:
            return self._model_dict['symbols_embedding_dim']

        return 512

    def get_encoder(self):
        """
        return encoder spec
        :return:
        """
        if 'encoder' not in self._model_dict:
            raise TacotronSpecError("Model has no encoder specification,"
                                    " Please check configuration.")
        return self._model_dict['encoder']

    def p_attention_dropout(self):
        """
        return encoder spec
        :return:
        """
        decoder = self.get_decoder()
        if 'attention_dropout' in decoder:
            return decoder['attention_dropout']
        return 0.1

    def p_decoder_dropout(self):
        """
        return encoder spec
        :return:
        """
        decoder = self.get_decoder()
        if 'decoder_dropout' in decoder:
            return decoder['decoder_dropout']
        return 0.1

    def decoder_fps(self) -> int:
        """
        return decoder frame per step.
        :return:
        """
        decoder = self.get_decoder()
        if 'fps' in decoder:
            return int(decoder['fps'])
        return 1

    def max_decoder_steps(self) -> int:
        """
        return encoder spec
        :return:
        """
        decoder = self.get_decoder()
        if 'max_decoder_steps' in decoder:
            return int(decoder['max_decoder_steps'])
        return 1000

    def gate_threshold(self) -> float:
        """
        return encoder spec
        :return:
        """
        decoder = self.get_decoder()
        if 'gate_threshold' in decoder:
            return float(decoder['gate_threshold'])
        return 0.5

    def decoder_rnn_dim(self) -> int:
        """
        :return: Return decoder RNN dimension
        """
        decoder = self.get_decoder()
        if 'rnn_dim' in decoder:
            return int(decoder['rnn_dim'])
        return 1024

    def pre_net_dim(self) -> int:
        """
        return encoder spec
        :return:
        """
        decoder = self.get_decoder()
        if 'pre_net_dim' in decoder:
            return int(decoder['pre_net_dim'])
        return 256

    def get_attention(self):
        """
        return encoder spec
        :return:
        """
        if 'attention' not in self._model_dict:
            raise TacotronSpecError("Model has no attention specification,"
                                    " Please check configuration.")
        return self._model_dict['attention']

    def get_attention_location(self):
        """
        Return attention location specs.
        :return:
        """
        if 'attention_location' not in self._model_dict:
            raise TacotronSpecError("Model has no attention_location specification,"
                                    " Please check configuration.")
        return self._model_dict['attention_location']

    def attention_location_n_filters(self) -> int:
        """
        :return: Returns attention location number of filters.
        """
        atten_loc = self.get_attention_location()
        if 'num_filters' in atten_loc:
            return atten_loc['num_filters']
        return 32

    def attention_location_kernel_size(self) -> int:
        """
        :return: Returns attention location kernel size.
        """
        atten_loc = self.get_attention_location()
        if 'kernel_size' in atten_loc:
            return atten_loc['kernel_size']
        return 31

    def attention_rnn_dim(self) -> int:
        """
        :return:
        """
        attention = self.get_attention()
        if 'rnn_dim' in attention:
            return attention['rnn_dim']
        return 1024

    def attention_dim(self) -> int:
        """
        :return:
        """
        attention = self.get_attention()
        if 'attention_dim' in attention:
            return attention['attention_dim']
        return 128

    def get_decoder(self):
        """
        return encoder spec
        :return:
        """
        if 'decoder' not in self._model_dict:
            raise TacotronSpecError("Model has no decoder specification,"
                                    " Please check configuration.")
        return self._model_dict['decoder']

    def encoder_kernel_size(self) -> int:
        """
        :return:
        """
        encoder = self.get_encoder()
        if 'kernel_size' in encoder:
            return encoder['kernel_size']
        return 5

    def encoder_n_convolutions(self) -> int:
        """
        Encoders number of conv layers.
        :return:
        """
        encoder = self.get_encoder()
        if 'num_convolutions' in encoder:
            return encoder['num_convolutions']
        return 3

    def dropout_rate(self) -> float:
        """
        Encoder dropout rate.
        :return:
        """
        encoder = self.get_encoder()
        if 'dropout_rate' in encoder:
            return encoder['dropout_rate']

        return 0.5

    def encoder_embedding_dim(self) -> int:
        """
        Encoder embedding dimension.
        :return:
        """
        encoder = self.get_encoder()
        if 'embedding_dim' in encoder:
            return encoder['embedding_dim']
        return 512

    def get_text_cleaner(self):
        """
        :return:
        """
        if 'sampling_rate' not in self._model_dict:
            raise TacotronSpecError("Model has no sampling_rate defined.")
        return ['english_cleaners']

    def get_seed(self):
        """
        Return fixed seed.
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
