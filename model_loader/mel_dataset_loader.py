import random
import time
import numpy as np
import torch
import torch.utils.data

from timeit import default_timer as timer
from datetime import timedelta
from loguru import logger
from model_loader.tacotron_stft import TacotronSTFT
from model_trainer.specs.tacatron_spec import TacotronSpec
from tacotron2.utils import load_wav_to_torch
from text import text_to_sequence
import librosa

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, model_spec: TacotronSpec, data, data_format, is_trace_time=False):
        """
        """

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

        # if raw we need transcode to stft's
        if self.is_a_raw:
            logger.debug("Creating TacotronSTFT for raw file processing.")
            self.stft = TacotronSTFT(
                model_spec.filter_length(), model_spec.hop_length(), model_spec.win_length(),
                model_spec.n_mel_channels(), model_spec.sampling_rate(), model_spec.mel_fmin(),
                model_spec.mel_fmax())
        #
        random.seed(model_spec.get_seed())
        #
        # random.shuffle(self.audiopaths_and_text)

        self.is_trace_time = False

    # def get_mel_text_pair(self, audiopath_and_text):
    #     """
    #
    #     :param audiopath_and_text:
    #     :return:
    #     """
    #     # separate filename and text
    #     audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
    #     text = self.text_to_tensor(text)
    #     mel = self.file_to_mel(audiopath)
    #     return text, mel

    def file_to_mel(self, filename):
        """

        :param filename:
        :return:
        """

       # logger.debug("Converting file {} to mel", filename)
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            mel_spec = self.stft.mel_spectrogram(audio_norm)
            print("mel original shape", mel_spec.shape)
            mel_numpy = mel_spec.numpy()
            mel_spec = torch.squeeze(mel_spec, 0)

            S, phase = librosa.magphase(librosa.stft(mel_numpy))
            print(S.shape)
            print(phase.shape)

        else:
            mel_spec = torch.from_numpy(np.load(filename))
            assert mel_spec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    mel_spec.size(0), self.stft.n_mel_channels))

        return mel_spec

    def numpy_to_mel(self, filename):
        """

        Args:
            filename:

        Returns:

        """
        mel_spec = torch.from_numpy(np.load(filename))
        assert mel_spec.size(0) == self.stft.n_mel_channels, (
            'Mel dimension mismatch: given {}, expected {}'.format(
                mel_spec.size(0), self.stft.n_mel_channels))

        return mel_spec

    def text_to_tensor(self, text):
        """
        One hot encoder for text seq

        :param text:
        :return:
        """
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        if self.is_a_tensor:
            text, mel = self._data[index]
            return text, mel
        if self.is_a_raw:
            if 'meta' not in self._data[index]:
                raise Exception("data must contain meta key")
            if 'path' not in self._data[index]:
                raise Exception("data must contain path key")
            text = self.text_to_tensor(self._data[index]['meta'])
            mel = self.file_to_mel(self._data[index]['path'])
            return text, mel

        return None, None

    def __len__(self):
        return len(self._data)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    """

    def __init__(self, n_frames_per_step=1, is_trace_time=False):
        """

        Args:
            n_frames_per_step:
        """
        self.n_frames_per_step = n_frames_per_step
        self.is_trace_time = False

    def trace(self):
        """

        Returns:

        """
        self.is_trace_time = True

    def __call__(self, batch):
        """
        Automatically collating individual fetched data samples into.
        Each batch containers [text_normalized, mel_normalized]

        Args:
            batch:  batch size.
        Returns:

        """
        # Right zero-pad all one-hot text sequences to max input length
        if self.is_trace_time:
            t = time.process_time()
            start = timer()

        input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(
            [len(x[0]) for x in batch]), dim=0, descending=True)

        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        if self.is_trace_time:
            elapsed_time = time.process_time() - t
            end = timer()
            logger.info("Collate single pass time {}".format(elapsed_time))
            logger.info("Collate single pass delta sec {}".format(timedelta(seconds=end - start)))

        print("mel padded {}", mel_padded.shape)
        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
