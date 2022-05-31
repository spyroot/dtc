# Collate v3 used by data loader.
#
# It prepares each batch,  the output of collate
#
# pad and encoded text.
#  - input_lengths,
#  - padded mel,
#  - gates padded.
#  - output_lengths
#  - spectral_data.  The data size fixed in block , it assumed this block bypass LSTM layer
#                    hence, we don't need any padding but sorted order must remain the same.
#
#  Shape:
#  text shape: torch.Size([104])
#  mel shape: torch.Size([80, 624])
#  spectral: torch.Size([1, 1024])
# Mustafa

import sys
import time
from datetime import timedelta
from timeit import default_timer as timer

import torch
import torch.utils.data
from loguru import logger


class TextMelCollate3:
    """

    """
    def __init__(self, device, nfps=1, sort_dim=0, descending=True, is_trace_time=False):
        """

        Extract frame per step nfps
        y, sr = librosa.load(librosa.ex('choice'))
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_samples = librosa.frames_to_samples(beats)

        :param device: if we want to send batch to gpu
        :param is_trace_time: if we want trace per batch read timer.  mainly for benchmark io.
        :param nfps: see librosa comment
        :param device:
        :param sort_dim:  what dim to use to sort tensors
        """
        self.n_frames_per_step = nfps
        self.is_trace_time = False
        self.largest_seq = 0
        self.sort_dim = sort_dim
        self.descending = descending
        self.device = None

        # all indexes of data.
        self.txt_id = 1
        self.mel = 2
        self.spectral_data = 3

    def __call__(self, batch):
        """
        Collating individual fetched data samples into batch

        :param batch:
        :return:
        """
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

        txt_id = 1
        lstm_input_len = len(batch)
        # sort text
        #####
        ###
        ##
        input_lengths, ids_sorted_decreasing = \
            torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)

        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(lstm_input_len, max_input_len).zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[txt_id].size(1) for x in batch])

        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        mel_padded = torch.FloatTensor(lstm_input_len, num_mels, max_target_len).zero_()
        gate_padded = torch.FloatTensor(lstm_input_len, max_target_len).zero_()
        output_lengths = torch.LongTensor(lstm_input_len)

        # spectral data fixed size.
        spectral_max = max([x[2].size(1) for x in batch])
        spectral_data = torch.FloatTensor(lstm_input_len,  1, spectral_max)
        for i in range(len(ids_sorted_decreasing)):
            # idx text , idx 1 mel
            mel = batch[ids_sorted_decreasing[i]][1]
            spectral = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)
            spectral_data[i] = spectral

        if self.is_trace_time:
            elapsed_time = time.process_time() - t
            end = timer()
            logger.info("Collate single pass time {}".format(elapsed_time))
            logger.info("Collate single pass delta sec {}".format(timedelta(seconds=end - start)))

        # print("mel padded", mel_padded.shape)
        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spectral_data
