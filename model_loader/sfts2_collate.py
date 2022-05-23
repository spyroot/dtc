import time
from datetime import timedelta
from timeit import default_timer as timer
import torch
import torch.utils.data
from loguru import logger


class TextMelCollate:
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
        self.txt_id = 1
        self.mel = 2

    def trace(self):
        """
        If trace enabled. (debug)
        :return:
        """
        self.is_trace_time = True

    def __call__(self, batch):
        """
        Dataloader will call this method.
        :param batch:
        :return:
        """
        assert self.mel == 2

        # Right zero-pad all one-hot text sequences to max input length
        if self.is_trace_time:
            t = time.process_time()
            start = timer()

        lstm_input_len = len(batch)
        # sort text
        #####
        ###
        ##
        input_lengths, sorted_idx = \
            torch.sort(torch.LongTensor([len(x[0]) for x in batch]),
                       dim=self.sort_dim, descending=self.descending)

        # update max len
        max_input_len = input_lengths[0]
        if max_input_len > max_input_len:
            self.largest_seq = max_input_len

        text_padded = torch.LongTensor(lstm_input_len, max_input_len).zero_()
        for i in range(len(sorted_idx)):
            text = batch[sorted_idx[i]][0]
            text_padded[i, :text.size(0)] = text

        # zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[self.txt_id].size(1) for x in batch])

        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        mel_padded = torch.FloatTensor(lstm_input_len, num_mels, max_target_len).zero_()
        gate_padded = torch.FloatTensor(lstm_input_len, max_target_len).zero_()
        output_lengths = torch.LongTensor(lstm_input_len)

        # sort batches
        outputs = []
        for i in range(len(sorted_idx)):
            # idx text , idx 1 mel
            mel = batch[sorted_idx[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        if self.is_trace_time:
            elapsed_time = time.process_time() - t
            end = timer()
            logger.info("Collate single pass time {}".format(elapsed_time))
            logger.info("Collate single pass delta sec {}".format(timedelta(seconds=end - start)))

        if self.device is not None:
            return text_padded.to(self.device), input_lengths.to(self.device), mel_padded.to(self.device), \
                   gate_padded.to(self.device), output_lengths.to(self.device)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
