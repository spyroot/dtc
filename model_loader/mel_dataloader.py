import time
import torch

from torch.utils.data import DataLoader, DistributedSampler
from model_loader.mel_dataset_loader import TextMelLoader, TextMelCollate
from model_trainer.model_trainer_specs import ExperimentSpecs
from tacotron2.utils import fmtl_print, to_gpu


class Mel_Dataloader:
    """
    torch.Size([64, 164])
    torch.Size([64])
    torch.Size([64, 80, 855])
    torch.Size([64, 855])
    torch.Size([64])
    """
    def __init__(self, experiment_specs: ExperimentSpecs, verbose=False):
        """

        :param experiment_specs:
        """
        self.model_trainer_spec = experiment_specs
        self.model_spec = experiment_specs.get_model_spec()
        self.encoder_spec = self.model_spec.get_encoder()
        self.verbose = verbose

        self.train_dataset = None
        self.validation_dataset = None
        self.collate_fn = None
        self.train_dataloader = None

    def get_loader(self):
        """

        Returns:

        """
        if self.train_dataloader is None:
            self.create()

        return self.train_dataloader, self.validation_dataset, self.collate_fn

    def create(self):
        """

        :return:
        """
        training_set, validation_set, test_set = self.model_trainer_spec.get_audio_ds_files()
        #
        self.train_dataset = TextMelLoader(self.encoder_spec, list(training_set.values()))
        self.validation_dataset = TextMelLoader(self.encoder_spec, list(validation_set.values()))
        self.collate_fn = TextMelCollate(self.encoder_spec.frames_per_step())

        if self.model_trainer_spec.is_distributed_run():
            train_sampler = DistributedSampler(self.train_dataset)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        if self.verbose:
            fmtl_print("Dataloader train set contains", len(self.train_dataset))
            fmtl_print("Dataloader Validation set contains", len(self.validation_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, num_workers=1, shuffle=shuffle,
                                           sampler=train_sampler,
                                           batch_size=self.model_trainer_spec.batch_size,
                                           pin_memory=False,
                                           drop_last=True, collate_fn=self.collate_fn)

    def to_gpu(x):
        """

        Returns:

        """
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
        return torch.autograd.Variable(x)

    def get_batch(self, batch):
        """

        :param batch:
        :return:
        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (text_padded, input_lengths, mel_padded, max_len, output_lengths), (mel_padded, gate_padded)

    def read_batch(self):
        """

        Returns:
        """
        if self.train_dataloader is None:
            self.create()

        for i, batch in enumerate(self.train_dataloader):
            x, y = self.get_batch(batch)
            for j in range(len(batch)):
                print(batch[j].shape)
                print(batch[j].device)
            break

    def benchmark_read(self):
        """

        Returns:

        """
        if self.train_dataloader is None:
            self.create()

        # enable trace for collate dataloader
        self.collate_fn.trace()

        total_batches = 0
        t = time.process_time()
        # do full pass over train batch and count time
        for i, batch in enumerate(self.train_dataloader):
            fmtl_print("Reading batch {} out of {}".format(i, len(self.train_dataloader)), "")
            total_batches += i

        elapsed_time = time.process_time() - t
        fmtl_print("Total dataloader read time", elapsed_time)
