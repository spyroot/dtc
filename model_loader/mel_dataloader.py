import time
import torch

from torch.utils.data import DataLoader, DistributedSampler
from model_loader.mel_dataset_loader import TextMelLoader, TextMelCollate
from model_trainer.specs import model_spec
from model_trainer.specs.dtc_spec import DTC
from model_trainer.trainer_specs import ExperimentSpecs
from tacotron2.utils import fmtl_print, to_gpu
from loguru import logger


class Mel_Dataloader:
    """
    torch.Size([64, 164])
    torch.Size([64])
    torch.Size([64, 80, 855])
    torch.Size([64, 855])
    torch.Size([64])
    """

    def __init__(self, experiment_specs: ExperimentSpecs, rank=0, world_size=1, verbose=False):
        """

        :param experiment_specs:
        """
        self.rank = rank
        self.world_size = world_size
        self.val_loader = None
        self.val_sampler = None
        self.trainer_spec: ExperimentSpecs = experiment_specs
        self.model_spec: DTC = experiment_specs.get_model_spec()
        self.encoder_spec = self.model_spec.get_encoder()
        self.verbose = verbose

        self.train_dataset = None
        self.validation_dataset = None
        self.collate_fn = None
        self.train_dataloader = None

    def get_loader(self):
        """
        Method return data loader
        :return:
        """
        if self.train_dataloader is None:
            self.create()

        return self.train_dataloader, self.val_loader, self.collate_fn

    def create(self):
        """

        :return:
        """
        # training_set, validation_set, test_set = self.model_trainer_spec.get_audio_ds_files()

        experiment_specs = ExperimentSpecs(verbose=False)
        pk_dataset = experiment_specs.get_audio_dataset()

        if pk_dataset['ds_type'] == 'tensor_mel':
            self.train_dataset = TextMelLoader(self.encoder_spec,
                                               pk_dataset['train_set'], data_format='tensor_mel')
            self.validation_dataset = TextMelLoader(self.encoder_spec,
                                                    pk_dataset['validation_set'], data_format='tensor_mel')
            self.collate_fn = TextMelCollate(self.encoder_spec.frames_per_step())

        if pk_dataset['ds_type'] == 'audio_raw':
            self.train_dataset = TextMelLoader(self.encoder_spec,
                                               list(pk_dataset['train_set'].values()),
                                               data_format='audio_raw')
            self.validation_dataset = TextMelLoader(self.encoder_spec,
                                                    list(pk_dataset['validation_set'].values()),
                                                    data_format='audio_raw')
            self.collate_fn = TextMelCollate(self.encoder_spec.frames_per_step())
        # test_set
        if self.trainer_spec.is_distributed_run():
            logger.info("Create distribute sampler rank {} , world size {}", self.rank, self.world_size)
            train_sampler = DistributedSampler(self.train_dataset,
                                               num_replicas=self.world_size)
            is_shuffle = False
        else:
            # we shuffle only if on single run otherwise it false.
            train_sampler = None
            is_shuffle = True

        if self.verbose:
            fmtl_print("Dataloader train set contains", len(self.train_dataset))
            fmtl_print("Dataloader validation set contains", len(self.validation_dataset))

        if len(self.train_dataset) == 0:
            raise Exception("Dataloader received empty train dataset.")
        if len(self.validation_dataset) == 0:
            raise Exception("Dataloader received empty validation dataset.")
        if self.trainer_spec.batch_size == 0:
            raise Exception("Dataloader need batch size > 0.")

        if self.trainer_spec.is_distributed_run():
            self.batch_size = int(self.trainer_spec.batch_size/ float(self.world_size))
        else:
            self.batch_size = self.trainer_spec.batch_size

        self.train_dataloader = DataLoader(self.train_dataset,
                                           num_workers=1,
                                           shuffle=is_shuffle,
                                           sampler=train_sampler,
                                           batch_size=self.trainer_spec.batch_size,
                                           pin_memory=True,
                                           drop_last=True, collate_fn=self.collate_fn)

        self.val_sampler = None
        if self.trainer_spec.is_distributed_run():
            self.val_sampler = DistributedSampler(self.validation_dataset,
                                                  num_replicas=self.world_size,
                                                )

        self.val_loader = DataLoader(self.validation_dataset,
                                     sampler=self.val_sampler,
                                     num_workers=1,
                                     pin_memory=True,
                                     shuffle=is_shuffle,
                                     batch_size=self.batch_size,
                                     collate_fn=self.collate_fn)

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
        Read a b batch testing
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
