# SFTS, Mel dataset
#
# It for dataset that outputs only one hot vector and mel spectrogram.
#
# Mustafa
#
import time
import warnings
from logging import warn
from typing import Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler

from model_loader.base_sfts_dataset import DatasetError
from model_loader.dataset_stft25 import SFTF2Dataset
from model_loader.dataset_stft30 import SFTF3Dataset
from model_loader.sfts2_collate import TextMelCollate2
from model_loader.sfts3_collate import TextMelCollate3
from model_trainer.specs.dtc_spec import ModelSpecDTC
from model_trainer.trainer_specs import ExperimentSpecs
from model_trainer.utils import fmtl_print, to_gpu


class SFTFDataloader:
    """
    torch.Size([64, 164])
    torch.Size([64])
    torch.Size([64, 80, 855])
    torch.Size([64, 855])
    torch.Size([64])
    """
    # TODO add option to pass dataset object directly
    # TODO add option indicate split ratio
    def __init__(self, experiment_specs: ExperimentSpecs, ping_memory=True,
                 rank=0, world_size=0, num_worker=1, verbose=False):
        """

        :param experiment_specs:
        :param ping_memory:
        :param rank:
        :param world_size:
        :param num_worker:
        :param verbose:
        """
        self.reduce_size = None
        self.set_logger(verbose)
        self._ds2_mandatory = ['path', 'meta']

        self.rank = rank
        self.world_size = world_size
        self.val_loader = None
        self.val_sampler = None
        self.trainer_spec: ExperimentSpecs = experiment_specs
        self.model_spec: ModelSpecDTC = experiment_specs.get_model_spec()
        self.mel_model_spec = self.model_spec.get_encoder()

        self.verbose = verbose
        self.ping_memory = ping_memory
        self.num_worker = num_worker

        self.train_dataset = None
        self.collate_fn = None
        self.train_dataloader = None
        self.validation_dataset = None
        self.batch_size = None

    def update_batch(self, new_batch: int):
        """
        Method update batch size

        :return:
        """
        if self.train_dataloader is None:
            print("Updating batch ", new_batch)
            self.batch_size = new_batch
            self.train_dataloader = None
            self.val_loader = None
            self.create(update=True)

        return self.train_dataloader, self.val_loader, self.collate_fn

    def get_loader(self):
        """
        Method return data loader.

        :return:
        """
        if self.train_dataloader is None:
            self.create()

        return self.train_dataloader, self.val_loader, self.collate_fn

    def _validate_v2(self, raw_dataset, strict: Optional[bool] = True):
        """

        :param raw_dataset:
        :return:
        """
        # validate, dataset
        for i in range(0, len(raw_dataset)):
            for m in self._ds2_mandatory:
                # print("comparing", self._ds2_mandatory[m])
                # print("comparing", raw_dataset[i])

                if self._ds2_mandatory[m] not in raw_dataset[i]:
                    if strict:
                        raise DatasetError("Each entry in dataset must contain key path.")
                    else:
                        warnings.warn("Each entry in dataset must contain key path.")

    def create_v2raw(self, strict=True):
        """
        Create dataset for v25 model
        :param strict:
        :return:
        """
        dataset_files = self.trainer_spec.get_audio_dataset()
        if len(dataset_files) == 0:
            raise ValueError("Empty dataset.")

        train_set = list(dataset_files['train_set'].values())
        validation_set = list(dataset_files['validation_set'].values())
        
        # drop if needed
        #if self.reduce_size:
            
        # validate    
        self._validate_v2(train_set, strict=strict)
        self._validate_v2(validation_set, strict=strict)
        logger.debug("Validated dataset")

        # create
        self.train_dataset = SFTF2Dataset(model_spec=self.mel_model_spec, data=train_set, data_format='audio_raw',
                                          overfit=self.trainer_spec.is_overfit())
        self.validation_dataset = SFTF2Dataset(self.mel_model_spec, validation_set, data_format='audio_raw',
                                               overfit=self.trainer_spec.is_overfit())
        self.collate_fn = TextMelCollate2(nfps=self.mel_model_spec.frames_per_step(), device=None)

    def createv2_from_tensor(self):
        """
        :return:
        """
        dataset_files = self.trainer_spec.get_audio_dataset()

        train_set = dataset_files['train_set'],
        validation_set = dataset_files['validation_set']

        # print(len(train_set))
        # print("Data Keys", train_set[0].keys())
        # print("Data Keys", train_set[1].keys())
        # sys.exit(1)
        # Data Keys dict_keys(['filter_length', 'hop_length', 'win_length', 'n_mel_channels', 'sampling_rate', 'mel_fmin', 'mel_fmax', 'data'])
        if dataset_files['ds_type'] == 'tensor_mel':
            self.train_dataset = SFTF2Dataset(model_spec=self.mel_model_spec, data=train_set, data_format='tensor_mel',
                                              overfit=self.trainer_spec.is_overfit())
            self.validation_dataset = SFTF2Dataset(self.mel_model_spec, validation_set,
                                                   data_format='tensor_mel',
                                                   overfit=self.trainer_spec.is_overfit())
            self.collate_fn = TextMelCollate2(nfps=self.mel_model_spec.frames_per_step(), device=None)

    def create_v3raw(self):
        """

        :return:
        """
        pk_dataset = self.train_dataset.get_audio_dataset()
        self.train_dataset = SFTF3Dataset(self.mel_model_spec,
                                          list(pk_dataset['train_set'].values()),
                                          data_format='audio_raw',
                                          overfit=self.trainer_spec.is_overfit())
        self.validation_dataset = SFTF3Dataset(self.mel_model_spec,
                                               list(pk_dataset['validation_set'].values()),
                                               data_format='audio_raw',
                                               overfit=self.trainer_spec.is_overfit())
        self.collate_fn = TextMelCollate3(nfps=self.mel_model_spec.frames_per_step(), device=None)

    def create_v3dataset(self, device):
        """

        :return:
        """
        pk_dataset = self.train_dataset.get_audio_dataset()
        if pk_dataset['ds_type'] == 'tensor_mel':
            self.train_dataset = SFTF3Dataset(self.mel_model_spec,
                                              pk_dataset['train_set'],
                                              data_format='tensor_mel',
                                              overfit=self.trainer_spec.is_overfit())
            self.validation_dataset = SFTF3Dataset(self.mel_model_spec,
                                                   pk_dataset['validation_set'],
                                                   data_format='tensor_mel',
                                                   overfit=self.trainer_spec.is_overfit())
            self.collate_fn = TextMelCollate3(nfps=self.mel_model_spec.frames_per_step(), device=None)

    def create(self, update: Optional[bool] = False):
        """
         :return:
        """
        # training_set, validation_set, test_set = self.model_trainer_spec.get_audio_ds_files()

        # print(self.trainer_spec)
        # experiment_specs = ExperimentSpecs(verbose=False)
        pk_dataset = self.trainer_spec.get_audio_dataset()
        mandatory_keys = ['train_set', 'validation_set', 'test_set', 'ds_type']
        if mandatory_keys != list(pk_dataset.keys()):
            raise ValueError("Dataset spec has no mandatory key.")

        if not update:
            if pk_dataset['ds_type'] == 'tensor_mel':
                print("data is tensor_mel")
                self.createv2_from_tensor()
            if pk_dataset['ds_type'] == 'audio_raw':
                self.create_v2raw()

        # test_set
        if self.trainer_spec.is_distributed_run():
            logger.info("Creating distribute sampler rank {} , world size {}", self.rank, self.world_size)
            train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size)
            val_sampler = DistributedSampler(self.validation_dataset, num_replicas=self.world_size)
            is_shuffle = False
        else:
            # we shuffle only if on single run otherwise it false.
            train_sampler = None
            val_sampler = None
            is_shuffle = True

        if self.verbose:
            logger.info("Dataloader train set contains {} entries.".format(len(self.train_dataset)))
            logger.info("Dataloader validation set contains {} entries.".format(len(self.validation_dataset)))

        if len(self.train_dataset) == 0:
            raise ValueError("Dataloader received empty train dataset.")
        if len(self.validation_dataset) == 0:
            raise ValueError("Dataloader received empty validation dataset.")
        if self.trainer_spec.batch_size == 0:
            raise ValueError("Dataloader need batch size > 0.")

        # we update only if need
        if self.batch_size is None:
            self.batch_size = self.trainer_spec.batch_size

        if self.trainer_spec.is_distributed_run():
            self.batch_size = int(self.trainer_spec.batch_size / float(self.world_size))

        if self.trainer_spec.is_overfit():
            warnings.warn("You are running in overfitting settings.")

        self.train_dataloader = DataLoader(self.train_dataset,
                                           num_workers=self.num_worker,
                                           shuffle=is_shuffle,
                                           sampler=train_sampler,
                                           batch_size=self.batch_size,
                                           pin_memory=True,
                                           drop_last=True,
                                           collate_fn=self.collate_fn)

        self.val_loader = DataLoader(self.validation_dataset,
                                     sampler=val_sampler,
                                     num_workers=self.num_worker,
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

    @staticmethod
    def get_batch(batch):
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
