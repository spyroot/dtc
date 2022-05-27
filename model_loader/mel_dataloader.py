# SFTS, Mel dataset
#
# It for dataset that outputs only one hot vector and mel spectrogram.
#
# Mustafa
#
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Callable

import torch
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler

from model_loader.base_sfts_dataset import DatasetError, BaseSFTFDataset
from model_loader.dataset_stft25 import SFTF2Dataset
from model_loader.dataset_stft30 import SFTF3Dataset
from model_loader.sfts2_collate import TextMelCollate2
from model_trainer.specs.dtc_spec import ModelSpecDTC
from model_trainer.trainer_specs import ExperimentSpecs
from model_trainer.utils import to_gpu


class SFTFDataloader:
    """
    A dataset loader provider interface that aggregate different type of SFTs'
    dataset representations, under a same object.

    There many way data loader can be created.
    if you passed optional dataset=BaseSFTFDataset, by default it split to train and validation.
    """

    def __init__(self,
                 experiment_specs: ExperimentSpecs,
                 dataset: Optional[BaseSFTFDataset] = None,
                 pin_memory: Optional[bool] = True,
                 rank: Optional[int] = 0,
                 world_size: Optional[int] = 0,
                 num_worker: Optional[int] = 1,
                 reduction_ration: Optional[int] = 0,
                 batch_size: Optional[int] = 0,
                 verbose: Optional[bool] = False):
        """
        :param experiment_specs:
        :param pin_memory: if we need pin to memory.
        :param rank: a rank of node instantiating data loader.
        :param world_size: a world_side ( check torch ddp doc)
        :param num_worker: a number of worker used by internal data loaders.
        :param verbose: enable debug.
        :param:reduction_ration if we need reduce a size of dataset.
        """
        self._reduce_ration = int(reduction_ration)
        self.set_logger(verbose)
        self._ds2_mandatory = ['path', 'meta']
        self._path_key = 'path'

        self._rank = rank
        self._world_size = world_size
        self._trainer_spec: ExperimentSpecs = experiment_specs
        self._model_spec: ModelSpecDTC = experiment_specs.get_model_spec()
        self._encoder_spec = self._model_spec.get_encoder()

        self.verbose = verbose
        self._ping_memory = pin_memory
        self._num_worker = num_worker

        self.collate_fn = None
        # by default, we take from trainer spec.
        if batch_size > 0:
            self._batch_size = batch_size
        else:
            if self._trainer_spec.batch_size() > 0:
                self._batch_size = self._trainer_spec.batch_size()
            else:
                raise ValueError("Batch size less than zero.")

        # datasets and dataloaders
        self._datasets = {}
        self._dataloaders = {}
        if dataset is not None:
            self.from_dataset(ds=dataset)

    def _initialized_before(self) -> bool:
        """
        Dataloaders and dataset should be created if not method return false.
        :return:
        """
        return self._dataloaders is not None and len(self._dataloaders) > 0 \
               and self._datasets is not None and len(self._datasets) > 0

    def update_batch(self, new_batch: int) -> tuple[list[DataLoader], Callable]:
        """
        Method update batch size, i.e re-create data loader but
        it doesn't re-invalidate dataset.

        :return:
        """
        # main point update batch size without entire dataset.
        if not self._initialized_before():
            logger.debug("Updating dataset.")
            print("Updating batch ", new_batch)
            self._batch_size = new_batch
            del self._dataloaders
            self._dataloaders = {}
            self._create(update=True)

        return list(self._dataloaders.values()), self.collate_fn

    def get_all(self) -> tuple[dict[str, DataLoader], Callable]:
        """
        Return all data loaders packed as dict,
        key : dataloader(Dataset)
        :return:
        """
        if not self._initialized_before():
            self._create()

        return self._dataloaders, self.collate_fn

    def get_loader(self) -> tuple[list[DataLoader], Callable]:
        """
        Method return all data loaders and collate callback.
        it will call factor method and create datasets
        and dataloaders and sampler for DDP case.
        :return:
        """
        if not self._initialized_before():
            self._create()

        return list(self._dataloaders.values()), self.collate_fn

    def _validate_v2(self, raw_dataset, strict: Optional[bool] = True):
        """
        Validate that raw audio dataset contains minimum set of data.
         - text and path to a file.

        :param raw_dataset: a dict with path and text
        :param strict: will raise the exception.
        :return: True data is valid.
        """
        # validate, dataset
        for i in range(0, len(raw_dataset)):
            for m in self._ds2_mandatory:
                if m not in raw_dataset[i]:
                    if strict:
                        raise DatasetError(f"Each entry in dataset dict must contain a key {m}.")
                    else:
                        warnings.warn(f"Each entry in dataset dict must contain a {m}.")
                else:
                    if m == self._path_key:
                        path_to_file = raw_dataset[i][m]
                        if not Path(path_to_file).exists():
                            if strict:
                                raise DatasetError(f"Can't find a file {raw_dataset[i][m]}")
                            else:
                                warnings.warn(f"Can't find a file {raw_dataset[i][m]}")

    def _create_v2raw(self, version=2, strict=True):
        """
        Create dataset for v25 model
        :param version:
        :param strict:
        :return:
        """
        dataset_files = self._trainer_spec.get_audio_dataset()
        ds_keys = self._trainer_spec.get_audio_dataset_keys()
        data_key = self._trainer_spec.get_dataset_data_key()

        assert len(ds_keys) > 0
        assert len(dataset_files) > 0

        for _, k in enumerate(ds_keys):
            # in case, we have only subset,  (only train set)
            if k not in dataset_files:
                continue

            ds = list(dataset_files[k].values())
            logger.debug(f"Dataset contains {len(dataset_files[k])} records")
            if not isinstance(ds, list):
                raise ValueError("Dataset spec for raw audio file, must be a list.")
            if self._reduce_ration > 0:
                last_index = max(1, round(len(ds) * (self._reduce_ration / 100)))
                ds = ds[: last_index]

            if version == 2:
                self._datasets[k] = SFTF2Dataset(model_spec=self._encoder_spec,
                                                 data=ds, data_format='audio_raw',
                                                 overfit=self._trainer_spec.is_overfit())
            elif version == 3:
                self._datasets[k] = SFTF3Dataset(model_spec=self._encoder_spec,
                                                 data=ds, data_format='audio_raw',
                                                 overfit=self._trainer_spec.is_overfit())
            else:
                raise ValueError("Unknown version.")

            logger.debug(f"Data loader updated {self._datasets.keys()} ver {version}.")

        if version == 2:
            self.collate_fn = TextMelCollate2(nfps=self._encoder_spec.frames_per_step(), device=None)
        elif version == 3:
            self.collate_fn = TextMelCollate2(nfps=self._encoder_spec.frames_per_step(), device=None)

    def createv2_from_tensor(self, version=2):
        """
        Method create datasets.

        The trainer spec returns a dict of all dataset files.
        in typical case it train_set, validation_set, test_set.

        get_audio_dataset_keys() all possible keys that trainer_spec
        can use, same keys used to represent internally all dataset
        and dataloaders.

        :return:
        """
        dataset_files = self._trainer_spec.get_audio_dataset()
        ds_keys = self._trainer_spec.get_audio_dataset_keys()
        data_key = self._trainer_spec.get_dataset_data_key()

        assert len(ds_keys) > 0
        assert len(dataset_files) > 0

        for _, k in enumerate(ds_keys):
            # in case, we have only subset,  (only train set)
            if k not in dataset_files:
                continue

            ds = dataset_files[k]
            logger.debug(f"Dataset contains {len(dataset_files[k])} records")
            if not isinstance(ds, dict):
                raise ValueError("Dataset spec must be a dict.")
            if self._reduce_ration > 0:
                last_index = max(1, round(len(ds[data_key]) * (self._reduce_ration / 100)))
                ds[data_key] = ds[data_key][: last_index]

            if version == 2:
                self._datasets[k] = SFTF2Dataset(model_spec=self._encoder_spec,
                                                 data=ds, data_format='tensor_mel',
                                                 overfit=self._trainer_spec.is_overfit())
            elif version == 3:
                self._datasets[k] = SFTF3Dataset(model_spec=self._encoder_spec,
                                                 data=ds, data_format='tensor_mel',
                                                 overfit=self._trainer_spec.is_overfit())
            else:
                raise ValueError("Unknown version.")

            logger.debug(f"Data loader updated {self._datasets.keys()} ver {version}.")

        if version == 2:
            self.collate_fn = TextMelCollate2(nfps=self._encoder_spec.frames_per_step(), device=None)
        elif version == 3:
            self.collate_fn = TextMelCollate2(nfps=self._encoder_spec.frames_per_step(), device=None)

    def _createv2_from_tensor(self, version=2):
        """
        Method create datasets.

        The trainer spec returns a dict of all dataset files.
        in typical case it train_set, validation_set, test_set.

        get_audio_dataset_keys() all possible keys that trainer_spec
        can use, same keys used to represent internally all dataset
        and dataloaders.

        :return:
        """
        dataset_files = self._trainer_spec.get_audio_dataset()
        ds_keys = self._trainer_spec.get_audio_dataset_keys()
        data_key = self._trainer_spec.get_dataset_data_key()

        assert len(ds_keys) > 0
        assert len(dataset_files) > 0

        for _, k in enumerate(ds_keys):
            # in case, we have only subset,  (only train set)
            if k not in dataset_files:
                continue

            ds = dataset_files[k]
            logger.debug(f"Dataset contains {len(dataset_files[k])} records")
            if not isinstance(ds, dict):
                raise ValueError("Dataset spec must be a dict.")
            if self._reduce_ration > 0:
                last_index = max(1, round(len(ds[data_key]) * (self._reduce_ration / 100)))
                ds[data_key] = ds[data_key][: last_index]

            if version == 2:
                self._datasets[k] = SFTF2Dataset(model_spec=self._encoder_spec,
                                                 data=ds, data_format='tensor_mel',
                                                 overfit=self._trainer_spec.is_overfit())
            elif version == 3:
                self._datasets[k] = SFTF3Dataset(model_spec=self._encoder_spec,
                                                 data=ds, data_format='tensor_mel',
                                                 overfit=self._trainer_spec.is_overfit())
            else:
                raise ValueError("Unknown version.")

            logger.debug(f"Data loader updated {self._datasets.keys()} ver {version}.")

        if version == 2:
            self.collate_fn = TextMelCollate2(nfps=self._encoder_spec.frames_per_step(), device=None)
        elif version == 3:
            self.collate_fn = TextMelCollate2(nfps=self._encoder_spec.frames_per_step(), device=None)

    def _create_loaders(self):
        """
        Method will create all data loaders.
        :return:
        """
        samplers = {}

        if self._trainer_spec.is_distributed_run():
            logger.info("Creating distribute sampler rank {} , world size {}", self._rank, self._world_size)
            try:
                # create k num samplers.
                for _, k in enumerate(self._datasets):
                    sampler = DistributedSampler(self._datasets[k], num_replicas=self._world_size)
                    samplers[k] = sampler
                    # samplers.append(sampler)
            except RuntimeError as e:
                logger.error(f"Dataloader in DDP mode, Make sure DDP is initialized. error {e} existing")
                print(f"Make sure DDP is initialized. error {e}. existing")
                sys.exit(1)
            is_shuffle = False
        else:
            # we shuffle only if on single run otherwise it false.
            train_sampler = None
            val_sampler = None
            is_shuffle = True

        for _, k in enumerate(self._datasets):
            # print("type ", self._datasets)
            if len(self._datasets) == 0:
                raise ValueError(f"Dataloader for key {k} is empty.")
            logger.info(f"Dataloader train set contains {len(self._datasets[k])} entries.")

        if self._batch_size == 0:
            raise ValueError("Dataloader batch size == 0.")

        # for DDP we recompute.
        if self._trainer_spec.is_distributed_run():
            self._batch_size = int(self._trainer_spec.batch_size() / float(self._world_size))

        if self._trainer_spec.is_overfit():
            warnings.warn("You are running in overfitting settings.")

        # each data loader has same key.
        for _, k in enumerate(self._datasets):
            sampler = None
            if samplers is not None and len(samplers) > 0:
                sampler = samplers[k]
            self._dataloaders[k] = DataLoader(self._datasets[k],
                                              num_workers=self._num_worker,
                                              shuffle=is_shuffle,
                                              sampler=sampler,
                                              batch_size=self._batch_size,
                                              pin_memory=True,
                                              drop_last=True,
                                              collate_fn=self.collate_fn)

    def from_dataset(self, ds: BaseSFTFDataset):
        """

        :param ds:
        :return:
        """
        if ds is None or len(ds) == 0:
            raise ValueError("empty dataset object.")

        last_index = max(1, round(len(ds) * (30 / 100)))
        train_len = len(ds) - last_index
        generator = torch.Generator().manual_seed(1234)
        dataset = torch.utils.data.random_split(ds, [train_len, last_index], generator=generator)
        if len(dataset) > 1:
            self._datasets['train_set'] = dataset[0]
            self._datasets['validation_set'] = dataset[1]
        else:
            self._datasets['train_set'] = dataset[0]

        self._create_loaders()

    def _create(self, update: Optional[bool] = False):
        """
         :return:
        """
        pk_dataset = self._trainer_spec.get_audio_dataset()
        mandatory_keys = ['train_set', 'validation_set', 'test_set', 'ds_type']
        for k in mandatory_keys:
            if k not in pk_dataset:
                raise ValueError(f"Dataset spec has no mandatory key {k}.")

        if not update:
            if pk_dataset['ds_type'] == 'tensor_mel':
                self._createv2_from_tensor()
            if pk_dataset['ds_type'] == 'audio_raw':
                self._create_v2raw()
        else:
            # this might happen if batch size changed and data loader is null
            if self._dataloaders is None:
                raise RuntimeError("Invalid state.")

        self._create_loaders()

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

    def read_batch(self) -> bool:
        """
        Reads single batches, parse it and return true.
        Can be used for pre-check.
        :return:
        """

        result = False
        if not self._initialized_before():
            self._create()

        for _, dataloader in enumerate(self._dataloaders):
            for _, batch in enumerate(dataloader):
                x, y = self.get_batch(batch)
                for j in range(len(batch)):
                    print(batch[j].shape)
                    print(batch[j].device)
                    result = True
                break

        return result

    def benchmark_read(self):
        """
        Benchmark read all dataloaders.

        :return:
        """
        if not self._initialized_before():
            self._create()
        else:
            logger.debug("Already initialized.")

        # enable trace for collate dataloader
        # self.collate_fn.trace()
        t = time.process_time()

        aggregate = 0
        num_batched = 0
        # do full pass over train batch and count time
        for _, dataloader in enumerate(self._dataloaders):
            print(f"data loader: dataset {dataloader}, contains entries {len(dataloader)}")
            total_batches = 0
            for i, batch in enumerate(dataloader):
                total_batches += i
            if total_batches > 0:
                num_batched += 1
            aggregate += total_batches

        elapsed_time = time.process_time() - t
        return elapsed_time, aggregate, num_batched

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


def test_from_subset_data_loader_from_raw():
    """
    # Test all basic cases.

    :return:
    """
    # batch
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    trainer_spec.set_active_dataset('LJSpeechSmallRaw')
    data_loader_a = SFTFDataloader(trainer_spec, batch_size=1, verbose=False)
    dataloaders, collate = data_loader_a.get_loader()
    for i, dataloader in enumerate(dataloaders):
        print("Num batches", len(dataloader))

    # reduction
    data_loader = SFTFDataloader(trainer_spec, batch_size=1, reduction_ration=10, verbose=False)
    dataloaders, collate = data_loader.get_loader()
    for i, dataloader in enumerate(dataloaders):
        print("Num batches", len(dataloader))

    batch_size = [1, 8, 16, 32, 64]
    for batch_size in batch_size:
        dataloader = SFTFDataloader(trainer_spec, batch_size=batch_size, verbose=False)
        elapsed, total_batches, num_batched = dataloader.benchmark_read()
        print(f"batch_size {batch_size}, time took {elapsed:.10f} sec, "
              f"total batches {total_batches} total batches {num_batched}")


def test_from_subset_data_loader_from_tensor():
    """
    # Test all basic cases.

    :return:
    """
    # batch
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    data_loader_a = SFTFDataloader(trainer_spec, batch_size=1, verbose=False)
    dataloaders, collate = data_loader_a.get_loader()
    for i, dataloader in enumerate(dataloaders):
        print("Num batches", len(dataloader))

    # reduction
    data_loader = SFTFDataloader(trainer_spec, batch_size=1, reduction_ration=10, verbose=False)
    dataloaders, collate = data_loader.get_loader()
    for i, dataloader in enumerate(dataloaders):
        print("Num batches", len(dataloader))

    batch_size = [1, 8, 16, 32, 64]
    for batch_size in batch_size:
        dataloader = SFTFDataloader(trainer_spec, batch_size=batch_size, verbose=False)
        elapsed, total_batches, num_batched = dataloader.benchmark_read()
        print(f"batch_size {batch_size}, time took {elapsed:.10f} sec, "
              f"total batches {total_batches} total batches {num_batched}")


def test_create_data_loader_from_tensor():
    """
    The most basic one.
    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    data_loader = SFTFDataloader(trainer_spec, verbose=True)
    dataloaders, collate = data_loader.get_loader()
    for i, dataloader in enumerate(dataloaders):
        print("Num batches", len(dataloader))

    data_loader.benchmark_read()


def test_download_numpy_files():
    """

    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('spectrogram_layer')
    train_dataset = SFTF2Dataset(model_spec, download=True, overwrite=False)

    batch_size = [1, 8, 16, 32, 64]
    for batch_size in batch_size:
        dataloader = SFTFDataloader(trainer_spec, dataset=train_dataset, batch_size=batch_size, verbose=True)
        elapsed, total_batches, num_batched = dataloader.benchmark_read()
        print(f"batch_size {batch_size}, time took {elapsed:.10f} sec, "
              f"total batches {total_batches} total batches {num_batched}")


def test_download_torch_files():
    """

    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('spectrogram_layer')
    train_dataset = SFTF2Dataset(model_spec, download=True, overwrite=False, data_format='tensor_mel', verbose=True)
    train_dataset.set_logger(True)

    batch_size = [1, 8, 16, 32, 64]
    for batch_size in batch_size:
        dataloader = SFTFDataloader(trainer_spec, dataset=train_dataset, batch_size=batch_size, verbose=True)
        elapsed, total_batches, num_batched = dataloader.benchmark_read()
        print(f"batch_size {batch_size}, time took {elapsed:.10f} sec, "
              f"total batches {total_batches} total batches {num_batched}")


if __name__ == '__main__':
    """
    """
    test_create_data_loader_from_tensor()
    test_from_subset_data_loader_from_tensor()
    test_from_subset_data_loader_from_raw()
    test_download_numpy_files()
    test_download_torch_files()
