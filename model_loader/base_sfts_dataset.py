# STFT25, Mel dataset
#
# It base class that does all low level, each class on top can overwrite
# Mustafa. B
#
import os
import random
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional
from urllib.error import URLError

import numpy as np
import torch
import torch.utils.data
from loguru import logger
from tqdm import tqdm

from model_loader import ds_util
from model_trainer.specs.tacatron_spec import TacotronSpec
from model_trainer.utils import load_wav_to_torch
from text import text_to_sequence


class DatasetError(Exception):
    """Base class for other exceptions"""
    pass


class BaseSFTFDataset(torch.utils.data.Dataset):
    """

    Base class for SFTS mel spectrogram dataset,
    I tried to match same spec for mel, in case we need compare different models.

    Model should contain following structure.

      max_wav_value: 32768.0  # normalization factor
      frames_per_step: 1      # frame per step
      sampling_rate: 22050    # at what rate we sample
      filter_length: 1024     # length of the FFT window
      win_length: 1024        # each frame of audio is windowed by
      hop_length: 256
      n_mel_channels: 80
      mel_fmin: 0.0
      mel_fmax: 8000.0

    # Loading example.
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('encoder')
    pk_dataset = trainer_spec.get_audio_dataset()

    # out data_format audio hence raw audio and text
    train_dataset = SFTF2Dataset(model_spec,
                                 list(pk_dataset['train_set'].values()),
                                 data_format='audio_raw')

    Save as numpy npy file
      train_dataset.save_as_numpy("test.npy", overwrite=True)

    Get iterator for a file, will empty one example at the time
      ds = train_dataset.example_from_numpy("test.npy", as_torch=False)
      for data in ds:
         x. y = data

    Load entire dataset to memory.
      train_dataset.load_from_numpy("test.npy", as_torch=False)


    Working with raw audio.  Dict list contain dict key path , path a file,  meta text.
    train_dataset = SFTF2Dataset(model_spec,
                                 list(pk_dataset['train_set'].values()),
                                 data_format='audio_raw')

    [{'path': '/LJSpeechSmall/wavs/LJ010-0188.wav',
      'meta': 'Oxford expressed little anxiety or concern.\n', 'label': '0'}


    For tensor data set. The

    Let say you serialized all object in batch to disk.
    The data must contain following keys.

    # Data Keys dict_keys(['filter_length', 'hop_length', 'win_length', 'n_mel_channels',
    'sampling_rate', 'mel_fmin', 'mel_fmax', 'data'])
    """

    def __init__(self,
                 model_spec: TacotronSpec,
                 data=None,
                 root: Optional[str] = "dtc",
                 data_format: Optional[str] = "numpy_mel",
                 fixed_seed: Optional[bool] = True,
                 shuffle: Optional[bool] = False,
                 is_trace_time: Optional[bool] = False,
                 in_memory: Optional[bool] = True,
                 download: Optional[bool] = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 verbose: Optional[bool] = False,
                 overfit=False) -> None:
        """
        Data formats
           tensor_mel  Use case, entire dataset passed, batchified via data loader and saved on disk.
           numpy_mel   Use case, entire dataset already in numpy format.
           numpy_file  Use case, entire dataset saved. i.e. each record one hot encoded text, mel,
                       other spectrum data.
           torch_file  Use case, entire dataset saved.
           audio_raw   Raw audio file that already annotated with text.

        :param model_spec:
        :param data: if data format is audio it must be a list that hold dict_keys(['path', 'meta', 'label'])
                     if data format torch, data must contain tensor is must dict must hold key data.
                     if data format numpy  and string it must point to dataset file
                     if data format torch_file and data string it path to a file.

        :param data_format: tensor_mel, numpy_mel, audio_raw
        :param is_trace_time:
        :param fixed_seed: if we want shuffle dataset
        :param shuffle: shuffle or not,  in case DDP you must not shuffle
        :param in_memory: store entire dataset in memory. (expect raw audio)
               if in_memory is false,  len can't return value, hence you need iterate manually.
        """
        super(torch.utils.data.Dataset, self).__init__()
        logger.disable(__name__)
        self.set_logger(verbose)

        #
        self._overfiting = overfit
        self._mirrors = [
            "https://www.dropbox.com/s/i2wzklf60vs5y3b/subset.npy?dl=0",
            "http://www.dropbox.com/s/i2wzklf60vs5y3b/subset.npy?dl=0"
        ]
        #
        self._resources = [
            ("subset.npy", "f526cb36b33aa0295166ba1cdc5204ee"),
        ]
        #
        self.target_transform = target_transform
        #
        self.transform = transform
        #
        self.download = download
        #
        self.root = root
        #
        self.is_download = False
        #
        if download:
            if not self.download_ifneed():
                warnings.warn("Failed download.")
                raise RuntimeError("Failed download")
            # data is string to a file
            self.data = self._dataset_file
        else:
            self.data = data
        #
        self._in_memory = in_memory
        # raw folder
        self.raw_folder = None
        #
        self.is_trace_time = is_trace_time
        #
        self._shuffle = shuffle
        # data types
        self.is_a_tensor = False
        self.is_a_numpy = False
        self.is_audio = False
        # this value used only if in_memory is False
        self._data_len = None
        # hold a model spec.
        self._model_spec = model_spec
        # validate specs
        self._validate_specs(model_spec)

        # note if we download we need indicate.
        if data_format is None or len(data_format) == 0:
            raise DatasetError("Dataset file type format is none or empty")

        # if data string, resolve
        if data is not None and isinstance(data, str):
            p = Path(data).expanduser()
            resolved = Path(data).resolve()
            if not resolved.exists():
                raise Exception("File {} not found.".format(data))
            self._dataset_file = str(p)

        # validate data format
        if 'tensor_mel' in data_format:
            # from file
            logger.debug("Creating dataset for dict with torch tensors.")
            self.is_a_tensor = True
        elif 'numpy_mel' in data_format:
            # from file
            logger.debug("Creating dataset from numpy file {}.", self._dataset_file)
            self.is_a_numpy = True
        elif 'audio_raw' in data_format:
            logger.debug("Creating dataset for audio file list.")
            self.is_audio = True
        else:
            raise ValueError("Dataset file type format is unsupported. Supported format",
                             'tensor', 'numpy', 'audio_raw')

        # check dataset contain key
        if self.is_audio is False and self.is_a_numpy is False:
            if not isinstance(data, dict):
                raise DatasetError("Dataset type torch tensor., "
                                   "must contain 'data' key', dict doesn't contain key 'data'")
            if 'data' not in data:
                raise DatasetError("Dataset type torch tensor, "
                                   "must contain 'data' key', dict doesn't contain key 'data'")
            if len(data['data']) == 0:
                raise DatasetError("Dataset type torch tensor: has no data.")

            # print("Type", type(data['data'][0][0]))

        self.text_cleaners = model_spec.get_text_cleaner()
        if self.text_cleaners is None:
            raise DatasetError("Text pre processor can't be a none. check specification.")

        if fixed_seed:
            random.seed(model_spec.get_seed())

        self.is_trace_time = False

        # data pointer,  data len read from header only if we use generator
        self._data = []
        self._data_len = 0
        self._num_tensors = 0
        self._data_iterator = None

        if self.is_a_tensor:
            self._data = data['data']
        elif self.is_audio:
            if len(data) == 0:
                warnings.warn("Received empty data.")
            if 'path' not in data[0]:
                raise ValueError("Audio dataset must contain key 'path' that point to audio file.")
            if 'meta' not in data[0]:
                raise ValueError("Audio dataset must contain key 'meta' contains original text.")
            self._data = data
        elif self.is_a_numpy:
            # path to a file
            if isinstance(data, str) and len(data) > 0 or self.is_download:
                if self._in_memory:
                    self._dataset_file = self.data
                    logger.debug("Loading dataset from numpy file in memory. {}".format(self._dataset_file))
                    self.load_from_numpy(self._dataset_file)
                    self._data_len = len(self._data)
                else:
                    logger.debug("Creating dataset from numpy file. as iterator, no random access")
                    self._dataset_file = self.data
                    self._data_len, self._num_tensors = self._read_numpy_header(self._dataset_file)
                    self._data_iterator = self.example_from_numpy(self._dataset_file)
                logger.debug("Dataset contains {} entries.".format(self._data_len))
        else:
            raise DatasetError("Unknown format.")

        if shuffle:
            # todo cross check shuffling
            if self.is_audio:
                random.shuffle(self._data)
            if self.is_a_numpy and self._in_memory:
                random.shuffle(self._data)
            if self.is_a_tensor:
                random.shuffle(self._data)

    def _validate_specs(self, model_spec):
        """

        :param model_spec:
        :return:
        """
        # mel specs, during load we validate
        if model_spec.filter_length() < 0:
            warnings.warn("Check dataset specification. Filter length is less than 0.")
        self.filter_len = model_spec.filter_length()
        if model_spec.win_length() < 0:
            warnings.warn("Check dataset specification. Window length is less than 0.")
        self.win_len = model_spec.win_length()
        if model_spec.hop_length() < 0:
            warnings.warn("Check dataset specification. Hop length is less than 0.")
        self.hop_len = model_spec.hop_length()
        if model_spec.sampling_rate() < 0:
            warnings.warn("Check dataset specification. Sampling rate is less than 0.")
        self.sampling_rate = model_spec.sampling_rate()
        if model_spec.mel_fmin() < 0:
            warnings.warn("Check dataset specification. fmin rate is less than 0.")
        self.fmin = model_spec.mel_fmin()
        if model_spec.max_wav_value() < 0:
            warnings.warn("Check dataset specification. normalization wav rate is less than 0.")
        self.max_wav_value = model_spec.max_wav_value()
        if model_spec.n_mel_channels() < 0:
            warnings.warn("Check dataset specification. number of mel filter is less than 0.")
        self.mel_channels = model_spec.n_mel_channels()

        self.fmax = model_spec.mel_fmax()
        self._header_size = 7

    def is_in_memory(self) -> bool:
        """
        Return ff data in memory or not.
        :return:
        """
        return self._in_memory

    def save_as_numpy(self, filename, overwrite=False):
        """
        Saves internal data to numpy file.  For example if dataset created
        from raw audio file, each get item return records from memory or at run
        time convert audio to tensor.

        All n tuples saved as separate record in numpy file.
        (one hot,  mel,  other features)

        First record is metadata,   num tuples each record contains, number entries total.
        sample rate , window len used to generate file.

        :param overwrite:
        :param filename:
        :return:
        """
        if self._in_memory:
            raise Exception("Unsupported for none in memory dataset.")

        p = Path(filename)
        if p.is_dir():
            raise Exception("Provide valid filename. {}".format(filename))

        if overwrite is False:
            if p.is_file():
                raise Exception("File already exists.{}".format(filename))

        if len(self._data) == 0:
            raise Exception("Dataset has no entries it, empty nothing to save.")

        # num tuples in record text, mel, other features
        num_tuples = len(self[0])
        assert num_tuples > 0

        header = np.zeros([self._header_size, 1], dtype=int)
        header[0] = num_tuples
        # len of records
        header[1] = len(self._data)
        # sample rate data mel generated
        assert self.sampling_rate > 0
        header[2] = self.sampling_rate
        # filter length used to generate mel
        assert self.filter_len > 0
        header[3] = self.filter_len
        # window length used to pad mel
        assert self.win_len > 0
        header[4] = self.win_len
        # hop length used to generate mel
        assert self.hop_len > 0
        header[5] = self.hop_len
        # fmax value
        assert self.fmax > 0
        header[6] = self.fmax

        with open(filename, 'wb') as f:
            # number of tuples, num records, and all data related how mel generated.
            np.save(f, header)
            for i in tqdm(range(0, len(self._data))):
                ds_record = self[i]
                assert num_tuples == len(ds_record)
                for idx in range(0, len(ds_record)):
                    np.save(f, ds_record[idx].numpy())

    def _read_numpy_header(self, filename):
        """
        Reads header from numpy file, the header contains data how mel was generated,
        num filters, records ec.
        :param filename:
        :return:
        """
        with open(filename, 'rb') as f:
            # load header and check
            header = np.load(f)
            if header.shape[0] < self._header_size:
                raise ValueError("Invalid header.")

            tuple_len = int(header[0])
            assert tuple_len > 0
            data_len = int(header[1])
            assert data_len > 0

        return data_len, tuple_len

    def example_from_numpy(self, filename: str, as_torch=True, strict=False):
        """
         Return one example from numpy dataset,
         Each records is one hot encoded vector numpy,
         mel matrix and other features.

         Method returns an iterator, each entry is tuple.

        :param filename:  filename to numpy npy file
        :param as_torch: return example as torch tensor
        :param strict: if header mismatched ( filter , window etc will crash)
        :return: tuples (one hot, mel, other feature)
        """
        if filename is None or len(filename) == 0:
            raise Exception("Empty file name.")

        p = Path(filename)
        if not p.exists():
            raise Exception(f"File {filename} not found.")

        with open(filename, 'rb') as f:
            # load header and check
            # num tuples
            header = np.load(f)
            if header.shape[0] < self._header_size:
                raise ValueError("Invalid header.")

            tuple_len = int(header[0])
            assert tuple_len > 0
            # total size records
            data_len = int(header[1])
            assert data_len > 0
            self._data_len = data_len

            # all meta
            sampling_rate = header[2]
            if sampling_rate != self.sampling_rate:
                warnings.warn(f"sample rate mismatch numpy file {sampling_rate} dataset {self.sampling_rate}")

            filter_len = header[3]
            if filter_len != self.filter_len:
                warnings.warn(f"filter length mismatch numpy file {filter_len} dataset {self.filter_len}")

            win_len = header[4]
            if win_len != self.win_len:
                warnings.warn(f"window length mismatch numpy file {win_len} dataset {self.win_len}")

            hop_len = header[5]
            if hop_len != self.hop_len:
                warnings.warn(f"window length mismatch numpy file {hop_len} dataset {self.hop_len}")

            fmax = header[6]
            if fmax != self.fmax:
                warnings.warn(f"window length mismatch numpy file {fmax} dataset {self.fmax}")

            for idx in range(0, data_len):
                data_object = []
                for tuple_idx in range(0, tuple_len):
                    data = np.load(f)
                    if as_torch:
                        data_object.append(torch.from_numpy(data))
                    else:
                        data_object.append(data)
                yield tuple(data_object)

    @abstractmethod
    def audiofile_to_mel(self, filename: str):
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

    @abstractmethod
    def get_item(self, index: int):
        """
        :param index:
        :return:
        """
        pass

    @abstractmethod
    def tensor_from_audio(self, index):
        """
        Should return tensors from audio
        :param index:
        :return:
        """
        pass

    def normalize_audio(self, filename) -> [torch.Tensor, int]:
        """
        Load wav file and normalize audio file.

        :param filename: path to audio file.
        :return: Tensor and sample rate.
        """
        audio, sample_rate = load_wav_to_torch(filename)
        normalized_wav = audio / self.max_wav_value
        normalized_wav = normalized_wav.unsqueeze(0)
        normalized_wav = torch.autograd.Variable(normalized_wav,
                                                 requires_grad=False)
        return normalized_wav, sample_rate

    def text_to_tensor(self, input_seq):
        """
        One hot encoder for text seq.

        :param input_seq:
        :return:
        """
        text_norm = torch.IntTensor(text_to_sequence(input_seq, self.text_cleaners))
        return text_norm

    def load_from_numpy(self, filename: str, as_torch=True, pbar_disable=False) -> None:
        """
        Load entire dataset to a memory from numpy file.

        :param pbar_disable: progress bar on or off.
        :param as_torch: load internally as torch Tensor, otherwise ndarray
        :param filename: path to a numpy npy file.
        :return: None
        """
        examples = self.example_from_numpy(filename, as_torch=as_torch)
        if self._data is None:
            self._data = []
        for example in tqdm(examples, disable=pbar_disable):
            self._data.append(example)

    def clear(self):
        """
        Clear entire data in dataset
        :return:
        """
        del self._data
        self._data = []

    def __getitem__(self, index: int) -> type[torch.Tensor, torch.Tensor]:
        """
        # TODO check dataloader random access
        :param index:
        :return:
        """
        # if not in memory no random access, fetch next
        if not self.is_in_memory():
            if self.is_a_numpy:
                return next(self._data_iterator)
        # if tensor or numpy return
        if self.is_a_tensor or self.is_a_numpy:
            return self._data[index]

        # if audio from dict
        if self.is_audio:
            self.tensor_from_audio(index)
            if 'meta' not in self._data[index]:
                raise DatasetError("data must contain meta key")
            if 'path' not in self._data[index]:
                raise DatasetError("data must contain path key")
            text = self.text_to_tensor(self._data[index]['meta'])
            mel = self.audiofile_to_mel(self._data[index]['path'])
            return text, mel

        raise Exception("Unknown data format")

    def __len__(self):
        """
        For in memory, return len of _data , otherwise read data size from header.
        Note in this case __getitem__ will return item from iterator.
        :return:
        """
        if len(self._data) > 0 and self._overfiting:
            return 1
        if self._in_memory:
            return len(self._data)
        return self._data_len

    # def _check_exists(self) -> bool:
    #     """
    #     No integrity check for now
    #     """
    #     return all(check_integrity(file) for file in (self.images_file, self.labels_file))

    def mirrors(self) -> tuple[str, str, str]:
        """
        Generator emit link for each file and mirror.
        :return:
        """
        for filename, checksum in self._resources:
            for mirror in self._mirrors:
                if mirror.endswith("/"):
                    yield f"{mirror}{filename}", filename, checksum
                else:
                    yield mirror, filename, checksum

    def download_ifneed(self) -> bool:
        """
        Download dataset.
        :return:
        """
        # check if exists
        # if self.check_exists():
        #     return
        # make dir if needed
        if self.is_download:
            warnings.warn("File already downloaded.")

        os.makedirs(self.root, exist_ok=True)
        # download if needed
        self.is_download = False
        for (mirror, filename, md5) in self.mirrors():
            try:
                logger.debug("downloading from mirror {} file {}".format(mirror, filename))
                self.is_download, file_path = ds_util.download_dataset(url=mirror,
                                                                       path=self.root,
                                                                       filename=filename,
                                                                       checksum=md5)
                self._dataset_file = file_path
                if self.is_download:
                    break
            except URLError as e:
                logger.debug("Failed to download {} {}. Moving to next mirror.".format(mirror, filename))
                logger.error(e)
                continue

        if self.is_download:
            logger.info("File downloaded.")

        return self.is_download

    @staticmethod
    def set_logger(is_enable: bool) -> None:
        """
        Method sets logging level.
        :param is_enable:
        :return:
        """
        if is_enable:
            logger.enable(__name__)
        else:
            logger.disable(__name__)
