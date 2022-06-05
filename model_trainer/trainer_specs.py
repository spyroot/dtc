import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from loguru import logger

import model_loader
from model_loader.ds_util import check_integrity
from model_trainer.utils import fmt_print
from text.symbols import symbols
from .model_files import ModelFiles
from .spec_dispatcher import SpecsDispatcher
from .specs.model_spec import ModelSpec

# from torch.utils.tensorboard import SummaryWriter

MODULE_NAME = "ExperimentSpecs"

logger.disable(__name__)


class TrainerSpecError(Exception):
    """Base class for other exceptions"""
    pass


class ExperimentSpecs:
    """
    """

    def __init__(self, spec_config='config.yaml', verbose=False, no_dir=False):
        """

        :param spec_config:
        :param verbose:
        :param no_dir if true will not build structure directory structures for trainer.
        """
        self._model_dispatcher = SpecsDispatcher()
        self._overfit = None

        # dispatcher
        self._model_dispatcher = SpecsDispatcher()
        ExperimentSpecs.set_logger(False)
        self._verbose: bool = False

        # a file name or io.string
        self._initialized = None
        self.lr_schedulers = None
        self._optimizers = None
        self.config_file_name = None
        #
        self._current_dir = os.path.dirname(os.path.realpath(__file__))

        # caller can pass string
        if isinstance(spec_config, str):
            logger.info("Loading {} file.", spec_config)
            p = Path(spec_config).expanduser()
            p = p.resolve()
            self.config_file_name = p

        self._setting = None

        self._model_spec: ModelSpec

        self._inited: bool = False
        # for now it disabled , ray has some strange issue with tensorboard.
        self.writer = None

        # reference to active settings
        self._active_setting = None

        # list of models
        self.models_specs = None

        # active model
        self.active_model = None

        # dataset specs
        self._dataset_specs = None

        # active dataset
        self._use_dataset = None

        # store pointer to config, after spec read serialize yaml to it.
        self.config = None

        # device
        self.device = 'cuda'
        logger.info("Device {}".format(self.device))

        # if clean tensorboard
        self.clean_tensorboard = False

        # ignore layers
        self.ignore_layers = ['embedding.weight']

        #
        self.load_mel_from_disk = False

        #
        self.text_cleaners = ['english_cleaners']

        ################################
        self.n_symbols = len(symbols)

        # Location Layer parameters
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Mel-post processing network parameters
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate = False

        self.mask_padding = True  # set model's padded outputs to padded values
        self.dynamic_loss_scaling = True

        self.cudnn_enabled = True
        self.cudnn_benchmark = True

        self.read_from_file()
        # self.optimizer = HParam('optimizer', hp.Discrete(['adam', 'sgd']))

        # model files
        self.model_files = ModelFiles(self.config, parent=self, verbose=self._verbose)
        if no_dir:
            self.model_files.build_dir()

        self.setup_tensorboard()
        self.initialized()

    def setup_tensorboard(self):
        """
        Setup tensorflow dir
        """
        # time_fmt = strftime("%Y-%m-%d-%H", gmtime())
        # logger.info("tensorboard log dir {}".format(self.model_files.get_model_log_dir()))
        # logging.basicConfig(filename=str(self.model_files.get_model_log_dir() / Path('train' + time_fmt + '.log')),
        #                     level=logging.DEBUG)

        # if bool(self.config['regenerate']):
        #     logger.info("tensorboard erasing old logs")
        #     if os.path.isdir("tensorboard"):
        #         shutil.rmtree("tensorboard")

    # self.writer = SummaryWriter()

    # from tensorboard.plugins.hparams import api as hp
    # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    # METRIC_ACCURACY = 'accuracy'

    # with SummaryWriter() as w:
    #     for i in range(5):
    #         w.add_hparams({'lr': 0.1 * i, 'bsize': i}, {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i})

    def models_list(self):
        """
        List of network types and sub models used for a model.
        For example GraphRNN has graph edge and graph node model.
        :return: list of models
        """
        models_types = []
        for k in self._model_spec:
            if k.find('model') != -1:
                models_types.append(k)

    def set_active_dataset(self, dataset_name: Optional[str] = "") -> None:
        """
        Sets active dataset,  by default it reads from spec and
        use use_dataset key.  If we need swap form cmd without changing
        in yaml file.

        :param dataset_name: a dataset that already present in config.
        :return:
        """

        if dataset_name is not None and len(dataset_name) > 0:
            if not self.is_initialized():
                raise TrainerSpecError("Uninitialized trainer spec. First you need create object from a spec.")

            datasets_specs = self.get_datasets_specs()
            if dataset_name in datasets_specs:
                self._validate_spec_config(self.config['datasets'][dataset_name])
                self.config['use_dataset'] = dataset_name
                self._use_dataset = dataset_name
                self._update_dataset_specs()
            else:
                raise TrainerSpecError(f"Dataset {dataset_name} not found in {self.config_file_name}.")

        if 'use_dataset' not in self.config:
            raise TrainerSpecError(f"{self.config_file_name}  must "
                                   f"contain valid 'use_dataset' settings.")

        self._use_dataset = self.config['use_dataset']

    def _update_dataset_specs(self):
        """
        Update dataset spec, for example if we need swap dataset.
        :return:
        """
        if 'datasets' not in self.config:
            raise TrainerSpecError(f"{self.config_file_name} spec must contain "
                                   f"corresponding datasets settings for {self._use_dataset}")

        dataset_list = self.config['datasets']
        if self._use_dataset not in dataset_list:
            raise TrainerSpecError(f"{self.config_file_name}  doesn't contain "
                                   f"{self._use_dataset} template, please check the config.")

        self._dataset_specs = self.config['datasets'][self._use_dataset]
        #   self.validate_spec_config()

    def get_dataset_names(self):
        """
        :return:
        """
        return list(self.config['datasets'].keys())

    def set_active_model(self, model_name: Optional[str] = None) -> None:
        """
        Sets active Model.

        :return:
        """
        # self.spec_dispatcher = self.create_spec_dispatch()
        if 'models' not in self.config:
            raise TrainerSpecError(f"{self.config_file_name} must contain at "
                                   f"least a models and specification for each model.")
        if model_name is not None:
            if model_name not in self.config['models']:
                raise TrainerSpecError(f"{self.config_file_name} unknown model.")
            self.config['use_model'] = model_name

        if 'use_model' not in self.config:
            raise TrainerSpecError("config.yaml must contain use_model "
                                   "and it must defined.")

        self.active_model = self.config['use_model']
        self.models_specs = self.config['models']
        self._model_dispatcher.has_creator(self.active_model)

        if not self._model_dispatcher.has_creator(self.active_model):
            raise TrainerSpecError(f"It looks like model {self.active_model} is unknown.")

        # get dispatch and pass to creator and update current, active model spec.
        spec_dispatcher = self._model_dispatcher.get_model(self.active_model)
        self._model_spec = spec_dispatcher(self.models_specs[self.active_model],
                                           self._dataset_specs, self._verbose)

        if self.active_model not in self.models_specs:
            raise TrainerSpecError("config.yaml doesn't contain model {}.".format(self.active_model))

        # set model spec
        # self.model_spec = self.models_specs[self.active_model]

    def set_active_settings(self, debug=False):
        """

        :param debug:
        :return:
        """
        self._active_setting = self.config['active_setting']
        _settings = self.config['settings']
        if debug:
            logger.debug("Settings list {}".format(_settings))

        if self._active_setting not in _settings:
            raise TrainerSpecError("config.yaml use undefined variable {} ".format(self._active_setting))

        self._setting = _settings[self._active_setting].copy()
        if debug:
            logger.debug("Active settings {}".format(self._setting))

    def read_optimizer(self):
        """
        Read optimizer setting.
        Single config can have different optimizer.  Each model spec has name bind.

        Returns:
        """
        # checks if optimizers setting present
        if 'optimizers' in self.config:
            self._optimizers = self.config['optimizers']
        else:
            raise TrainerSpecError("config.yaml doesn't contain optimizers section. Example {}".format(""))

    def read_lr_scheduler(self):
        """
        Read lr lr_schedulers, each model can have different lr lr_schedulers.
        Returns:

        """
        if 'lr_schedulers' in self.config:
            self.lr_schedulers = self.config['lr_schedulers']

    def _validate_spec_config(self, spec):
        """ Validate dataset spec,  each spec has own semantics.
        :return:
        """

        target_checksum = {}

        mandatory_kv = ["dir", "training_meta", "validation_meta", "test_meta", "file_type", "format", "file_type"]
        file_kv = ["validation_meta", "validation_meta", "test_meta"]

        if 'ds_type' not in spec:
            raise TrainerSpecError(f"Audio dataset must contain ds_type config entry.")

        dataset_type = spec['ds_type']
        if dataset_type == "audio":
            for k in mandatory_kv:
                if k not in spec:
                    raise TrainerSpecError(f"Audio dataset must contain {k} config entry.")

        ds_dict = {}

        if 'checksums' in spec:
            target_checksum = spec['checksums']

        # TODO this need be merged with dataset files
        for k in file_kv:
            some_file = self._dataset_specs[k]
            self.resolve_home(spec['dir'])
            ds_dir = self.resolve_home(spec['dir'])
            full_path = Path(ds_dir) / some_file
            final_path = full_path.expanduser().resolve()

            if not final_path.absolute():
                raise TrainerSpecError(f"Failed resolve path to a file. {some_file}")
            if not final_path.exists():
                raise TrainerSpecError(f"Please check config {self.config_file_name}, "
                                       f"file not found {some_file} in directory {spec['dir']}")
            if not final_path.is_file():
                raise TrainerSpecError(f"Please check config {self.config_file_name}, a path resolved, "
                                       f"not a file. {some_file}. dir {spec['dir']}")

        if 'dataset_files' in spec:
            ds_files = spec['dataset_files']
            for ds_file in ds_files:
                ds_dir = self.resolve_home(self.get_dataset_dir())
                full_path = Path(spec['dir']) / ds_file
                final_path = full_path.expanduser().resolve()
                if not final_path.absolute():
                    raise TrainerSpecError(f"Failed resolve path to a file. {ds_file}")
                if not final_path.exists():
                    raise TrainerSpecError(f"Please check config {self.config_file_name}, "
                                           f"file not found {ds_file} in directory {spec['dir']}")
                if not final_path.is_file():
                    raise TrainerSpecError(f"Please check config {self.config_file_name}, a path resolved, "
                                           f"not a file. {ds_file}. dir {spec['dir']}")

                if len(target_checksum) > 0:
                    checksum = model_loader.ds_util.md5_checksum(final_path)
                    if checksum not in target_checksum:
                        print("expected", target_checksum)
                        raise TrainerSpecError(f"Checksum {checksum} mismatched. check file {final_path}")

    def validate_spec_config(self):
        """ Validate dataset spec,  each spec has own semantics.
        :return:
        """
        return self._validate_spec_config(self._dataset_specs)

    def read_config(self, debug=False):
        """
        Parse config file and initialize trainer.
        :param debug: will output debug into
        :return: nothing
        """
        if debug:
            logger.debug("Parsing... ", self.config)

        self.set_active_dataset()
        self._update_dataset_specs()
        self.set_active_model()
        self.read_lr_scheduler()
        self.read_optimizer()
        self.validate_spec_config()

        # settings stored internally
        if debug:
            fmt_print("active setting", self.config['active_setting'])

        self.set_active_settings()
        self._update_all_keys()
        self._validate_all_mandatory()
        self._inited = True

    def read_from_file(self, debug=False):
        """
        Read config file and initialize trainer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(self.config_file_name, "r") as stream:
            try:
                logger.info("Reading configuration file from {}".format(self.config_file_name))
                self.config = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print("Check file with yaml linter it has error. ", str(exc))
                sys.exit(2)

        self.read_config()

    def read_from_stream(self, buffer, debug=False):
        """
        Read config file from a stream
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            if self._verbose:
                logger.info("Reading config from io buffer")
            self.config = yaml.load(buffer, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit("Failed parse yaml")

        self.read_config()

    def get_model_sub_models(self) -> List[str]:
        """
        @return:  Return list of all sub models.
        """
        keys = self._model_spec.keys()
        return [k for k in keys]

    @staticmethod
    def resolve_home(path_dir):
        """
        If config.yaml contain ~ resolve home and make a path to directory
        full path.

        :param path_dir:
        :return:
        """
        _dir = str(path_dir)
        if _dir.find('~') != -1:
            sub_dir = _dir.split('~')
            if len(sub_dir) > 1:
                home = Path.home()
                full_path = home / (Path(sub_dir[1][1:]))
                expanded = full_path.expanduser()
                resolved = expanded.resolve()
                if resolved.exists():
                    return resolved

        return path_dir

    def load_text_file(self, file_name: str, delim: Optional[str] = "|",
                       _filter: Optional[str] = 'DUMMY/', ds_dir: Optional[str] = None):
        """
        Load text, parse the file and create dictionary
        Load from text file name and metadata (text)

        :param file_name:
        :param delim: delimiter for each colum space
        :param _filter:
        :param ds_dir:  if require indicate dir, by default method that call load_text_file uses active dataset.
        :return:
        """

        if ds_dir is None:
            target_dir = self.get_dataset_dir()
        else:
            target_dir = ds_dir

        file_path = Path(target_dir) / file_name
        file_meta_kv = {}
        with open(file_path, 'r', newline='', encoding='utf8') as meta_file:
            lines = meta_file.readlines()
            for line in lines:
                tokens = line.split(delim)
                if len(tokens) == 2:
                    file_name = tokens[0].replace(_filter, '')
                    file_meta_kv[file_name] = tokens[1]

        return file_meta_kv

    def _get_dataset_dir(self, spec):
        """
        Dataset dir are in specs. key dir
        :return: dataset dir.
        """

        if 'dir' not in self._dataset_specs:
            raise TrainerSpecError(f"{self.config_file_name} must contain dir entry.")

        # resolve home
        path_to_dir = self.resolve_home(self._dataset_specs['dir'])
        if path_to_dir != self._dataset_specs['dir']:
            self._dataset_specs['dir'] = path_to_dir

        return path_to_dir

    def get_dataset_dir(self):
        """
        Dataset dir are in specs. key dir
        :return: dataset dir.
        """

        if 'dir' not in self._dataset_specs:
            raise TrainerSpecError(f"{self.config_file_name} must contain dir entry.")

        # resolve home
        path_to_dir = self.resolve_home(self._dataset_specs['dir'])
        if path_to_dir != self._dataset_specs['dir']:
            self._dataset_specs['dir'] = path_to_dir

        return path_to_dir

    def update_meta(self, metadata_file: str, file_type_filter="wav", ds_dir=None):
        """
         Builds a dict that hold each file as key,
         a nested dict contains full path to a file,
         metadata such as label, text, translation etc.

        :param metadata_file:
        :param file_type_filter:
        :param ds_dir: a dir from where we resolve path to a file.
        :return:
        """
        if ds_dir is not None:
            target_dir = ds_dir
        else:
            target_dir = self.get_dataset_dir()

        path_to_dir = self.resolve_home(target_dir)
        file_meta_kv = self.load_text_file(metadata_file, ds_dir=path_to_dir)
        files = self.model_files.make_file_dict(path_to_dir,
                                                file_type_filter,
                                                filter_dict=file_meta_kv)

        logger.debug("Updating metadata {}", metadata_file)
        for k in file_meta_kv:
            if k in files:
                files[k]['meta'] = file_meta_kv[k]

        return files

    def get_dataset_meta_file(self, key: str, spec=None):
        """
        Return path to a meta file.
        :param key:
        :param spec:
        :return:
        """
        if spec is not None and len(key) > 0:
            if key not in spec:
                raise TrainerSpecError(f"config.yaml must contain {key} entry.")
            return spec[key]

        if key not in self._dataset_specs:
            raise TrainerSpecError(f"config.yaml must contain {key} entry.")

        return self._dataset_specs[key]

    def get_training_meta_file(self, spec=None) -> str:
        """
        Return path to a meta file.
        :param spec:
        :return:
        """
        return self.get_dataset_meta_file(key='training_meta', spec=spec)

    def get_validation_meta_file(self, spec=None) -> str:
        """
        Returns:  Path to file contain meta information , such as file - text
        """
        return self.get_dataset_meta_file(key='validation_meta', spec=spec)

    def get_test_meta_file(self, spec=None) -> str:
        """
        Returns:  Path to file contain meta information , such as file - text
        """
        return self.get_dataset_meta_file(key='test_meta', spec=spec)

    def build_training_set(self, spec=None):
        """
        :return:
        """
        if spec is not None and 'training_meta' in spec:
            return self.update_meta(self.get_training_meta_file(spec=spec), ds_dir=spec['dir'])

        if 'training_meta' in self._dataset_specs:
            return self.update_meta(self.get_training_meta_file())

        logger.warning("training_meta is empty dict.")
        return {}

    def build_validation_set(self, spec=None):
        """
        :return:
        """
        if spec is not None:
            if 'validation_meta' in spec:
                return self.update_meta(self.get_validation_meta_file(spec=spec))

        if 'validation_meta' in self._dataset_specs:
            return self.update_meta(self.get_validation_meta_file(spec))

        logger.warning("validation_meta is empty dict.")
        return {}

    def build_test_set(self, spec=None):
        """
        Build a dictionary prepared for training.
        :param spec:
        :return:
        """
        if spec is not None:
            if 'test_meta' in spec:
                return self.update_meta(self.get_test_meta_file(spec=spec))

        if 'test_meta' in self._dataset_specs:
            return self.update_meta(self.get_test_meta_file(spec))

        logger.warning("test_meta is empty dict.")
        return {}

    def num_records_in_txt(self, file_path, target_dir=None):
        """Count number of lines in text file

        Args:
            target_dir:
            file_path:
        Returns:

        """
        full_path = file_path
        if target_dir is not None:
            self.resolve_home(target_dir)
            logger.info("target dir {}".format(target_dir))
            full_path = Path(target_dir) / Path(file_path)

        if Path(full_path).exists():
            with open(full_path, 'r', newline='', encoding='utf8') as file:
                nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
                line_count = len(nonempty_lines)
                return line_count

        return 0

    def get_raw_audio_ds_files(self, spec, ds_dir=None, strict=False):
        """
        Return audio files dataset.

        :param strict: will validate every file exists.
        :param spec: a spec that will use to perform lookup
        :param ds_dir: in case we need overwrite dir where meta files are.
        :return:
        """
        if ds_dir is not None:
            expanded = Path(str(ds_dir)).expanduser()
            resolved_path = expanded.resolve()

            if resolved_path.is_dir() and resolved_path.is_dir():
                target_dir = str(resolved_path)
            else:
                raise ValueError("invalid directory.")
        else:
            target_dir = self.get_dataset_dir()

        logger.info(f"Building dataset from {target_dir}")
        train_set = self.build_training_set(spec=spec)
        validation_set = self.build_validation_set(spec=spec)
        test_set = self.build_test_set(spec=spec)

        # in case we need to do sanity check how many file and meta resolved.
        if self._verbose:
            logger.info("{} rec in train set contains.".format(len(train_set)))
            logger.info("{} rec validation set contains.".format(len(validation_set)))
            logger.info("{} rec test set contains.", format(len(test_set)))

        if strict:
            train_record = self.num_records_in_txt(self.get_training_meta_file(spec=spec), target_dir)
            val_record = self.num_records_in_txt(self.get_validation_meta_file(spec=spec), target_dir)
            test_record = self.num_records_in_txt(self.get_test_meta_file(spec=spec), target_dir)
            logger.info("Train set metadata contains {}".format(train_record))
            logger.info("Validation set metadata contains {}".format(val_record))
            logger.info("Test set metadata contains {}".format(test_record))

        ds_dict = dict(train_set=self.build_training_set(spec=spec),
                       train_meta=self.get_training_meta_file(spec=spec),
                       validation_set=self.build_validation_set(spec=spec),
                       validation_meta=self.get_validation_meta_file(spec=spec),
                       test_meta=self.get_validation_meta_file(spec=spec),
                       test_set=self.build_test_set(spec=spec))

        if len(ds_dict) > 0:
            ds_dict['ds_type'] = 'audio_raw'

        return ds_dict

    def get_tensor_audio_ds_files(self):
        """
        :return:
        """
        ds_dir = self.resolve_home(self.get_dataset_dir())
        ds_dict = dict(train_set=Path(ds_dir) / self.get_training_meta_file(),
                       validation_set=Path(ds_dir) / self.get_validation_meta_file(),
                       test_set=Path(ds_dir) / self.get_test_meta_file())

        pt_dict = {}
        for k in ds_dict:
            # check if file exists
            if ds_dict[k].exists():
                if self._verbose:
                    logger.info("Loading tensor mel from {}".format(str(ds_dict[k])))
                dataset_from_pt = torch.load(ds_dict[k])
                if self._verbose:
                    logger.info("Dataset filter length {}".format(dataset_from_pt['filter_length']))
                    logger.info("Dataset mel channels {}".format(dataset_from_pt['n_mel_channels']))
                    logger.info("Dataset contains records {}".format(len(dataset_from_pt['data'])))
                pt_dict[k] = dataset_from_pt
            else:
                raise TrainerSpecError(f"Failed locate {str(ds_dict[k])} file. "
                                       f"Please check config {self.config_file_name}")

        if len(pt_dict) > 0:
            pt_dict['ds_type'] = 'tensor_mel'

        return pt_dict

    def get_datasets_specs(self):
        """
        Return all datasets specs.

        Example how yaml look like.

        datasets:
          LJSpeech:
            ds_type: "audio"
            dir: "~/Dropbox/Datasets/LJSpeech-1.1"
            training_meta: ljs_audio_text_train_filelist.txt
            validation_meta:  ljs_audio_text_val_filelist.txt
            test_meta: ljs_audio_text_test_filelist.txt
            meta: metadata.csv
            recursive: False
            file_type: "wav"

        :return:
        """
        if not self.is_initialized():
            raise TrainerSpecError("Uninitialized trainer spec. First you need create object from a spec.")

        if 'datasets' not in self.config:
            raise TrainerSpecError(f"Current configuration in {self.config_file_name} has no dataset.")

        return self.config['datasets']

    def get_dataset_spec(self, dataset_name="") -> dict:
        """
         Method return dataset based on name i.e key.

        :param dataset_name:
        :return:
        """
        if not self.is_initialized():
            raise TrainerSpecError("Training must be initialized first.")

        dataset_specs = self.get_datasets_specs()
        if dataset_name not in dataset_specs:
            raise TrainerSpecError(f"Dataset not found in {self.config_file_name}.")

        return dataset_specs[dataset_name]

    def get_audio_dataset(self, dataset_name: Optional[str] = None):
        """
        Method return audio dataset spec, by default it uses active dataset.
        If dataset_name provided by a caller , it will return respected
        dataset.
        :param dataset_name: a dataset name
        :return: a dict
        """

        if dataset_name is None or len(dataset_name) == 0:
            dataset_spec = self._dataset_specs
        else:
            dataset_spec = self.get_dataset_spec(dataset_name)

        if 'format' not in dataset_spec:
            raise TrainerSpecError(f"{self.config_file_name} {dataset_name} doesn't dataset format entry.")

        if 'file_type' not in dataset_spec:
            raise TrainerSpecError(f"{self.config_file_name} {dataset_name} doesn't dataset file_type entry.")

        if 'ds_type' not in dataset_spec:
            raise TrainerSpecError(f"{self.config_file_name} {dataset_name} doesn't dataset ds_type entry.")

        self._verbose = True
        data_format = dataset_spec['format']
        file_type = dataset_spec['file_type']
        ds_type = dataset_spec['ds_type']

        if ds_type.lower().strip() == 'audio':
            if data_format.lower().strip() == 'raw':
                return self.get_raw_audio_ds_files(spec=dataset_spec)
            elif data_format.lower().strip() == 'tensor_mel':
                logger.debug("Dataset type torch tensor mel.")
                # todo
                return self.get_tensor_audio_ds_files()
            elif data_format.lower().strip() == 'numpy_mel':
                logger.debug("Dataset type numpy mel.")
                # todo
                return self.get_tensor_audio_ds_files()
            else:
                raise TrainerSpecError(f"Unsupported format {ds_type} for audio dataset.")
        else:
            raise TrainerSpecError(f"active dataset type {ds_type} error.")

    def tensorboard_sample_update(self):
        """
        Return true if early stopping enabled.
        :return:  default value False
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'early_stopping' in self._setting:
            return True

    def seed(self):
        """

        Returns:

        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'seed' in self._setting:
            logger.debug("Model uses fixed seed", self._setting['seed'])
            return self._setting['seed']

        return 1234

    def is_amp(self) -> False:
        """
        Return true if active setting set to run in mixed precision mode.
        :return:
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'is_amp' in self._setting:
            logger.debug(f"Model uses is_amp {self._setting['is_amp']} log level {self._verbose}")
            return self._setting['is_amp']

        return False

    def is_distributed_run(self) -> bool:
        """
        If trainer need to distribute training.
        :return:
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'distributed' in self._setting:
            if self._verbose:
                logger.debug("Model uses distributed", self._setting['distributed'])
            return self._setting['distributed']

        return False

    def get_backend(self):
        """
        Return model backend setting such as nccl
        :return:
        """
        if 'backend' in self._setting:
            if self._verbose:
                logger.debug("Model backend", self._setting['backend'])
            return self._setting['backend']

        return "nccl"

    def get_master_address(self):
        """
        Return model master address.
        :return: Default local host
        """
        if 'backend' in self._setting:
            if self._verbose:
                logger.debug("Model master address {}".format(self._setting['master_address']))
            return str(self._setting['master_address'])

        return "localhost"

    def get_master_port(self) -> str:
        """Return a DDP tcp port
        :return: Default "54321"
        """
        if 'backend' in self._setting:
            if self._verbose:
                logger.debug("Model master port".format(self._setting['master_port']))
            return str(self._setting['master_port'])

        return "54321"

    def dist_url(self) -> str:
        """Return tcp url used for ddp.
        :return:
        """
        if 'url' in self._setting:
            if self._verbose:
                logger.debug("Model backend {}".format(self._setting['url']))
            return self._setting['url']

        return ""

    def get_model_spec(self) -> ModelSpec:
        """
        Return active model spec.

        :return:
        """
        return self._model_spec

    def fp16_run(self):
        """

        :return:
        """
        return self.is_amp()

    def epochs(self) -> int:
        """
        Return epochs, Note each graph has own total epochs.
        (Depend on graph size).

        :return: number of epochs to run for given dataset, default 100
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'epochs' in self._setting:
            return int(self._setting['epochs'])

        return 100

    def validate_settings(self):
        """

        :return:
        """
        if 'batch_size' in self._setting:
            return int(self._setting['batch_size'])

    def batch_size(self):
        """Returns batch size for current active model, each dataset has own batch size.
        Model batch size
        :return:
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        # when we overfiting we take only 1 example
        if self._overfit:
            return 1

        if 'batch_size' in self._setting:
            return int(self._setting['batch_size'])

        return 32

    def set_batch_size(self, batch_size):
        """Returns batch size for current active model, each dataset has own batch size.
        Model batch size
        :return:
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        # when we overfiting we take only 1 example
        if self._overfit:
            return 1

        self._setting['batch_size'] = batch_size

    def is_load_model(self) -> bool:
        """
        Return true if model must be loaded, for resuming training,
        default will return false.
        """
        if 'load_model' in self.config:
            return bool(self.config['load_model'])

        return False

    def is_save_required(self) -> bool:
        """
        Return true if model saved during training.
        """
        return bool(self.config['save_model'])

    def is_save_iteration(self) -> bool:
        """
        Return true if model saved during training.
        """
        if 'save_per_iteration' in self._setting:
            return bool(self._setting['save_per_iteration'])

        return False

    def get_active_model_name(self) -> str:
        """
         Return model that indicate as current active model.
         It is important to understand, we can switch between models.
        """
        return self.active_model

    def get_active_sub_models(self):
        """
         Return model, all sub_models specification.
        :return:
        """
        _active_model = self.config['use_model']
        _models = self.config['models']
        if _active_model not in _models:
            raise Exception("config.yaml doesn't contain model {}.".format(_active_model))

        _model = _models[_active_model]
        sub_models = []
        for k in _model:
            if 'model' not in _model[k]:
                continue
            if 'state' in _model[k]:
                if _model[k]['state'].lower() == 'disabled'.lower():
                    logger.debug(f"Model {k} disabled.")
                    continue
            sub_models.append(k)
        return sub_models

    def epochs_save(self) -> int:
        """
        Save model at epochs , by default trainer will use early stopping
        TODO add early stopping optional
        """
        if 'epochs_save' in self._setting:
            return int(self._setting['epochs_save'])

        return 100

    def predict(self) -> int:
        """
        Do prediction / validation inside a training loop.
        """
        if 'predict' in self._setting:
            return int(self._setting['predict'])

        return 100

    def is_train_verbose(self):
        """
        @return: Return true if we do verbose training
        """
        if 'debug' in self.config:
            t = self.config['debug']
            if 'train_verbose' in t:
                return bool(t['train_verbose'])
        return False

    def lr_scheduler(self, alias_name):
        """
         Returns lr scheduler by name, each value of lr_scheduler in dict
        :param alias_name: alias name defined in config.
                           The purpose of alias bind different scheduler config to model
        :return:
        """
        if alias_name is None or len(alias_name) == 0:
            return None

        if self.lr_schedulers is not None:
            for elm in self.lr_schedulers:
                spec = elm
                if 'name' in spec and alias_name in spec['name']:
                    return spec

        return None

    def lr_scheduler_type(self, alias_name):
        """
        Returns lr scheduler type.

        :param alias_name: alias_name: alias name defined in config.
                           The purpose of alias bind different scheduler config to model
        :return:
        """
        if alias_name is None or len(alias_name) == 0:
            raise TrainerSpecError("Supplied alias_name name is empty.")

        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'type' in scheduler_spec:
                return scheduler_spec['type']

        return None

    def get_sub_model_lr_scheduler(self, model_layer) -> str:
        """
        Method return scheduler for a given models sub layer.

        :param model_layer:
        :return:
        """
        if model_layer is None or len(model_layer) == 0:
            raise TrainerSpecError("Supplied model layer name is empty.")

        _active_model = self.config['use_model']
        _models = self.config['models']

        if _active_model not in _models:
            raise TrainerSpecError(f"Config doesn't contain model {_active_model}.")

        _model = _models[_active_model]
        if 'state' in _models and _model['state'] == 'disabled':
            return ""

        model_layer = _model[model_layer]
        if 'lr_scheduler' in model_layer:
            return model_layer['lr_scheduler']

        return ""

    def get_optimizer(self, alias_name: str):
        """
        Method return what optimizer to attach. alias name is alias as it defined in config.
        Example:

        optimizers:
          node_optimizer:
            eps: 1e-8
            weight_decay: 0
            amsgrad: False
            momentum=0:
            betas: [0.9, 0.999]
            type: Adam

        :param alias_name: alias_name: alias as it defined in config.yaml
        :return:
        """
        if alias_name is None or len(alias_name) == 0:
            raise TrainerSpecError("Supplied optimizer alias_name layer name is empty.")

        return self._optimizers[alias_name]

    def optimizer_type(self, alias_name: str, default=False) -> str:
        """
        Method return what optimizer to attach. alias name is alias as it defined in config.
        Example:
        Returns optimizer type for a given alias , if default is passed , will return default.

        :param alias_name: alias as it defined in config.yaml
        :param default:
        :return:
        """
        if alias_name is None or len(alias_name) == 0:
            raise TrainerSpecError("Supplied optimizer alias_name layer name is empty.")

        if default is True:
            return "Adam"

        opt = self.get_optimizer(alias_name)
        if 'type' in opt:
            return str(opt['type'])

        return "Adam"

    def get_active_model(self) -> str:
        """
        Return active model
        :return:
        """
        if 'use_model' not in self.config:
            raise TrainerSpecError("Make sure spec contains use_model key value pair.")
        return self.config['use_model']

    def get_models(self):
        """
        Return entire model specification.

        :return:
        """
        if 'models' not in self.config:
            raise TrainerSpecError("Make sure spec contains use_model key value.")

        return self.config['models']

    def get_sub_model_optimizer(self, sub_model_name) -> str:
        """
         Return optimizer name for a given sub_model.
         Each sub model might have own optimizer.

        Args:
            sub_model_name:

        Returns:

        """
        _active_model = self.get_active_model()
        _models = self.get_models()
        if _active_model not in _models:
            raise TrainerSpecError("config.yaml doesn't contain model {}.".format(_active_model))

        logger.info("received {}", sub_model_name)

        if sub_model_name is not None:
            _model = _models[_active_model]
            if sub_model_name in _model:
                sub_model = _model[sub_model_name]
                if 'optimizer' in sub_model:
                    return sub_model['optimizer']

        return ""

    def get_model_lr_scheduler(self, model_name=None) -> str:
        """

        :param model_name:
        :return:
        """
        _active_model = self.config['use_model']
        _models = self.config['models']
        if _active_model not in _models:
            raise TrainerSpecError("config.yaml doesn't contain model {}.".format(_active_model))

        if model_name is None:
            _model = _models[_active_model]
            if 'lr_scheduler' in _model:
                return _model['lr_scheduler']

        return ""

    def milestones(self, alias_name):
        """

        :return:
        """
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'milestones' in scheduler_spec:
                return scheduler_spec['milestones']

        return ""

    def set_milestones(self, alias_name, milestone):
        """

        :param:
        :return:
        """
        self.graph_specs['milestones'] = milestone

    def min_lr(self, alias_name):
        """

        Args:
            alias_name:

        Returns:

        """
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'milestones' in scheduler_spec:
                return scheduler_spec['milestones']
        pass

    def lr_rate(self, alias_name):
        """

        :param alias_name:
        :return:
        """
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'learning_rate' in scheduler_spec:
                return scheduler_spec['learning_rate']

    def lr_lambdas(self, alias_name):
        """
         A function which computes a multiplicative factor given an integer parameter epoch,
         or a list of such functions, one for each group in optimizer.param_groups.

        :param alias_name:
        :return:
        """
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'lr_lambdas' in scheduler_spec:
                return scheduler_spec['lr_lambdas']

        return None

    def adam_betas(self, alias_name, default=False) -> [float, float]:
        """
         adam coefficients used for computing running averages of gradient
         and its square (default: (0.9, 0.999))

        :param alias_name:
        :param default:
        :return:
        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'betas' in opt:
                return opt['betas']

        return [0.9, 0.999]

    def adam_eps(self, alias_name, default=False) -> float:
        """
        Term added to the denominator to improve numerical stability
        Default: 1e-8

        Args:
            alias_name: optimizer name
            default: return default value

        Returns:

        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'eps' in opt:
                return float(opt['eps'])

        return 1e-8

    def weight_decay(self, alias_name: str, default=False) -> float:
        """
            Adam or SGD weight decay (L2 penalty) (default: 0)

        Args:
            alias_name: optimize alias name
            default: true if ams grad must be enabled, default False

        Returns:

        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'weight_decay' in opt:
                return float(opt['weight_decay'])

        return float(0)

    def amsgrad(self, alias_name: str, default=False) -> bool:
        """
         Whether to use the AMSGrad variant of this algorithm,
         Setting dictates whether to use the AMSGrad variant.

        Args:
            alias_name: optimize alias name
            default: true if ams grad must be enabled, default False

        Returns: true if ams grad must be enabled, default False

        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'amsgrad' in opt:
                return bool(opt['amsgrad'])

        return False

    def sample_time(self) -> int:
        """
        :return: number of time take sample during prediction.
        """
        if 'training' in self.config:
            t = self.config['training']
            if 'sample_time' in t:
                return int(t['sample_time'])

        return 0

    def momentum(self, alias_name: str, default=False) -> float:
        """
         SGD momentum factor (default: 0)
        Args:
            alias_name:
            default: (default: 0)

        Returns: SGD momentum facto

        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'momentum' in opt:
                return float(opt['momentum'])

        return float(0)

    def dampening(self, alias_name: str, default=False) -> float:
        """
        SGD dampening for momentum (default: 0)
        Returns:
        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'dampening' in opt:
                return float(opt['dampening'])

        return float(0)

    def nesterov(self, alias_name: str, default=False) -> bool:
        """
            Enables nesterov momentum (default: False)

        Args:
            alias_name: optimize alias name
            default: true if ams grad must be enabled, default False
        Returns:
        """

        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'nesterov' in opt:
                return bool(opt['nesterov'])

        return False

    def eta_min(self, alias_name: str):
        """
        Minimum learning rate. Default: 0.

        Args:
            alias_name:

        Returns:

        """
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'eta_min' in scheduler_spec:
                return scheduler_spec['eta_min']

        return 0

    def gamma(self, alias_name: str) -> float:
        """
        Multiplicative factor of learning rate decay. Default: 0.1.
        Args:
            alias_name:

        Returns:

        """
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'gamma' in scheduler_spec:
                return float(scheduler_spec['gamma'])

        return 0.1

    def t_max(self, alias_name: str):
        """
        Maximum number of iterations.
        Returns:

        """
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 't_max' in scheduler_spec:
                return scheduler_spec['t_max']

        return None

    def optimizer_learning_rate(self, alias_name: str, default=False) -> float:
        """
        learning rate (default: 1e-3),  if no setting return default.
        :param alias_name:  a name in config.
        :param default:     (default: 1e-3)
        :return:
        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'learning_rate' in opt:
                return float(opt['learning_rate'])
            if 'lr' in opt:
                return float(opt['lr'])

        return float(1e-3)

    def console_log_rate(self) -> int:
        """
        Setting dictates when to log each epoch statistic.
        Returns: Default 100
        """
        if self.is_initialized() is False:
            raise TrainerSpecError("Training must be initialized first.")

        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'console_log_rate' in self._setting:
            return int(self._setting['console_log_rate'])

        return 100

    def tensorboard_update_rate(self) -> int:
        """
        Setting dictates when to log each epoch statistic.
        Returns: Default update rate is batch_size if batch size > 1
                 otherwise update rate is batch_size
        """
        if self.is_initialized() is False:
            raise TrainerSpecError("Training must be initialized first.")

        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'tensorboard_update' in self._setting:
            if self.batch_size() <= int(self._setting['tensorboard_update']):
                logger.warning("Tensorboard update rate less than batch size.")
                return 1
            return int(self._setting['tensorboard_update'])

        if self.batch_size == 1:
            return 1

        return min(self.batch_size(), 1)

    def get_tensorboard_writer(self):
        """
        Return tensorboard writer object.
        :return:
        """
        return self.writer

    def initialized(self):
        """
        :return:
        """
        self._initialized = True

    def is_initialized(self) -> bool:
        """
        Return true if trainer spec properly initialized and parsed read all yaml file.
        :return:
        """
        if self._initialized and self._setting is not None:
            return True
        return False

    def set_distributed(self, value):
        """
         Update if distributed run or not.
        :return:
        """
        if self.is_initialized and 'distributed' in self._setting:
            self._setting['distributed'] = value

    def is_grad_clipped(self) -> bool:
        """
        Return of settings set to grap clipped.
        :return: Default false.
        """
        if self.is_initialized() is False:
            raise TrainerSpecError("Training must be initialized first.")

        if 'grad_clipping' in self._setting:
            return bool(self._setting['grad_clipping'])

        return False

    def grad_clip_thresh(self) -> float:
        """
        Returns of settings set to gradient clipped thresh
        :return: Default 1.0
        """
        if self.is_initialized() is False:
            raise TrainerSpecError("Training must be initialized first.")

        if 'grad_max_norm' in self._setting:
            return float(self._setting['grad_max_norm'])

        print("return default")
        return 1.0

    @staticmethod
    def set_logger(is_enable: bool) -> None:
        """

        :param is_enable:
        :return:
        """
        if is_enable:
            logger.disable(__name__)
            # logger.enable(__name__)
        else:
            logger.disable(__name__)

    def get_training_strategy(self, model_name: str) -> str:
        """
        Return strategy.

        Example..
        strategy:
              tacotron25:
                type: sequential
                order:
                 - spectrogram_layer
                 - wav2vec
              dts:
                type: sequential
                order:
                 - spectrogram_layer
                 - wav2vec

        :return: Strategy type sequential, round-robin etc.
        """
        if self.is_initialized() is False:
            raise TrainerSpecError("Training must be initialized first.")

        _active_model = self.config['use_model']
        _models = self.config['models']

        if 'strategy' not in self.config:
            raise TrainerSpecError("For complex model with many sub-models you need define a strategy.")

        _strategy = self.config['strategy']
        if model_name not in _strategy:
            raise TrainerSpecError(f"Can't find any train strategy for active model '{model_name}'. Check config.")

        model_strategy_spec = _strategy[model_name]
        if 'type' not in model_strategy_spec:
            raise TrainerSpecError("Strategy must contain type.")

        return model_strategy_spec['type'].lower()

    def is_backup_before_save(self):
        pass

    def set_overfit(self):
        """
        :return:
        """
        self._overfit = True

    def is_overfit(self):
        """
        :return:
        """
        return self._overfit

    @staticmethod
    def get_default_train_set_key() -> str:
        """ default key suppose  to be everywhere the same """
        return 'train_set'

    @staticmethod
    def get_default_val_set_key() -> str:
        return 'validation_set'

    @staticmethod
    def get_default_test_set_key() -> str:
        return 'test_set'

    @staticmethod
    def get_audio_dataset_keys() -> list[str]:
        """
        :return:
        """
        return ['train_set', 'validation_set', 'test_set']

    @staticmethod
    def get_dataset_data_key() -> str:
        """
        Default key for data
        :return:
        """
        return "data"

    def get_active_dataset_name(self) -> str:
        print(self._dataset_specs)
        return self._use_dataset

    def _update_all_keys(self):
        # self._config = change_keys(self.config)
        # print(self._config)
        pass

    def _validate_root_mandatory(self, spec):
        """
        :param spec:
        :return:
        """
        mandatory_root_key = ['train', 'use_dataset', 'use_model',
                              'draw_prediction', 'load_model',
                              'load_epoch', 'save_model', 'regenerate',
                              'active_setting', 'evaluate',
                              'root_dir', 'log_dir', 'nil_dir', 'graph_dir',
                              'results_dir', 'figures_dir',
                              'prediction_dir', 'model_save_dir',
                              'metrics_dir', 'datasets', 'settings',
                              'optimizers', 'lr_schedulers',
                              'strategy', 'models']
        for k in mandatory_root_key:
            if k not in spec:
                raise TrainerSpecError(f"Invalid specification {k} not present.")

    def _validate_all_mandatory(self):
        self._validate_root_mandatory(self.config)

    def get_active_dataset_spec(self):
        """
        :return:
        """
        if not self.is_initialized():
            raise TrainerSpecError("Uninitialized trainer spec. First you need create object from a spec.")

        return self._dataset_specs

    def get_dataset_format(self, ds_spec):
        """
        this mainly placeholder.
        i.e I'll move all dataset to separate class.
        :param ds_spec:
        :return:
        """
        if 'format' not in ds_spec:
            return ""
        return ds_spec['format']


    def is_audio_raw(self, spec: dict):
        """

        :param spec:
        :return:
        """
        if 'format' in spec and 'ds_type' in spec:
            return spec['format'] == 'raw' and spec['ds_type'] == "audio"

        return False

    def set_epochs(self, epochs: int):
        """

        :param epochs:
        :return:
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")
        self._setting['epochs'] = epochs

    def is_random_sample(self) -> True:
        """

        :return:
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'random_sampler' in self._setting:
            return self._setting['random_sampler']

        return False

    def is_sequential(self):
        """

        :return:
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'sequential_sampler' in self._setting:
            return self._setting['sequential_sampler']

        return False

    def get_tuner_spec(self):
        """
        Return ray specification,  config re presets
        a dict pass to tuner.
        :return:
        """
        if 'ray' in self.config:
            return self.config['ray']
        return {}

    def update_grad_clip(self, grad_clip_rate: float):
        """
        Update gradient max norm clip.

        :param grad_clip_rate:
        :return:
        """
        if grad_clip_rate > 1.0:
            warnings.warn("Grad clip must be in range 0.0 - 1.0.")
            return

        if grad_clip_rate < 0.0:
            warnings.warn("Grad clip must be in range 0.0 - 1.0.")
            return

        if 'grad_max_norm' not in self._setting:
            warnings.warn("Current setting has no grad_max_norm.")
            return

        self._setting['grad_max_norm'] = float(grad_clip_rate)

    def get_data_loader(self):
        """
        :return:
        """
        if self._setting is None:
            raise TrainerSpecError("Initialize settings first")

        if 'dataloader' in self._setting:
            return self._setting['dataloader']

        return {}

    def get_data_loader_setting(self, loader_name: str, key: str):
        """
        Data loader settings.

        dataloader:
          train_set:
            num_workers: 1
            drop_last: True
            pin_memory: True
            shuffle: True
          validation_set:
            num_workers: 1
            drop_last: False
            pin_memory: True
            shuffle: False

        :return:
        """
        data_loader = self.get_data_loader()
        if loader_name in data_loader:
            if key in data_loader[loader_name]:
                return data_loader[loader_name][key]
        return None

    def is_drop_last(self, k) -> bool:
        """
        :param k:
        :return:
        """
        val = self.get_data_loader_setting(k, 'drop_last')
        if val is None:
            return False

        return bool(val)

    def num_workers(self, k):
        """

        :param k:
        :return:
        """
        val = self.get_data_loader_setting(k, 'num_workers')
        if val is None:
            return False

        return int(val)

    def is_pin_memory(self, k):
        """
        :param k:
        :return:
        """
        val = self.get_data_loader_setting(k, 'pin_memory')
        if val is None:
            return False

        return bool(val)

    def is_shuffle(self, k):
        """

        :param k:
        :return:
        """
        val = self.get_data_loader_setting(k, 'shuffle')
        if val is None:
            return False
        return bool(val)


def remove_junk(val):
    return val.strip().to_lower()


def change_keys(obj, convert):
    """
    Recursively goes through the dictionary obj and replaces keys with the convert function.
    """
    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, dict):
        new = obj.__class__()
        for k, v in obj.items():
            new[convert(k)] = change_keys(v, convert)
    elif isinstance(obj, (list, set, tuple)):
        new = obj.__class__(change_keys(v, convert) for v in obj)
    else:
        return obj
    return new
