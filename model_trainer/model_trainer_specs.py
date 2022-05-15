import logging
import os
import shutil
import sys
from pathlib import Path
from time import gmtime, strftime
from typing import List

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from tacotron2.utils import fmtl_print, fmt_print
from text.symbols import symbols
from .specs.dtc_spec import DTC
from .model_files import ModelFiles
from .specs.model_spec import ModelSpec
from loguru import logger


class ExperimentSpecs:
    """

    """

    def __init__(self, template_file_name='config.yaml', verbose=False):
        """

        :param template_file_name:
        :param verbose:
        """

        # a file name or io.string
        self._initialized = None
        self.lr_schedulers = None
        self._optimizers = None
        self.config_file_name = None
        current_dir = os.path.dirname(os.path.realpath(__file__))

        if isinstance(template_file_name, str):
            logger.info("Loading {} file.", template_file_name)
            # a file name or io.string
            self.config_file_name = Path(template_file_name)

        self._verbose = verbose

        #
        self._setting = None

        #
        self._model_spec = None

        #
        self.inited = None

        #
        self.writer = None

        #
        self._active_setting = None

        # list of models
        self.models_specs = None

        # active model
        self.active_model = None

        # dataset specs
        self.dataset_specs = None

        # active dataset
        self.use_dataset = None

        # store pointer to config, after spec read serialize yaml to it.
        self.config = None

        # device
        self.device = 'cuda'
        logger.info("Device {}".format(self.device))

        # if clean tensorboard
        self.clean_tensorboard = False

        self.ignore_layers = ['embedding.weight']

        self.load_mel_from_disk = False

        self.text_cleaners = ['english_cleaners']

        ################################
        # Model Parameters             #
        ################################
        self.n_symbols = len(symbols)
        self.symbols_embedding_dim = 512

        # Encoder parameters
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = 512

        # Decoder parameters
        self.n_frames_per_step = 1  # currently, only 1 is supported
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention parameters
        self.attention_rnn_dim = 1024
        self.attention_dim = 128

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

        # self.learning_rate = 1e-3
        # self.weight_decay = 1e-6
        # 
        self.grad_clip_thresh = 1.0
        self.mask_padding = True  # set model's padded outputs to padded values
        self.dynamic_loss_scaling = True

        self.cudnn_enabled = True
        self.cudnn_benchmark = True

        self.read_from_file()
        # self.optimizer = HParam('optimizer', hp.Discrete(['adam', 'sgd']))

        # model files
        self.model_files = ModelFiles(self.config)
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

        self.writer = SummaryWriter()

        # from tensorboard.plugins.hparams import api as hp
        # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
        # METRIC_ACCURACY = 'accuracy'

        # with SummaryWriter() as w:
        #     for i in range(5):
        #         w.add_hparams({'lr': 0.1 * i, 'bsize': i}, {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i})

    def models_list(self):
        """
        List of network types and sub-network models used for a given model.
        For example GraphRNN has graph edge and graph node model.
        @return: list of models.
        """
        models_types = []
        for k in self._model_spec:
            if k.find('model') != -1:
                models_types.append(k)

    def set_active_dataset(self):
        """Sets active dataset
        :return:
        """
        if 'use_dataset' not in self.config:
            raise Exception("config.yaml must contains valid active settings.")

        self.use_dataset = self.config['use_dataset']

    def set_dataset_specs(self):
        """Sets dataset spec, for example if we need swap dataset.
        :return:
        """
        if 'datasets' not in self.config:
            raise Exception("config.yaml must contains corresponding "
                            "datasets settings for {}".format(self.use_dataset))

        dataset_list = self.config['datasets']
        if self.use_dataset not in dataset_list:
            raise Exception("config.yaml doesn't contain {} template, check config.".format(self.use_dataset))

        self.dataset_specs = self.config['datasets'][self.use_dataset]

    def set_active_model(self):
        """Sets active Model
        :return:
        """
        if 'models' not in self.config:
            raise Exception("config.yaml must contain at least one models list and one model.")

        if 'use_model' not in self.config:
            raise Exception("config.yaml must contain use_model and it must defined.")

        self.active_model = self.config['use_model']
        self.models_specs = self.config['models']
        if self.active_model == 'dts':
            self._model_spec = DTC(self.models_specs[self.active_model], self.dataset_specs)

        if self.active_model not in self.models_specs:
            raise Exception("config.yaml doesn't contain model {}.".format(self.active_model))

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
            raise Exception("config.yaml use undefined variable {} ".format(self._active_setting))

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
            raise Exception("config.yaml doesn't contain optimizers section. Example {}".format(""))

    def read_lr_scheduler(self):
        """
        Read lr lr_schedulers, each model can have different lr lr_schedulers.
        Returns:

        """
        if 'lr_schedulers' in self.config:
            self.lr_schedulers = self.config['lr_schedulers']

    def validate_spec_config(self):
        """ Validate dataset spec,  each spec has own semantics.
        :return:
        """
        mandatory_kv = ["dir", "training_meta", "validation_meta", "test_meta", "file_type"]

        dataset_type = self.dataset_specs['ds_type']
        if dataset_type == "audio":
            for k in mandatory_kv:
                if k not in self.dataset_specs:
                    raise Exception("Audio dataset must contain config entry {}".format(k))

    def read_config(self, debug=False):
        """
        Parse config file and initialize trainer.
        :param debug: will output debug into
        :return: nothing
        """
        if debug:
            logger.debug("Parsing... ", self.config)

        self.set_active_dataset()
        self.set_dataset_specs()
        self.set_active_model()
        self.read_lr_scheduler()
        self.read_optimizer()
        self.validate_spec_config()

        # settings stored internally
        if debug:
            fmt_print("active setting", self.config['active_setting'])

        self.set_active_settings()
        self.inited = True

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
                print(exc)

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
        Args:
            path_dir:

        Returns:

        """
        _dir = str(path_dir)
        if _dir.find('~') != -1:
            sub_dir = _dir.split('~')
            if len(sub_dir) > 1:
                home = Path.home()
                full_path = home / (Path(sub_dir[1][1:]))
                if full_path.exists():
                    return full_path

        return path_dir

    def load_text_file(self, file_name, delim="|", _filter='DUMMY/'):
        """Load from text file name and metadata (text)

        Args:
             file_name: file name.

        Returns:
            dict where key is file name and value text.
            :param _filter:
            :param file_name:
            :param delim:
        """

        target_dir = self.get_dataset_dir()
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

    def get_dataset_dir(self):
        """

        Returns:
        """
        if 'dir' not in self.dataset_specs:
            raise Exception("config.yaml must contain dir entry")

        # resolve home
        path_to_dir = self.resolve_home(self.dataset_specs['dir'])
        if path_to_dir != self.dataset_specs['dir']:
            self.dataset_specs['dir'] = path_to_dir

        return path_to_dir

    def update_meta(self, metadata_file, file_type_filter="wav"):
        """
        Build a dict that hold each file as key, a nested dict contains
        full path to a file,  metadata such as label , text , translation etc.
        :return:
        """
        path_to_dir = self.resolve_home(self.get_dataset_dir())
        file_meta_kv = self.load_text_file(metadata_file)
        files = self.model_files.make_file_dict(path_to_dir,
                                                file_type_filter,
                                                filter_dict=file_meta_kv)

        logger.debug("Updating metadata {}", metadata_file)
        for k in file_meta_kv:
            if k in files:
                files[k]['meta'] = file_meta_kv[k]

        return files

    def get_training_meta_file(self):
        """
        Returns:  Path to file contain meta information , such as file - text
        """
        if 'training_meta' not in self.dataset_specs:
            raise Exception("config.yaml must contain training_meta entry.")
        return self.dataset_specs['training_meta']

    def get_validation_meta_file(self):
        """
        Returns:  Path to file contain meta information , such as file - text
        """
        if 'validation_meta' not in self.dataset_specs:
            raise Exception("config.yaml must contain validation_meta entry.")
        return self.dataset_specs['validation_meta']

    def get_test_meta_file(self):
        """
        Returns:  Path to file contain meta information , such as file - text
        """
        if 'test_meta' not in self.dataset_specs:
            raise Exception("config.yaml doesn't contain test_meta entry.")
        return self.dataset_specs['test_meta']

    def build_training_set(self):
        """
        :return:
        """
        if 'training_meta' in self.dataset_specs:
            return self.update_meta(self.get_training_meta_file())

        logger.warning("training_meta is empty dict.")
        return {}

    def build_validation_set(self):
        """
        :return:
        """
        if 'validation_meta' in self.dataset_specs:
            return self.update_meta(self.get_validation_meta_file())

        logger.warning("validation_meta is empty dict.")
        return {}

    def build_test_set(self):
        """
        :return:
        """
        if 'test_meta' in self.dataset_specs:
            return self.update_meta(self.get_test_meta_file())
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

    def get_raw_audio_ds_files(self):
        """
        :return: dict
        """
        train_set = self.build_training_set()
        validation_set = self.build_validation_set()
        test_set = self.build_test_set()

        # in case we need to do sanity check how many file and meta resolved.
        if self._verbose:
            logger.info("{} rec in train set contains.".format(len(train_set)))
            logger.info("{} rec validation set contains.".format(len(validation_set)))
            logger.info("{} rec test set contains.", format(len(test_set)))

            train_record = self.num_records_in_txt(self.get_training_meta_file(), self.get_dataset_dir())
            val_record = self.num_records_in_txt(self.get_validation_meta_file(), self.get_dataset_dir())
            test_record = self.num_records_in_txt(self.get_test_meta_file(), self.get_dataset_dir())

            logger.info("Train set metadata contains {}".format(train_record))
            logger.info("Validation set metadata contains {}".format(val_record))
            logger.info("Test set metadata contains {}".format(test_record))

        ds_dict = dict(train_set=self.build_training_set(),
                       validation_set=self.build_validation_set(),
                       test_set=self.build_test_set())

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
            # check if  file exists
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
                raise Exception("Failed locate {} file.".format(str(ds_dict[k])))

        if len(pt_dict) > 0:
            pt_dict['ds_type'] = 'tensor_mel'

        return pt_dict

    def get_audio_dataset(self):
        """Method return audio dataset spec.
        :return:
        """
        if 'format' not in self.dataset_specs:
            raise Exception("config.yaml doesn't dataset format entry.")

        self._verbose = True

        data_type = self.dataset_specs['format']
        file_type = self.dataset_specs['file_type']
        if data_type == 'raw':
            logger.debug("Dataset type raw file")
            return self.get_raw_audio_ds_files()
        if data_type == 'tensor_mel':
            logger.debug("Dataset type torch tensor mel")
            return self.get_tensor_audio_ds_files()

        return None

    def tensorboard_sample_update(self):
        """
        Return true if early stopping enabled.
        :return:  default value False
        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'early_stopping' in self._setting:
            return True

    def seed(self):
        """

        Returns:

        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'seed' in self._setting:
            logger.debug("Model uses fixed seed", self._setting['seed'])
            return self._setting['seed']

        return 1234

    def is_fp16_run(self):
        """

        Returns:

        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'fp16' in self._setting:
            if self._verbose:
                fmtl_print("Model uses fp16", self._setting['fp16'])
            return self._setting['fp16']

        return False

    def is_distributed_run(self) -> bool:
        """
        If trainer need to distribute training.
        :return:
        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'distributed' in self._setting:
            if self._verbose:
                fmtl_print("Model uses distributed", self._setting['distributed'])
            return self._setting['distributed']

        return False

    def get_backend(self):
        """
        Return model backend setting such as nccl
        :return:
        """
        if 'backend' in self._setting:
            if self._verbose:
                fmtl_print("Model backend", self._setting['backend'])
            return self._setting['backend']
        return False

    def dist_url(self):
        """

        :return:
        """
        if 'url' in self._setting:
            if self._verbose:
                fmtl_print("Model backend", self._setting['url'])
            return self._setting['url']
        return False

    def get_model_spec(self) -> ModelSpec:
        """
        Return active model spec
        :return:
        """
        return self._model_spec

    def fp16_run(self):
        """

        :return:
        """
        return self.is_fp16_run()

    def epochs(self) -> int:
        """
        Return epochs,
        Note each graph has own total epochs. ( depend on graph size)
        :return: number of epochs to run for given dataset, default 100
        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'epochs' in self._setting:
            return int(self._setting['epochs'])

        return 100

    def validate_settings(self):
        """

        Returns:

        """
        if 'batch_size' in self._setting:
            return int(self._setting['batch_size'])

    @property
    def batch_size(self):
        """Returns batch size for current active model, each dataset has own batch size.
        Model batch size
        :return:
        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'batch_size' in self._setting:
            return int(self._setting['batch_size'])

        return 32

    def is_load_model(self) -> bool:
        """
        Return true if model must be loaded, default will return false.
        """
        if 'load_model' in self.config:
            return bool(self.config['load_model'])

        return False

    def is_save(self) -> bool:
        """
        Return true if model saved during training.
        """
        return bool(self.config['save_model'])

    def is_save_per_iteration(self) -> bool:
        """
        Return true if model saved during training.
        """
        if 'save_per_iteration' in self._setting:
            return bool(self._setting['save_per_iteration'])

        return False

    def get_active_model(self) -> str:
        """
         Return model that indicate as current active model.
         It is important to understand, we can switch between models.
        """
        return self.active_model

    def get_active_sub_models(self):
        """

        Returns:

        """
        _active_model = self.config['use_model']
        _models = self.config['models']
        if _active_model not in _models:
            raise Exception("config.yaml doesn't contain model {}.".format(_active_model))

        _model = _models[_active_model]
        sub_models = []
        for k in _model:
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
        Save model at epochs , by default trainer will use early stopping
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
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'type' in scheduler_spec:
                return scheduler_spec['type']

        return None

    def get_sub_model_lr_scheduler(self, sub_model_name) -> str:
        """

        Args:
            sub_model_name:

        Returns:

        """
        _active_model = self.config['use_model']
        _models = self.config['models']
        if _active_model not in _models:
            raise Exception("config.yaml doesn't contain model {}.".format(_active_model))

        if sub_model_name is not None:
            _model = _models[_active_model]
            sub_model = _model[sub_model_name]
            if 'lr_scheduler' in sub_model:
                return sub_model['lr_scheduler']

        return ""

    def get_optimizer(self, alias_name: str):
        """
        Method return optimizer setting.
        Args:
            alias_name: alias name in config.  It binds optimizer to model

        Returns: dict that hold optimizer settings

        """
        return self._optimizers[alias_name]

    def optimizer_type(self, alias_name: str, default=False) -> str:
        """
        Returns optimizer type for a given alias , if default is passed , will return default.
        Args:
            alias_name:
            default:

        Returns:

        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'type' in opt:
                return str(opt['type'])

        return "Adam"

    def get_active_mode(self):
        """

        :return:
        """
        if 'use_model' not in self.config:
            raise Exception("Make sure spec contains use_model key value.")

        return self.config['use_model']

    def get_models(self):
        """

        :return:
        """
        if 'models' not in self.config:
            raise Exception("Make sure spec contains use_model key value.")

        return self.config['models']

    def get_sub_model_optimizer(self, sub_model_name) -> str:
        """
         Return optimizer name for a given sub_model.
         Each sub model might have own optimizer.

        Args:
            sub_model_name:

        Returns:

        """
        _active_model = self.get_active_mode()
        _models = self.get_models()
        if _active_model not in _models:
            raise Exception("config.yaml doesn't contain model {}.".format(_active_model))

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
            raise Exception("config.yaml doesn't contain model {}.".format(_active_model))

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

        Args:
            alias_name:

        Returns:

        """
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'learning_rate' in scheduler_spec:
                return scheduler_spec['learning_rate']

    def lr_lambdas(self, alias_name):
        """
         A function which computes a multiplicative factor given an integer parameter epoch,
         or a list of such functions, one for each group in optimizer.param_groups.
        Args:
            alias_name:

        Returns:

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
        Adam learning rate (default: 1e-3)

        Args:
            default:  will return default value
            alias_name:

        Returns:

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
            raise Exception("Training must be initialized first.")

        if self._setting is None:
            raise Exception("Initialize settings first")

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
            raise Exception("Training must be initialized first.")

        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'tensorboard_update' in self._setting:
            if self.batch_size <= int(self._setting['tensorboard_update']):
                logger.warning("Tensorboard update rate less than batch size")
                return 1
            return int(self._setting['tensorboard_update'])

        if self.batch_size == 1:
            return 1

        return min(self.batch_size, 1)

    def get_tensorboard_writer(self):
        return self.writer

    def initialized(self):
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized
