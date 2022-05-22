import os
import pathlib
from os.path import join
from pathlib import Path

import torch
from loguru import logger


class ModelFiles:
    """

    """

    def __init__(self, config, root_dir=".", name_generator=None, dir_walker_callback=None, verbose=False):
        """

        :param root_dir:
        """
        self._verbose = verbose
        self.set_logger(verbose)
        self.config = config
        self.name_generator = name_generator
        self.dir_walker_callback = dir_walker_callback

        self.root_dir = root_dir
        if root_dir == ".":
            self.root_dir = Path(".").resolve()

        self._dir_input = self.root_dir
        self._dirs = {}
        self._dir_result = Path(self._dir_input) / Path(self.spec_results_dir())
        self._model_save_path = self._dir_result / Path(self.model_save_dir())

        self._dirs["results"] = self._dir_result
        self._dirs["logs"] = self._dir_result / Path(self.metrics_dir())
        self._dirs["metric"] = self._dir_result / Path(self.metrics_dir())
        self._dirs["metric_batch"] = self._dir_result / Path(self.metrics_dir())
        self._dirs["time_trace"] = self._dir_result / Path(self.timing_dir())
        self._dirs["figures"] = self._dir_result / Path(self.figures_dir())
        self._dirs["graphs"] = self._dir_result / Path(self.figures_dir())
        self._dirs["prediction"] = self._dir_result / Path(self.figures_dir())

        # default dir where we store serialized prediction graph as image
        #  self._dir_model_prediction = self._dir_result / Path(self.prediction_dir())

        self.filename = None

        self._active_model = self.config['use_model']
        self._active_setting = self.config['active_setting']

        self._models = self.config['models']
        if self._active_model not in self._models:
            raise Exception("config.yaml doesn't contain model {}.".format(self._active_model))

        self._model = self._models[self._active_model]
        _settings = self.config['settings']
        self._setting = _settings[self._active_setting].copy()

        self.generate_file_name_template()

    def generate_file_name_template(self):
        """
         Generates file name templates.
        """
        if self.name_generator is not None:
            self.filename = self.name_generator()
        else:
            self.filename = self._active_model

        self.filename = "{}_{}".format(self._active_model, self.config['active_setting'])

    def normalize_dir(self, dir_str):
        """

        :param dir_str:
        :return:
        """
        if len(dir_str) > 0 and dir_str[0] == '/':
            p = Path(dir_str)
            if p.exists():
                return str(Path.resolve(dir_str))

        return dir_str

    def spec_results_dir(self) -> str:
        """
        Return main directory where all results stored.
        """
        if self.config is not None and 'results_dir' in self.config:
            return self.normalize_dir(self.config['results_dir'])

        return 'results'

    def spec_log_dir(self) -> str:
        """
        Return directory that used to store logs.
        """
        if self.config is not None and 'log_dir' in self.config:
            return self.normalize_dir(self.config['log_dir'])

        return 'logs'

    def graph_dir(self) -> str:
        """
        Return directory where store original graphs
        """
        if self.config is not None and 'graph_dir' in self.config:
            return self.normalize_dir(self.config['graph_dir'])

        return 'graphs'

    def timing_dir(self) -> str:
        """
        Return directory we use to store metrics dir traces
        """
        if self.config is not None and 'timing_dir' in self.config:
            return self.normalize_dir(self.config['timing_dir'])

        return 'timing'

    def metrics_dir(self) -> str:
        """
        Return directory we use to store time traces
        """
        if self.config is not None and 'metrics_dir' in self.config:
            self.normalize_dir(self.config['metrics_dir'])

        return 'metrics'

    def model_save_dir(self) -> str:
        """
        Default directory where model checkpoint stored.
        """
        if self.config is not None and 'model_save_dir' in self.config:
            return self.config['model_save_dir']

        return 'model'

    def prediction_dir(self) -> str:
        """
        Default directory where model prediction serialized.
        """
        if self.config is not None and 'figures_prediction_dir' in self.config:
            return self.config['figures_prediction_dir']

        return 'prediction'

    def prediction_figure_dir(self) -> str:
        """
        Default directory where model prediction serialized.
        """
        if self.config is not None and 'figures_prediction_dir' in self.config:
            return self.config['prediction_figures']
        return 'prediction'

    def figures_dir(self) -> str:
        """
        Default directory where test figures serialized.
        """
        if self.config is not None and 'figures_dir' in self.config:
            return self.config['figures_dir']
        return 'figures'

    def build_dir(self):
        """
        Creates all directories required for trainer.
        """
        if not os.path.isdir(self._dir_result):
            os.makedirs(self._model_save_path)

        if not os.path.isdir(self._model_save_path):
            os.makedirs(self._model_save_path)

        for k in self._dirs:
            if not os.path.isdir(self._dirs[k]):
                os.makedirs(self._dirs[k])

    def get_dirs(self):
        """

        :return:
        """
        return self._dirs

    def make_file_dict(self, target_dir, file_ext=None, filter_dict=None):
        """
        Recursively walk and build a dict where key is file name,
        value is dict that store path and metadata.

        :param filter_dict:
        :param file_ext:
        :param target_dir:
        :return:
        """
        target_files = {}
        for dir_path, dir_names, filenames in os.walk(target_dir):
            logger.debug("Dir walker {}", dir_names)
            for a_file in filenames:
                # additional callback if a caller need filter.
                if self.dir_walker_callback is not None:
                    if not self.dir_walker_callback(dir_path, dir_names, filenames):
                        continue

                if file_ext is not None and a_file.find(file_ext) != -1:
                    if filter_dict is None:
                        target_files[a_file] = {'path': join(dir_path, a_file), 'meta': '', 'label': '0'}
                    else:
                        if a_file in filter_dict:
                            target_files[a_file] = {'path': join(dir_path, a_file), 'meta': '', 'label': '0'}

        return target_files

    def get_model_log_file_path(self, file_type="log"):
        """Return log dir
        """
        return str(self._dirs["logs"] / self.file_name_generator(suffix="metric", file_type=file_type))

    def get_model_log_dir(self, file_type="log"):
        """Return log dir
        """
        if self.config is not None and 'graph_dir' in self.config:
            return self.normalize_dir(self.config['log_dir'])

        return 'log_dir'

    def file_name_generator(self, suffix=None, file_type='dat'):
        """Default filename generator.
        :param suffix:
        :param file_type:
        :return:
        """
        batch_size = 0
        if 'batch_size' in self._setting:
            batch_size = int(self._setting['batch_size'])

        # time_fmt = strftime("%Y-%m-%d-%H", gmtime())

        if suffix is None:
            return f"{self.filename}_batch_{str(batch_size)}_epoch_{self.load_epoch()}.{file_type}"

        return f"{self.filename}_batch_{str(batch_size)}_epoch_{self.load_epoch()}_{suffix}.{file_type}"

    def get_metric_file_path(self, file_type="npy"):
        """Return full path to a metric path.
        :param file_type:
        :return:
        """
        return str(self._dirs["metric"] / self.file_name_generator(suffix="metric", file_type=file_type))

    def get_metric_batch_file_path(self, file_type="npy"):
        """Return full path to a metric path.
        :param file_type:
        :return:
        """
        return str(self._dirs["metric_batch"] / self.file_name_generator(suffix="metric_batch", file_type=file_type))

    def get_time_file_path(self, file_type="npy"):
        """Return full path to a metric path.
        :param file_type:
        :return:
        """
        return str(self._dirs["time_trace"] / self.file_name_generator(suffix="time_trace", file_type=file_type))

    def get_model_file_path(self, model_name, file_type='dat'):
        """
        Returns dict that hold sub-model name and respected checkpoint filename.
        Args:
            model_name:
            file_type:

        Returns:
        """
        for k in self._model:
            if k == model_name:
                return str(self._model_save_path /
                           Path(f"{model_name}_{self.file_name_generator(file_type=file_type)}"))

    def model_filenames(self, file_type='dat'):
        """

        Returns dict that hold sub-model name and
        respected checkpoint filename.

        @param file_type:
        @return:
        """
        models_filenames = {}
        for k in self._model:
            models_filenames[k] = str(
                self._model_save_path / Path(f"{k}_{self.file_name_generator(file_type=file_type)}"))

        return models_filenames

    def load_epoch(self) -> int:
        """
        Setting dictates whether load model or not.
        """
        return int(self.config['load_epoch'])

    def is_trained(self) -> bool:
        """
        Return true if model trained, it mainly checks if dat file created or not.

        :return: True if trainer
        """
        models_filenames = self.model_filenames()
        if self._verbose:
            logger.info("Model files {}", models_filenames)

        for k in models_filenames:
            if not os.path.isfile(models_filenames[k]):
                return False

        return True

    def get_last_saved_epoc(self):
        """
         Return last checkpoint saved as dict where key sub-model: last checkpoint
         If model un trained will raise exception
        """
        checkpoints = {}
        if self._verbose:
            logger.debug("Trying load models last checkpoint for {}".format(self._active_model))

        if not self.is_trained():
            raise Exception("Untrained model")

        models_filenames = self.model_filenames()
        if self._verbose:
            logger.debug("Model filename {}".format(models_filenames))

        for m in models_filenames:
            if self._verbose:
                logger.debug("Trying to load checkpoint file  from {}".format(models_filenames[m]))

            check = torch.load(models_filenames[m])
            if self._verbose:
                logger.debug("Model keys {}".format(check.keys()))

            if 'epoch' in check:
                checkpoints[m] = check['epoch']

        return checkpoints

    def get_results_dir(self) -> pathlib.PosixPath:
        """

        :return:
        """
        if 'results' in self._dirs:
            return self._dirs["results"]

        return "results"

    def get_figure_dir(self) -> pathlib.PosixPath:
        """

        :return:
        """
        if 'figures' in self._dirs:
            return self._dirs["figures"]
        return "figures"

    @staticmethod
    def set_logger(is_enable: bool) -> None:
        """

        :param is_enable:
        :return:
        """
        if is_enable:
            logger.enable(__name__)
        else:
            logger.disable(__name__)
