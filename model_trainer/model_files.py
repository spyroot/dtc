import os
import pathlib
from os.path import join
from pathlib import Path
from typing import Optional, Callable

import torch
from loguru import logger


class ModelFileError(Exception):
    """Base class for other exceptions"""
    pass


class ModelFiles:
    """

    """

    def __init__(self, config, parent, root_dir: Optional[str] = ".",
                 name_generator: Optional[Callable] = None,
                 dir_walker_callback: Optional[Callable] = None,
                 verbose: Optional[bool] = False):
        """

        :param config:
        :param root_dir:
        :param name_generator:
        :param dir_walker_callback:
        :param verbose:
        """
        self._loaded_model = None
        self._verbose = verbose
        self.set_logger(verbose)
        self.parent = parent

        if config is None:
            raise ModelFileError("config argument is none.")

        self._config = config
        self.name_generator = name_generator
        self.dir_walker_callback = dir_walker_callback

        self.root_dir = root_dir
        if root_dir == ".":
            self.root_dir = Path(".").resolve()
        if root_dir.startswith("~"):
            self.root_dir = Path(root_dir).expanduser()
            self.root_dir = self.root_dir.resolve()

        self._dir_input = self.root_dir
        # all dirs
        self._dirs = {}
        self._dir_result = Path(self._dir_input) / Path(self.spec_results_dir())
        self._model_save_path = self._dir_result / Path(self.model_save_dir())

        self._dirs["results"] = self._dir_result
        self._dirs["model"] = self._dir_result / Path(self.model_save_dir())
        self._dirs["logs"] = self._dir_result / Path(self.metrics_dir())
        self._dirs["metric"] = self._dir_result / Path(self.metrics_dir())
        self._dirs["metric_batch"] = self._dir_result / Path(self.metrics_dir())
        self._dirs["time_trace"] = self._dir_result / Path(self.timing_dir())
        self._dirs["figures"] = self._dir_result / Path(self.figures_dir())
        self._dirs["graphs"] = self._dir_result / Path(self.figures_dir())
        self._dirs["prediction"] = self._dir_result / Path(self.figures_dir())
        self._dirs["tuner"] = self._dir_result / Path(self.tuner_dir())
        self._dirs["tuner_logs"] = self._dir_result / Path(self.tuner_log_dir())
        self._dirs["tuner_safe_dir"] = self._dir_result / Path(self.tuner_safe_dir())

        # default dir where we store serialized prediction graph as image
        # self._dir_model_prediction = self._dir_result / Path(self.prediction_dir())

        self.filename = None

        self._active_model = self._config['use_model']
        self._active_setting = self._config['active_setting']

        self._models = self._config['models']
        if len(self._models.keys()) == 0:
            raise ModelFileError("No model defined in specification.")

        if self._active_model not in self._models:
            raise ModelFileError(f"config.yaml doesn't contain model configuration {self._active_model}.")

        self._model = self._models[self._active_model]
        _settings = self._config['settings']
        self._setting = _settings[self._active_setting]

        # generate all names
        self.generate_file_name_template()

    def generate_file_name_template(self):
        """
         Generates file name templates.
        """
        if self.name_generator is not None:
            self.filename = self.name_generator()
        else:
            self.filename = self._active_model

        self.filename = "{}_{}".format(self._active_model, self._config['active_setting'])

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
        :return: Return main directory where all results are stored.
        """
        if 'results_dir' in self._config:
            return self.normalize_dir(self._config['results_dir'])
        return 'results'

    def spec_log_dir(self) -> str:
        """
        :return: Return directory that used to store logs.
        """
        if 'log_dir' in self._config:
            return self.normalize_dir(self._config['log_dir'])

        return 'logs'

    def graph_dir(self) -> str:
        """
        Return directory where store original graphs
        """
        if 'graph_dir' in self._config:
            return self.normalize_dir(self._config['graph_dir'])

        return 'graphs'

    def timing_dir(self) -> str:
        """
        Return directory we use to store metrics dir traces
        """
        if 'timing_dir' in self._config:
            return self.normalize_dir(self._config['timing_dir'])

        return 'timing'

    def metrics_dir(self) -> str:
        """
        Return directory we use to store time traces
        """
        if 'metrics_dir' in self._config:
            self.normalize_dir(self._config['metrics_dir'])

        return 'metrics'

    def model_save_dir(self) -> str:
        """
        Default directory where model checkpoint stored.
        """
        if 'model_save_dir' in self._config:
            return self._config['model_save_dir']

        return 'model'

    def prediction_dir(self) -> str:
        """
        Default directory where model prediction serialized.
        """
        if self._config is not None and 'figures_prediction_dir' in self._config:
            return self._config['figures_prediction_dir']

        return 'prediction'

    def prediction_figure_dir(self) -> str:
        """
        Default directory where model prediction serialized.
        """
        if self._config is not None and 'figures_prediction_dir' in self._config:
            return self._config['prediction_figures']
        return 'prediction'

    def figures_dir(self) -> str:
        """
        Default directory where test figures serialized.
        """
        if self._config is not None and 'figures_dir' in self._config:
            return self._config['figures_dir']
        return 'figures'

    def build_dir(self):
        """
        Creates all directories required for trainer.
        """
        if not os.path.isdir(self._dir_result):
            os.makedirs(self._model_save_path)
            if not Path(self._model_save_path).is_dir():
                raise ModelFileError(f"Failed created a directory {str(self._model_save_path)}")

        if not os.path.isdir(self._model_save_path):
            os.makedirs(self._model_save_path)
            if not Path(self._model_save_path).is_dir():
                raise ModelFileError(f"Failed created a directory {str(self._model_save_path)}")

        for k in self._dirs:
            if not os.path.isdir(self._dirs[k]):
                os.makedirs(self._dirs[k])
                if not Path(self._dirs[k]).is_dir():
                    raise ModelFileError(f"Failed created a directory {str(self._model_save_path)}")

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

    def get_model_log_file_path(self, file_type="log", remove_old=False):
        """Return path to a log file.
        TODO merge the code
        """
        file_path = str(self._dirs["logs"] / self.file_name_generator(suffix="metric", file_type=file_type))
        if remove_old and Path(file_path).exists():
            try:
                    os.remove(file_path)
            except PermissionError as e:
                print(f'Failed to delete a file permission error. {file_path}, err:{e}')
            except Exception as e:
                print(f'Failed to delete a file. {file_path}, err:{e}')
        return str(self._dirs["logs"] / self.file_name_generator(suffix="metric", file_type=file_type))

    def get_trace_log_file(self, trace_name: str, file_type="log", remove_old=False):
        """Return full path for a trace log file, trace_name a name appended to a file name.
           used to register different log traces.
        :param trace_name:
        :param file_type:
        :param remove_old: in case we want remove it first.
        :return:
        """
        file_path = str(self._dirs["logs"] / self.file_name_generator(suffix=trace_name, file_type=file_type))
        if remove_old and Path(file_path).exists():
            try:
                os.remove(file_path)
            except PermissionError as e:
                print(f'Failed to delete a file permission error. {file_path}, err:{e}')
            except Exception as e:
                print(f'Failed to delete a file. {file_path}, err:{e}')
        return file_path

    def get_model_log_dir(self, file_type="log"):
        """Return log dir
        """
        if self._config is not None and 'log_dir' in self._config:
            return self.normalize_dir(self._config['log_dir'])

        return 'log_dir'

    def file_name_generator(self, suffix=None, file_type='dat'):
        """
        Default filename generator.

        :param suffix:
        :param file_type:
        :return:
        """
        batch_size = self.parent.batch_size()
        if suffix is None:
            return f"{self.filename}_batch_{str(batch_size)}_epoch_{self.load_epoch()}.{file_type}"

        return f"{self.filename}_batch_{str(batch_size)}_epoch_{self.load_epoch()}_{suffix}.{file_type}"

    def get_metric_file_path(self, file_type="npy"):
        """Return full path to a metric path.
        :param file_type:
        :return:
        """
        return str(self._dirs["metric"] / self.file_name_generator(suffix="step_metric", file_type=file_type))

    def get_metric_batch_file_path(self, file_type="npy"):
        """Return full path to a metric path.
        :param file_type:
        :return:
        """
        return str(self._dirs["metric_batch"] / self.file_name_generator(suffix="metric_batch", file_type=file_type))

    def get_metric_dir(self):
        """
        Return dir where model metric must be stored.
        :return:
        """
        return str(self._dirs["metric"])

    def get_time_file_path(self, file_type="npy"):
        """Return full path to a metric path.
        :param file_type:
        :return:
        """
        return str(self._dirs["time_trace"] / self.file_name_generator(suffix="time_trace", file_type=file_type))

    def get_model_file_path(self, model_layer_name: str, file_type='dat') -> str:
        """
        Method return model file name, the name generated based specification.
        and based on template format.

        :param model_layer_name: model specification in config spec.
        :param file_type:
        :return:
        """
        if model_layer_name is None or len(model_layer_name) == 0:
            raise ValueError("Model layer is empty.")

        if self._loaded_model is not None:
            return self._loaded_model

        for k in self._model:
            if k == model_layer_name:
                return str(self._model_save_path /
                           Path(f"{model_layer_name}_{self.file_name_generator(file_type=file_type)}"))

        return str(self._model_save_path / Path("default.dat"))

    def get_model_dir(self) -> str:
        """
        :return: Return dir where model checkpoints must be stored.

        """
        return str(self._dirs["model"])

    def get_all_model_filenames(self, file_type='dat'):
        """

        Returns dict that hold sub-model name and
        respected checkpoint filename.

        @param file_type:
        @return:
        """
        models_filenames = {}
        for k in self._model:
            models_filenames[k] = str(self._model_save_path /
                                      Path(f"{k}_{self.file_name_generator(file_type=file_type)}"))

        return models_filenames

    def load_epoch(self) -> int:
        """
        Setting dictates whether load model or not.
        """
        return int(self._config['load_epoch'])

    def is_trained(self) -> bool:
        """
        Return true if model trained, it mainly checks if dat file created or not.

        :return: True if trainer
        """
        models_filenames = self.get_all_model_filenames()
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

        models_filenames = self.get_all_model_filenames()
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

    def get_results_dir(self) -> Path:
        """

        :return:
        """
        if 'results' in self._dirs:
            return self._dirs["results"]
        return Path("results").resolve()

    def get_figure_dir(self) -> Path:
        """
        :return:
        """
        if 'figures' in self._dirs:
            return self._dirs["figures"]
        return Path("results/figures").resolve()

    def get_tuner_dir(self) -> Path:
        """
        :return:
        """
        if 'tuner' in self._dirs:
            return self._dirs["tuner"]
        return Path("results/tuner").resolve()

    def get_tuner_log_dir(self) -> Path:
        """
        :return:
        """
        if 'tuner_logs' in self._dirs:
            return self._dirs["tuner_logs"]
        return Path("results/tuner_logs").resolve()

    def get_tuner_safe_dir(self) -> Path:
        """
        :return:
        """
        if 'tuner_logs' in self._dirs:
            return self._dirs["tuner_logs"]
        return Path("results/tuner_logs").resolve()

    @staticmethod
    def set_logger(is_enable: bool) -> None:
        """
        Switch logger on
        :param is_enable:
        :return:
        """
        if is_enable:
            logger.enable(__name__)
        else:
            logger.disable(__name__)

    def tuner_dir(self) -> str:
        """
        :return:
        """
        if self._config is not None and 'tuner' in self._config:
            return self._config['tuner']
        return 'tuner'

    def tuner_log_dir(self) -> str:
        """
        :return:
        """
        if self._config is not None and 'tuner_logs' in self._config:
            return self._config['tuner_logs']
        return 'tuner_logs'

    def tuner_safe_dir(self):
        """

        :return:
        """
        if self._config is not None and 'tuner_save_dir' in self._config:
            return self._config['tuner_save_dir']
        return 'tuner_save_dir'

    def update_model_file(self, path: str) -> bool:
        """

        :param path:
        :return:
        """
        if path is None or len(path) == 0:
            return False

        resolved = Path(path).expanduser().resolve()
        if resolved.exists():
            self._loaded_model = str(resolved)
            return True

        return False


