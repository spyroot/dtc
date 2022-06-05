from pathlib import Path
from typing import Optional

import numpy as np
from timeit import default_timer as timer
from loguru import logger

from model_loader.stft_dataloader import SFTFDataloader
from model_trainer.internal.const import ReduceMode, MetricType
from .trainer_specs import ExperimentSpecs


class Metrics:
    """

    """

    def __init__(self,
                 metric_step_file_path: Optional[Path] = None,
                 metric_batch_file_path: Optional[Path] = None,
                 metric_perf_trace_path: Optional[Path] = None,
                 num_epochs=0,
                 num_batches=0,
                 num_iteration=0,
                 batch_size=0,
                 metric_type: Optional[MetricType] = "mean",
                 mode: Optional[ReduceMode] = "min",
                 verbose=False):
        """

        :param metric_step_file_path: path metrix file used to serialize per step trace
        :param metric_batch_file_path:  path to a file used to serialize batch per step
        :param metric_perf_trace_path:  path to traces
        :param num_epochs:  num total epoch
        :param num_batches: num total batches
        :param num_iteration: num total iteration
        :param verbose:  verbose or not
        """

        self.metric_type = metric_type

        if mode == ReduceMode.MIN:
            self.default_metric_val = np.inf
            self.monitor_op = np.less
        elif mode == ReduceMode.MAX:
            self.default_metric_val = -np.inf
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Unknown {mode} metric.")

        # accumulated validation loss per batch iteration.
        self.batch_val_loss = None

        # accumulated epoch validation loss per batch iteration.
        self.epoch_val_loss = None

        Metrics.set_logger(verbose)
        # epoch loss
        self.epoch_train_loss = None
        self.epoch_train_gn_loss = None

        # batch stats
        self.batch_loss = None
        self.batch_grad_loss = None

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.total_iteration = None
        self.num_batches = num_batches
        self.num_iteration = num_iteration
        self.epoch_timer = None

        # file to save and load metrics
        if not isinstance(metric_step_file_path, str):
            self.metric_step_file_path = metric_step_file_path
        self.metric_step_file_path = metric_step_file_path

        if not isinstance(metric_batch_file_path, str):
            self.metric_batch_file_path = metric_batch_file_path
        self.metric_batch_file_path = metric_batch_file_path

        if not isinstance(metric_perf_trace_path, str):
            self.metric_perf_trace_path = metric_perf_trace_path
        self.metric_perf_trace_path = metric_perf_trace_path

        self._self_metric_files = [self.metric_step_file_path,
                                   self.metric_batch_file_path,
                                   self.metric_perf_trace_path]

        self._epoc_counter = 0

    def on_prediction_batch_start(self):
        """
        Note if loader has no value , this value will be ether -inf or +inf
        :return:
        """
        self.batch_val_loss = np.ones((self.num_batches, 1)) * self.default_metric_val

    def on_prediction_batch_end(self):
        """
        called on prediction batch end , it updates an epoch validation stats.
        :return:
        """
        logger.info(f"{self.batch_val_loss.sum():5.2f} "
                    f"mean {self.batch_val_loss.mean():5.4f} | batch pred")

        if self.metric_type == MetricType.MEAN:
            self.epoch_val_loss[self._epoc_counter] = self.batch_val_loss.mean()
        if self.metric_type == MetricType.SUM:
            self.epoch_val_loss[self._epoc_counter] = self.batch_val_loss.sum()

    def on_prediction_epoch_start(self):
        pass

    def on_batch_start(self):
        """

        :return:
        """
        self.batch_loss = np.ones((self.num_batches, 1)) * self.default_metric_val
        self.batch_grad_loss = np.ones((self.num_batches, 1)) * self.default_metric_val

    def on_batch_end(self):
        """
        On train batch end we update epoch count with mean or sum
        :return:
        """
        logger.info(f"{self.batch_loss.sum():5.2f} | {self.batch_grad_loss.sum():5.2f}, "
                    f"mean {self.batch_loss.mean():5.4f} | {self.batch_grad_loss.mean():5.4f} | batch train ")

        # on batch end we update initial value.
        if self.metric_type == MetricType.MEAN:
            self.epoch_train_loss[self._epoc_counter] = self.batch_loss.mean()
            self.epoch_train_gn_loss[self._epoc_counter] = self.batch_grad_loss.mean()

        if self.metric_type == MetricType.SUM:
            self.epoch_train_loss[self._epoc_counter] = self.batch_loss.sum()
            self.epoch_train_gn_loss[self._epoc_counter] = self.batch_grad_loss.sum()

    def on_prediction_epoch_end(self):
        """
        Update metric for prediction , validation loss.
        :return:
        """
        self.update_epoch_timer()
        logger.info(f"{self.epoch_val_loss.sum():5.2f} | "
                    f"mean {self.epoch_val_loss.mean():5.4f} | epoch pred ")

    def on_epoch_begin(self):
        """
        On epoch begin , reset if needed.
        :return:
        """
        self.start_epoch_timer()
        self.on_prediction_epoch_start()

    def on_epoch_end(self):
        """
        :return:
        """
        self.on_prediction_epoch_end()
        logger.info(f"{self.epoch_train_loss.sum():5.2f} | {self.epoch_train_gn_loss.sum():5.2f}, "
                    f"mean {self.epoch_train_loss.mean():5.4f} | {self.epoch_train_gn_loss.mean():5.4f} | epoch train | "
                    f"{self.epoch_timer.mean():3.3f} | {np.average(self.epoch_timer):3.3f}")

        self._epoc_counter = self._epoc_counter + 1

    def on_begin(self):
        """
        It called when training begin.
        :return:
        """
        self.epoch_train_loss = np.zeros((self.num_epochs + 1, 1))
        self.epoch_train_gn_loss = np.zeros((self.num_epochs + 1, 1))
        self.epoch_val_loss = np.zeros((self.num_epochs + 1, 1))
        self.epoch_timer = np.zeros((self.num_epochs, 1))

    def on_end(self):
        pass

    def update(self, batch_idx, step, loss: float, grad_norm=None, validation=True):
        """
        Update metric history, each step per epoch..
        :param validation:
        :param batch_idx: - batch idx used to index to internal id
        :param step: - current execution step.
        :param loss: - loss
        :param grad_norm: - grad norm loss, in case we track both loss and grad norm loss after clipping.
        :return: nothing11
        """
        if validation:
            self.batch_val_loss[batch_idx] = loss

        self.batch_loss[batch_idx] = loss
        if grad_norm is not None:
            self.batch_grad_loss[batch_idx] = grad_norm

    def set_num_iteration(self, num_iteration):
        """
        Update number of total iterations.
        :param num_iteration: - should be total iteration
        :return: nothing
        """
        self.num_iteration = max(1, num_iteration)

    def update_bach_estimated(self, num_batches):
        """
        Updates number of total batches.  i.e if batch size reduce
        we reflect in size of matrix used to calculate batch stats.

        :param num_batches: - should total batches
        :return: nothing
        """
        self.num_batches = max(1, num_batches)

    def init(self):
        """

        :return:
        """
        self.total_iteration = max(1, self.num_batches) * max(1, self.num_iteration)
        if self.num_batches > 0 and self.num_iteration > 0:
            logger.info("Creating metric data, num batches {} ".format(self.num_batches))

            # initial value initialized either as + or - INF
            # initilized batch size of batch size.
            self.batch_loss = np.ones((self.num_batches, 1)) * self.default_metric_val
            self.batch_grad_loss = np.ones((self.num_batches, 1)) * self.default_metric_val
            # per epoch stats matrix , size of num total epochs
            self.epoch_train_loss = np.zeros((self.num_epochs, 1))
            self.epoch_train_gn_loss = np.zeros((self.num_epochs, 1))
            # timer
            self.epoch_timer = np.zeros((self.num_epochs, 1))

            logger.info(f"Metric shapes batch loss {self.batch_loss.shape[0]}")
            logger.info(f"Metric shapes batch loss {self.batch_grad_loss.shape[0]}")
            logger.info(f"Metric shapes epoch shape {self.epoch_train_loss.shape[0]}")

        return

    def start_epoch_timer(self):
        """Start epoch timer and save start time.
        :return:
        """
        self.epoch_timer[self._epoc_counter] = timer()

    def update_epoch_timer(self):
        """Update epoch timer, and update time trace.
        :return:
        """
        self.epoch_timer[self._epoc_counter] = timer() - max(0, self.epoch_timer[self._epoc_counter])
        # logger.info("Timer {} average {}", self.epoch_timer[epoch_idx], self.epoch_timer.mean(0)[-1])

    def save(self):
        """Method saves all metrics.
        :return:
        """
        logger.info("Saving metric files.")
        np.save(self.metric_step_file_path, self.epoch_train_loss)
        np.save(self.metric_batch_file_path, self.batch_loss)
        np.save(self.metric_perf_trace_path, self.epoch_timer)

    def load(self):
        """Method loads all metric traces back.
        :return:
        """
        logger.info("Loading metric files.")
        self.epoch_train_loss = np.load(self.metric_step_file_path)
        self.batch_loss = np.load(self.metric_batch_file_path)
        self.epoch_timer = np.load(self.metric_perf_trace_path)

    def total_train_mean_loss(self):
        """Return mean loss, compute mean from entire loss history.
        :return:
        """
        if self.epoch_train_loss is None:
            return self.default_metric_val

        return self.epoch_train_loss.mean()

    def epoch_average_loss(self):
        """
        Compute average loss
        :return:
        """
        return self.epoch_train_loss.mean()

    def total_prediction_mean_loss(self):
        """
        :return:
        """
        if self.epoch_train_loss is None:
            return self.default_metric_val

        return self.epoch_val_loss.mean()

    def get_metric_value(self, monitor):
        """
        Should return metric by value.
        :param monitor:
        :return:
        """
        pass

    @staticmethod
    def set_logger(is_enable: bool) -> None:
        """
        Enable logging.
        :param is_enable: if caller need enable logging.
        :return:
        """
        if is_enable:
            logger.enable(__name__)
        else:
            logger.disable(__name__)

    @staticmethod
    def get_logger_name():
        return __name__

    def update_batch_size(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        self.batch_size = batch_size



def batch_loss_compute():
    """
    :return:
    """
    spec = ExperimentSpecs(spec_config='../config.yaml')
    loader = SFTFDataloader(spec, verbose=False)
    loaders, colleter = loader.get_loader()

    total_batches = len(loaders['train_set'])

    # self.tf_logger = TensorboardTrainerLogger(trainer_spec.tensorboard_update_rate())
    metric = Metrics(metric_step_file_path=spec.model_files.get_metric_file_path(),
                     metric_batch_file_path=spec.model_files.get_metric_batch_file_path(),
                     metric_perf_trace_path=spec.model_files.get_time_file_path(),
                     num_epochs=spec.epochs(),
                     num_batches=total_batches,
                     verbose=False)

    metric.set_num_iteration(spec.epochs() * total_batches)
    metric.init()

    # metric.update(batch_idx, step, normal_loss, grad_norm=grad_norm.item())


if __name__ == '__main__':
    """
    """
    batch_loss_compute()
