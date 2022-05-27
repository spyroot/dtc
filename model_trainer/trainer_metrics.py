from pathlib import Path
from typing import Optional

import numpy as np
from timeit import default_timer as timer
from loguru import logger

from model_loader.mel_dataloader import SFTFDataloader
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
        self.set_logger(verbose)
        self.loss = None
        self.total_loss = None
        self.num_epochs = num_epochs
        self.grad_norm_loss = None

        self.total_iteration = None
        self.num_batches = num_batches
        self.num_iteration = num_iteration
        self.epoch_timer = None

        # file to save and load metrics
        self.metric_perf_trace_path = metric_step_file_path
        self.metric_batch_file_path = metric_batch_file_path
        self.metric_step_file_path = metric_perf_trace_path

    def update(self, batch_idx, step, loss: float, grad_norm=None):
        """Update metric history.
        :param batch_idx: - batch idx used to index to internal id
        :param step: - current execution step.
        :param loss: - loss
        :param grad_norm: - grad norm loss, in case we track both loss and grad norm loss after clipping.
        :return: nothing11
        """
        self.loss[step] = loss
        self.total_loss[batch_idx] += loss
        if grad_norm is not None:
            self.grad_norm_loss[step] = grad_norm
            logger.info("Batch {} Step {} Loss {} mean {} grad norm loss {}",
                        batch_idx, step, loss, self.loss.mean(), grad_norm)
        else:
            logger.info("Batch {} Step {} Loss {} mean {}",
                        batch_idx, step, loss, self.loss.mean())

    def set_num_iteration(self, num_iteration):
        """Update number of total iterations
        :param num_iteration: - should total iteration
        :return: nothing
        """
        self.num_iteration = max(1, num_iteration)

    def set_num_batches(self, num_batches):
        """Updates number of total batches
        :param num_batches: - should total batches
        :return: nothing
        """
        self.num_batches = max(1, num_batches)

    # def update_total_loss(self):
    #     """
    #
    #     :return:
    #     """
    #     self.total_iteration = max(1, self.num_batches) * max(1, self.num_iteration)
    #     self.total_loss = np.zeros((self.num_batches, 1))
    #     self.loss = np.zeros((self.total_iteration, 1))
    #     return

    def init(self):
        """

        :return:
        """
        self.total_iteration = max(1, self.num_batches) * max(1, self.num_iteration)
        if self.num_batches > 0 and self.num_iteration > 0 and self.total_iteration > 0:
            logger.info("Creating metric data. {} ".format(self.num_batches))
            self.total_loss = np.zeros((self.num_batches, 1))
            self.loss = np.zeros((self.total_iteration, 1))
            self.grad_norm_loss = np.zeros((self.total_iteration, 1))
            self.epoch_timer = np.zeros((self.num_epochs, 1))

        logger.info("Metric started, expected num batches {}".format(self.num_batches))
        logger.info("Metric expected iter per step {} and {} total iteration".format(self.num_iteration,
                                                                                     self.total_iteration))
        logger.debug("Metric shapes loss {} total_loss {} delta timer {}",
                     self.loss.shape,
                     self.total_loss.shape,
                     self.epoch_timer.shape)

        return

    def start_epoch_timer(self, epoch_idx):
        """Start epoch timer and save start time.
        :param epoch_idx:
        :return:
        """
        self.epoch_timer[epoch_idx] = timer()

    def update_epoch_timer(self, epoch_idx):
        """Update epoch timer, and update time trace.
        :param epoch_idx:
        :return:
        """
        self.epoch_timer[epoch_idx] = timer() - max(0, self.epoch_timer[epoch_idx])
        logger.info("Timer {} average {}", self.epoch_timer[epoch_idx], self.epoch_timer.mean(0)[-1])

    def save(self):
        """Method saves all metrics
        :return:
        """
        if self.metric_step_file_path is not None:
            np.save(str(self.metric_step_file_path.resolve()), self.loss)

        if self.metric_batch_file_path is not None:
            np.save(str(self.metric_batch_file_path.resolve()), self.total_loss)

        if self.metric_perf_trace_path is not None:
            np.save(str(self.metric_perf_trace_path.resolve()), self.epoch_timer)

    def load(self):
        """Method loads all metric traces.
        :return:
        """
        if self.metric_step_file_path is not None:
            if not isinstance(self.metric_step_file_path, str):
                self.loss = np.load(str(self.metric_step_file_path.resolve()))
            else:
                self.loss = np.load(self.metric_step_file_path)

        if self.metric_batch_file_path is not None:
            if not isinstance(self.metric_batch_file_path, str):
                self.total_loss = np.load(str(self.metric_batch_file_path.resolve()))
            else:

                self.total_loss = np.load(self.metric_batch_file_path)

        if self.metric_perf_trace_path is not None:
            if not isinstance(self.metric_batch_file_path, str):
                self.epoch_timer = np.load(str(self.metric_perf_trace_path.resolve()))
            else:
                self.epoch_timer = np.load(self.metric_perf_trace_path)

    def total_mean_loss(self):
        """Return mean loss, compute mean from entire loss history
        :return:
        """
        if self.loss is None:
            return 0.0

        return self.loss.mean(0)[-1]

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


def batch_loss_compute():
    """

    :return:
    """
    spec = ExperimentSpecs(spec_config='../config.yaml')
    loader = SFTFDataloader(spec, verbose=False)
    loaders, colleter= loader.get_loader()

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
