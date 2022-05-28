from pathlib import Path
from typing import Optional

import numpy as np
from timeit import default_timer as timer
from loguru import logger

from model_loader.stft_dataloader import SFTFDataloader
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
        self.grad_norm_val_loss = None
        self.batch_grad_val_loss = None
        self.batch_val_loss = None
        self.val_loss = None
        Metrics.set_logger(verbose)
        # epoch loss
        self.loss = None
        self.grad_norm_loss = None

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
        self.batch_val_loss = np.zeros((self.num_batches, 1))
        self.batch_grad_val_loss = np.zeros((self.num_batches, 1))

    def on_prediction_batch_end(self):
        logger.info(f"validation batch loss {self.batch_val_loss.sum():.2f} | {self.batch_grad_val_loss.sum():.2f}, "
                    f"mean {self.batch_val_loss.mean():.4f} | {self.batch_grad_val_loss.mean():.4f}")
        self.val_loss[self._epoc_counter] = self.batch_val_loss.mean()
        self.grad_norm_val_loss[self._epoc_counter] = self.batch_grad_val_loss.mean()

    def on_prediction_epoch_start(self):
        pass

    def on_batch_start(self):
        # self.batch_loss = np.zeros((self.num_batches, 1))
        # self.batch_grad_loss = np.zeros((self.num_batches, 1))
        self.batch_loss = np.zeros((self.num_batches, 1))
        self.batch_grad_loss = np.zeros((self.num_batches, 1))

    def on_batch_end(self):
        logger.info(f"train batch loss {self.batch_loss.sum():.2f} | {self.batch_grad_loss.sum():.2f}, "
                    f"mean {self.batch_loss.mean():.4f} | {self.batch_grad_loss.mean():.4f}")

        self.loss[self._epoc_counter] = self.batch_loss.mean()
        self.grad_norm_loss[self._epoc_counter] = self.batch_grad_loss.mean()

    def on_epoch_begin(self):
        self.on_prediction_epoch_start()
        # logger.info(f"Epoch {self._epoc_counter}/{self.num_epochs} started.")

    def on_prediction_epoch_end(self):
        logger.info(f"Epoch validation loss {self._epoc_counter}/{self.num_epochs} epoch "
                    f"{self.val_loss.sum():.2f} | {self.grad_norm_val_loss.sum():.2f}, "
                    f"mean {self.val_loss.mean():.4f} | {self.grad_norm_val_loss.mean():.4f}")

    def on_epoch_end(self):
        self.on_prediction_epoch_end()
        logger.info(f"Epoch train loss {self._epoc_counter}/{self.num_epochs} epoch "
                    f"{self.loss.sum():.2f} | {self.grad_norm_loss.sum():.2f}, "
                    f"mean {self.loss.mean():.4f} | {self.grad_norm_loss.mean():.4f}")
        self._epoc_counter = self._epoc_counter + 1

    def on_begin(self):
        self.loss = np.zeros((self.num_epochs, 1))
        self.grad_norm_loss = np.zeros((self.num_epochs, 1))
        self.val_loss = np.zeros((self.num_epochs, 1))
        self.grad_norm_val_loss = np.zeros((self.num_epochs, 1))

    def on_end(self):
        pass

    def update(self, batch_idx, step, loss: float, grad_norm=None, validation=True):
        """Update metric history.
        :param validation:
        :param batch_idx: - batch idx used to index to internal id
        :param step: - current execution step.
        :param loss: - loss
        :param grad_norm: - grad norm loss, in case we track both loss and grad norm loss after clipping.
        :return: nothing11
        """
        if validation:
            #self.val_loss[step] = loss
            self.batch_val_loss[batch_idx] = loss
            if grad_norm is not None:
                self.batch_grad_val_loss[batch_idx] = loss
               # self.grad_norm_val_loss[step] = loss
            if grad_norm is not None:
                #self.grad_norm_val_loss[step] = grad_norm
                logger.info(
                        f"validation step {step} batch {batch_idx} loss {loss:.5f} "
                        f"mean {self.loss.mean():.5f} grad norm loss {grad_norm:.5f}")
            else:
                logger.info(f"validation step {step} batch {batch_idx} loss {loss:.5f} mean {self.loss.mean()}:5f")

            return

       # self.loss[step] = loss
        self.batch_loss[batch_idx] = loss
        if grad_norm is not None:
            self.batch_grad_loss[batch_idx] = loss
           # self.grad_norm_loss[step] = loss

        # print(batch_idx, self.batch_loss)
        if grad_norm is not None:
            self.grad_norm_loss[step] = grad_norm
            logger.info(f"Step {step} batch {batch_idx} loss {loss:.5f} "
                        f"mean {self.loss.mean():.5f} grad norm loss {grad_norm:.5f}")
        else:
            logger.info(f"Step {step} batch {batch_idx} loss {loss:.5f} mean {self.loss.mean()}:5f")

    def set_num_iteration(self, num_iteration):
        """Update number of total iterations
        :param num_iteration: - should be total iteration
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
            self.batch_loss = np.zeros((self.num_batches, 1))
            self.batch_grad_loss = np.zeros((self.num_batches, 1))
            self.loss = np.zeros((self.total_iteration, 1))
            self.grad_norm_loss = np.zeros((self.total_iteration, 1))
            self.epoch_timer = np.zeros((self.num_epochs, 1))

        logger.info(f"Metric started, expected num batches {self.num_batches}")
        logger.info(f"Metric expected iter per step {self.num_iteration} and {self.total_iteration} total iteration")
        logger.debug("Metric shapes loss {} total_loss {} delta timer {}",
                     self.loss.shape,
                     self.batch_loss.shape,
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
        logger.info("Saving metric files.")
        np.save(self.metric_step_file_path, self.loss)
        np.save(self.metric_batch_file_path, self.batch_loss)
        np.save(self.metric_perf_trace_path, self.epoch_timer)

    def load(self):
        """Method loads all metric traces.
        :return:
        """
        logger.info("Loading metric files.")
        self.loss = np.load(self.metric_step_file_path)
        self.batch_loss = np.load(self.metric_batch_file_path)
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
            print(f"Logging {__name__} enabled.")
            logger.enable(__name__)
        else:
            print(f"Logging {__name__} disabled.")
            logger.disable(__name__)

    @staticmethod
    def get_logger_name():
        return __name__


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
