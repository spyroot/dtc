import numpy as np
from timeit import default_timer as timer
from loguru import logger


class Metrics:
    """

    """

    def __init__(self,
                 metric_step_file_path=None,
                 metric_batch_file_path=None,
                 metric_perf_trace_path=None,
                 num_epochs=0,
                 num_batches=0,
                 num_iteration=0):
        """

        :param num_batches:
        """

        self.loss = None
        self.total_loss = None
        self.num_epochs = num_epochs

        self.total_iteration = None
        self.num_batches = num_batches
        self.num_iteration = num_iteration
        self.epoch_timer = None

        # file to save and load metrics
        self.metric_perf_trace_path = metric_step_file_path
        self.metric_batch_file_path = metric_batch_file_path
        self.metric_step_file_path = metric_perf_trace_path

    def update(self, batch_idx, step, loss):
        """

        :param batch_idx:
        :param step:
        :param loss:
        :return:
        """
        self.loss[step] = loss
        self.total_loss[batch_idx] += loss
        logger.info("Batch {} Step {} Loss {} mean {}", batch_idx, step, loss, self.loss.mean())

    def set_num_iteration(self, num_iteration):
        """

        :param num_iteration:
        :return:
        """
        self.num_iteration = max(1, num_iteration)

    def set_num_batches(self, num_batches):
        """

        :param num_batches:
        :return:
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
            logger.info("Creating metric data.")
            self.total_loss = np.zeros((self.num_batches, 1))
            self.loss = np.zeros((self.total_iteration, 1))
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
            np.save(self.metric_step_file_path, self.loss)

        if self.metric_batch_file_path is not None:
            np.save(self.metric_batch_file_path, self.total_loss)

        if self.metric_perf_trace_path is not None:
            np.save(self.metric_perf_trace_path, self.epoch_timer)

    def load(self):
        """Method loads all metric traces.
        :return:
        """
        if self.metric_step_file_path is not None:
            self.loss = np.load(self.metric_step_file_path)

        if self.metric_batch_file_path is not None:
            self.total_loss = np.load(self.metric_batch_file_path)

        if self.metric_perf_trace_path is not None:
            self.epoch_timer = np.save(self.metric_perf_trace_path, self.epoch_timer)
