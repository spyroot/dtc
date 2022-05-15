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
        print("self.total_iteration", self.total_iteration)
        print("self.num_batches", self.num_batches)

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

    def total_mean_loss(self):
        """Return mean loss, compute mean from entire loss history
        :return:
        """
        if self.loss is None:
            return 0.0

        return self.loss.mean(0)[-1]
