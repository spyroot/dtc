import pickle
from abc import abstractmethod, ABCMeta

import os
import socket
from abc import ABC
from typing import Optional

import torch
# import torch.distributed as dist

from loguru import logger
import torch.distributed as dist

from model_trainer.internal.trainer_state import TrainerState
from model_trainer.trainer_specs import ExperimentSpecs
# from distributed import apply_gradient_allreduce
from model_trainer.utils import fmt_print


# from torch.nn.parallel import DistributedDataParallel
# from torch.autograd import Variable
# import numpy as np

class TrainerError(Exception):
    pass


class AbstractTrainer(ABC, metaclass=ABCMeta):
    """

    """

    @abstractmethod
    def __init__(self,
                 trainer_spec: ExperimentSpecs,
                 data_loader=None,
                 verbose: Optional[bool] = False,
                 is_notebook: Optional[bool] = False,
                 rank: Optional[int] = 0,
                 world_size: Optional[int] = 2,
                 disable_pbar: Optional[int] = False,
                 device: Optional[int] = torch.device,
                 cuda_device_id: Optional[int] = 0,
                 is_inference: Optional[bool] = False):

        # self.state = None
        self.state = TrainerState()
        self.state.cuda_device_id = cuda_device_id
        self.state.disable_pbar = disable_pbar
        self.state.device = device
        self.state.rank = rank
        self.state.world_size = world_size
        self.state.disable_pbar = disable_pbar
        self.state.is_notebook = is_notebook
        self.state.trainer_spec = trainer_spec
        self.state.verbose = verbose
        self.data_loader = data_loader
        self.state.n_gpus = 1

        print(f"Data initialized {rank} {device} {cuda_device_id}")
        if trainer_spec is None:
            raise TrainerError("Trainer specification is None")

        if not is_inference:
            logger.info("Trainer created, active model {}", trainer_spec.get_active_model())

        # inference or training
        self.state.is_inference = is_inference


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load(self, model_name: str, ignore_layers):
        pass

    @staticmethod
    def split_tensor(tensor, n_gpus):
        """
        "
        Args:
            tensor:
            n_gpus:

        Returns:

        """
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= n_gpus
        return rt

    @staticmethod
    def save_graphs(g, file_name, verbose=False):

        if file_name is None:
            raise Exception("File name is none.")

        if len(file_name) == 0:
            raise Exception("empty file name")

        if verbose:
            fmt_print("Saving graph to a file ", file_name)

        with open(file_name, "wb") as f:
            pickle.dump(g, f)

    @staticmethod
    def compute_average_loss(self):
        pass


    def set_notebook(self, param):
        """
        Update trainer and set it in notebook mode
        :param param:
        :return:
        """
        self.is_notebook = param

    def set_verbose(self, param):
        """
        Set verbose level
        :param param:
        :return:
        """
        self.verbose = param

    def _loop_up_device(self, is_set_cuda: bool):
        """
        This mainly fix for some unknown torch issue related how it checks device.
        :param is_set_cuda:
        :return:
        """
        if torch.cuda.is_available():
            n = torch.cuda.device_count() // self.state.n_gpus

        if is_set_cuda:
            device = f"cuda:{dist.get_rank()}"
            logger.info("Number gpu on the node {} device".format(n, device))
            torch.cuda.set_device(self.state.cuda_device_id)
        else:
            device = self.state.device

        return device

    def init_distributed(self) -> None:
        """
        Initialize DDP
        :return:
        """
        os.environ['MASTER_ADDR'] = self.state.trainer_spec.get_master_address()
        os.environ['MASTER_PORT'] = self.state.trainer_spec.get_master_port()
        # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"

        assert torch.cuda.is_available(), "Distributed mode requires CUDA."
        logger.info("Distributed Available".format(torch.cuda.device_count()))
        logger.info("Distribute protocol nccl available {}".format(torch.distributed.is_nccl_available()))
        logger.info("Distribute protocol mpi available {}".format(torch.distributed.is_mpi_available()))
        logger.info("Distribute protocol glow available {}".format(torch.distributed.is_gloo_available()))
        logger.info("Distribute endpoint {} my rank {}".format(self.state.trainer_spec.get_backend(), self.state.rank))

        # Set cuda device so everything is done on the right GPU.
        # torch.cuda.set_device(self.rank % torch.cuda.device_count())
        logger.info("Set cuda device".format(self.state.rank % torch.cuda.device_count()))
        # Initialize distributed communication
        if self.state.rank == 0:
            host = socket.gethostname()
            address = socket.gethostbyname(host)
            logger.info("resolve hostname {}".format(host))
            logger.info("resolve hostname {}".format(address))

        torch.distributed.init_process_group(
                backend=self.state.trainer_spec.get_backend(),
                init_method=self.state.trainer_spec.dist_url(),
                world_size=self.state.n_gpus,
                rank=self.state.rank)

        logger.debug("Done initializing distributed {}".format(dist.get_rank()))

