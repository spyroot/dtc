# import torch.distributed as dist
# import queue
# from abc import ABC

# from frozendict import frozendict
# @attr.s()

# from numba import jit
# @jitclass(nopython=True)
from typing import Optional

import torch.nn


class TrainerState:
    """

    """

    def __init__(self):

        self.cuda_device_id = 0

        self.disable_pbar = 0

        self.n_gpus = 1

        self.disable_pbar = False

        self.is_notebook = False

        self.verbose = False

        self.rank = None

        self.is_hp_tunner = False

        self.trainer_spec = None

        self.batch_size = None

        self.device = None

        self.is_distributed = False

        self.is_amp = True

        self.current_model_name: Optional[str] = None

        self.current_layer_name: Optional[str] = None

        # current epoch trainer executing.
        self.epoch = None

        self.step = None

        # last saved run
        self.saved_run = None

        self.data_loaders = None

        self.collate_fn = None

        # update rate for tqdm
        self.tbar_update_rate = 0

        # each de-queue set current model , opt , scheduelr
        self.current_model: Optional[torch.nn.Module] = None
        self.current_optimizer = None
        self.current_schedulers = None

        self.grade_accumulate = 0
