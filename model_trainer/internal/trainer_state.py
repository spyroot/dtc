# import torch.distributed as dist
# import queue
# from abc import ABC

# from frozendict import frozendict
# @attr.s()

# from numba import jit
# @jitclass(nopython=True)
class TrainerState:
    """

    """

    def __init__(self):
        self.cuda_device_id = 0
        self.disable_pbar = 0
        self.n_gpus = 1

        self.disable_pbar = False
        self.verbose = False
        self.is_notebook = False
        self.rank = None

        self.is_hp_tunner = False

        self.trainer_spec = None
        self.batch_size = None
        self.device = None
        self.is_distributed = False
        self.is_amp = True

        self.current_model = None
        self.current_layer = None

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
        self.current_model = None
        self.current_optimizer = None
        self.current_schedulers = None

        # self.set_logger(verbose)
        #
        # # device
        # self.device = device
        #
        # #
        # self.disable_pbar = disable_pbar
        #
        # # dict that hold all schedulers that trainer need to use. TODO This will be moved.
        # self._schedulers = {}
        # # dict hold all optimizers. ( note trainer can train 2 model like in gan settings)
        # self._optimizers = {}
        #
        # if not is_inference:
        #     if not trainer_spec.is_initialized():
        #         raise TrainerError("you need initialize trainer specs first.")
        #
        # self._tkey = self.trainer_spec.get_default_train_set_key()
        # self._vkey = self.trainer_spec.get_default_val_set_key()
        #
        # # if not self.is_inference:
        # #     if data_loader is None:
        # #         raise TrainerError("Trainer need torch data loader.")
        # #     self._data_loaders, self.collate_fn = data_loader.get_all()
        # #     self._train_loader = self._data_loaders[self._tkey]
        # #     self._validation_loader = self._data_loaders[self._vkey]
        # #
        # #     if len(self._train_loader.dataset) == 0:
        # #         warnings.warn("Training dataset empty")
        # #     if len(self._validation_loader.dataset) == 0:
        # #         warnings.warn("Training dataset empty")
        #
        # # TODO need refactor that, and move to dict and abstract,
        # self.criterion = Tacotron2Loss()
        # # dict store all model
        # self._models = {}
        # # store last epoch
        #
        # self._last_ckt_epochs: dict[str, dict[str, int]] = {}
        # self._steps: dict[str, int] = {}  # dict holds model name = last iterator value
        # self.scaler = None
        # # self.tqdm_iter = None  # tqdm_iter, if we need fix post of iter
        #
        # self.epoch = None  # current epoch trainer executing.
        # self.saved_run = None  # last saved run
        #
        # self.clip_grad = False
        # self.total_batches = 0
        # if self.is_inference is False:
        #     # total batches
        #     self.total_batches = len(self._data_loaders[self._tkey])
        #     # clip or not grad
        #     self.clip_grad = trainer_spec.is_grad_clipped()

#
# @attr.s(frozen=True)
# class FrozenTrainerState(TrainerState, ABC):
#     def __init__(self,
#                  verbose: Optional[bool] = False,
#                  is_notebook: Optional[bool] = False,
#                  rank: Optional[int] = 0,
#                  world_size: Optional[int] = 2,
#                  disable_pbar: Optional[int] = False,
#                  device: Optional[int] = torch.device,
#                  cuda_device_id: Optional[int] = 0,
#                  is_inference: Optional[bool] = False,
#                  callback: Optional[list[Callback]] = None,
#                  hp_tunner=False,
#                  config=None,
#                  checkpoint_dir=None) -> None:
#         """
#
#         :param verbose:       enabled verbose output
#         :param is_notebook:   if trainer run it notebook mode. ( tqdm and pbar need to re-adjusted)
#         :param rank:          node rank
#         :param world_size:    world_size
#         :param disable_pbar:  disabled pbar
#         :param device:        device we run
#         :param is_inference:  inference mode or not
#         """
#         super(FrozenTrainerState, self).__init__(verbose=verbose,
#                                                  is_notebook=is_notebook,
#                                                  rank=rank,
#                                                  world_size=world_size,
#                                                  disable_pbar=disable_pbar,
#                                                  device=device,
#                                                  cuda_device_id=cuda_device_id,
#                                                  is_inference=is_inference)
