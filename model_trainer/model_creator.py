# This unfinshed.  All generic factory creator will moved here.

# from typing import Callable
#
# from .trainer_specs import ExperimentSpecs
# from .model_trainer import GeneratorTrainer
# from models.loss_function import Tacotron2Loss
# from models.dtc_loss_function import dtcLoss
#
# class ModelCreator:
#     """
#      Create model and model trainer.
#     """
#
#     def __init__(self, trainer_spec: ExperimentSpecs,
#                  device='cuda', debug=False, verbose=False):
#         """
#          Construct model creator,  it takes trainer spec
#          and creates respected model that indicate
#          as active model in specification.
#
#         :param trainer_spec: model trainer specification
#         :param device:  device used to train a model.
#         :param verbose:  output verbose data during model creation
#         """
#         if trainer_spec is None:
#             raise Exception("Model spec is nil")
#
#         self.verbose = verbose
#         self.device = device
#         self.debug = debug
#         self.model_dispatch, self.trainer_dispatch = self.create_model_dispatch()
#         self.trainer_spec = trainer_spec
#         self.create_model_dispatch()
#
#     def __call__(self):
#         """
#         :return:
#         """
#         if self.trainer_spec is None:
#             raise Exception("Model spec is nil")
#
#         self.model_dispatch, self.trainer_dispatch = self.create_model_dispatch()
#         self.create_model_dispatch()
#
#     def create_model_dispatch(self) -> tuple[dict[str, dict[str: Callable]],
#                                              dict[str, Callable],
#                                              dict[str, dict[str: Callable]]]:
#         """
#         Create two dispatcher,
#         Model dispatcher and trainer dispatcher.
#         The first dispatch model creator, where each key model name
#         as it defined in config.yaml and value is callable creator function,
#
#         method, class etc.
#         Similarly, second dispatch is trainer callable objects.
#
#         :return: model creator Callable , and trainer creator Callable.
#         """
#         model_dispatch = {
#             'tacotron25': {
#                 'spectrogram_layer': self.create_tacotron25,
#                 'vocoder': self.create_tacotron25,
#                 'loss': Tacotron2Loss,
#             },
#             'dtc': {
#                 'spectrogram_layer': self.create_tacotron30,
#                 'vocoder': self.create_tacotron25,
#                 'loss':
#             }
#         }
#
#         trainer_dispatch = {
#             # 'GraphGRU': RnnGenerator,
#             # 'GraphLSTM': RnnGenerator,
#         }
#
#         batch_loader = {
#             'tacotron25': {
#                 'spectrogram_layer': self.tacotron25_batch,
#             },
#             'dtc': {
#                 'spectrogram_layer': self.tacotron30_batch,
#             }
#         }
#
#         return model_dispatch, trainer_dispatch, batch_loader
#
#     def create_lstm_rnn(self, trainer_spec: ExperimentSpecs):
#         """
#         Factory method create model based https://arxiv.org/abs/1802.08773
#         Instead we use LSTM
#         GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models
#         """
#         models = {}
#         # node_rnn = GraphLSTM(self, input_size=trainer_spec.depth(),
#         #                      embedding_size=trainer_spec.embedding_size_rnn,
#         #                      hidden_size=trainer_spec.hidden_size_rnn,
#         #                      has_input=True,
#         #                      has_output=True,
#         #                      num_layers=trainer_spec.num_layers(),
#         #                      device=self.device).to(self.device)
#         #
#         # edge_rnn = GraphLSTM(input_size=1,
#         #                      embedding_size=trainer_spec.embedding_size_rnn_output,
#         #                      hidden_size=trainer_spec.hidden_size_rnn_output,
#         #                      num_layers=trainer_spec.num_layers(),
#         #                      has_input=True,
#         #                      has_output=True,
#         #                      output_size=1,
#         #                      device=self.device).to(self.device)
#
#         # models['node_model'] = node_rnn
#         # models['edge_model'] = edge_rnn
#
#         return models
#
#     def create_gan(self, trainer_spec: ExperimentSpecs):
#         models = {}
#
#         generator = Generator(H_inputs=trainer_spec.H_inp(),
#                               H=trainer_spec.H_gen(),
#                               N=trainer_spec.N(),
#                               rw_len=trainer_spec.rw_len(), z_dim=trainer_spec.z_dim(),
#                               temp=trainer_spec.temp_start()).to(self.device)
#
#         discriminator = Discriminator(H_inputs=trainer_spec.H_inp(),
#                                       H=trainer_spec.H_disc(),
#                                       N=trainer_spec.N(),
#                                       rw_len=trainer_spec.rw_len()).to(self.device)
#
#         models['generator'] = generator
#         models['discriminator'] = discriminator
#
#     def create_gru_rnn(self, trainer_spec: ExperimentSpecs):
#         """
#         Create model based https://arxiv.org/abs/1802.08773
#         GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models
#         """
#         models = {}
#         node_rnn = GraphGRU(input_size=trainer_spec.max_depth(),
#                             batch_size=trainer_spec.batch_size(),
#                             embedding_size=trainer_spec.embedding_size_rnn,
#                             hidden_size=trainer_spec.hidden_size_rnn,
#                             num_layers=trainer_spec.num_layers(),
#                             has_input=True,
#                             has_output=True,
#                             output_size=trainer_spec.hidden_size_rnn_output,
#                             device=self.device).to(self.device)
#
#         edge_rnn = GraphGRU(input_size=1, batch_size=trainer_spec.batch_size(),
#                             embedding_size=trainer_spec.embedding_size_rnn_output,
#                             hidden_size=trainer_spec.hidden_size_rnn_output,
#                             num_layers=trainer_spec.num_layers(),
#                             has_input=True,
#                             has_output=True,
#                             output_size=1,
#                             device=self.device).to(self.device)
#
#         models['node_model'] = node_rnn
#         models['edge_model'] = edge_rnn
#
#         return models
#
#     def create_model(self, verbose=False):
#         """
#         Creates a model and return model name as key and value of dict a model.
#         :param verbose: if set True will output verbose data
#         :return:  return a dict with model and model trainer
#         """
#         sub_models = self.trainer_spec.get_model_submodels()
#
#         if self.debug:
#             fmtl_print("Model settings", self.trainer_spec.model)
#             fmtl_print("Model settings", self.trainer_spec.lr_schedulers)
#
#         if self.trainer_spec.is_model_verbose() or verbose:
#             print("")
#             fmtl_print("Model sub-models", sub_models)
#             for sub in sub_models:
#                 scheduler_alias_name = self.trainer_spec.get_model_lr_scheduler(sub)
#                 opt_alias_name = self.trainer_spec.get_model_optimizer(sub)
#
#                 fmtl_print("Model {} lr scheduler".format(sub), scheduler_alias_name)
#                 fmtl_print("Model {} scheduler type".format(sub),
#                            self.trainer_spec.lr_scheduler_type(scheduler_alias_name))
#                 fmtl_print("Model {} optimizer name".format(sub), self.trainer_spec.get_model_optimizer(sub))
#                 fmtl_print("Model {} optimizer spec".format(sub), self.trainer_spec.get_optimizer_type(opt_alias_name))
#
#         # case when we have two model
#         if 'node_model' in self.trainer_spec.model and 'edge_model' in self.trainer_spec.model:
#             model_attr = self.trainer_spec.model['node_model']
#             model_name = model_attr['model']
#         else:
#             model_attr = self.trainer_spec.model['single_model']
#             model_name = model_attr['model']
#
#         # get factory
#         model_factory = self.model_dispatch[model_name]
#         models_dict = model_factory(self.trainer_spec)
#
#         if self.debug:
#             print(models_dict)
#
#         return models_dict
#
#     def create_model_dispatch(self):
#         """
#         Create two dispatcher,
#         Model dispatcher and trainer dispatcher.
#         The first dispatch model creator, where each key model name
#         as it defined in config.yaml and value is callable creator function,
#
#         method, class etc.
#         Similarly, second dispatch is trainer callable objects.
#         :return:
#         """
#         model_dispatch = {
#             'tacotron25': self.create_tacotron25,
#             'dtc': self.create_tacotron30,
#         }
#
#         trainer_dispatch = {
#             # 'GraphGRU': RnnGenerator,
#             # 'GraphLSTM': RnnGenerator,
#         }
#
#         return model_dispatch, trainer_dispatch
#
#     def create_trainer(self, models: dict, dataset_loader, decoder=None) -> GeneratorTrainer:
#         """
#          A factory method,  create trainer based on spec and model and return to caller.
#         :param models:
#         :param dataset_loader:
#         :param decoder:
#         :return:
#         """
#         model_attr = self.trainer_spec.model['node_model']
#         model_name = model_attr['model']
#         generator = self.trainer_dispatch[model_name]
#         return generator(self.trainer_spec, models,
#                          dataset_loader,
#                          decoder,
#                          device=self.device, verbose=self.verbose)
