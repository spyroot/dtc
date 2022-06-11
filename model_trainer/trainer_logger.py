import os
import random
from typing import Optional
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model_trainer.plotting_utils import plot_alignment_to_numpy
from model_trainer.plotting_utils import plot_spectrogram_to_numpy
from model_trainer.plotting_utils import plot_gate_outputs_to_numpy
from model_trainer.trainer_specs import ExperimentSpecs
import time


class TensorboardTrainerLogger(SummaryWriter):
    """
    """

    def __init__(self, trainer_spec: ExperimentSpecs,
                 model_name: Optional[str] = "dtc",
                 batch_size: Optional[int] = 32,
                 precision: Optional[str] = "fp32",
                 comments="", logdir=None, is_distributed=False):
        """
        :param logdir:
        :param is_distributed:
        """
        super(TensorboardTrainerLogger, self).__init__(f"results/tensorboard/{model_name}/{batch_size}/{precision}",
                                                       comment="dtc",
                                                       filename_suffix="dtc",
                                                       flush_secs=2)

        self.is_stft_loss = None
        self.trainer_spec = trainer_spec
        self.model_spec = trainer_spec.get_model_spec()
        self.spectogram_spec = self.model_spec.get_spectrogram()
        self.update_rate = trainer_spec.tensorboard_update_rate()
        self.spectrogram_spec = trainer_spec.get_model_spec().get_spectrogram()
        self.is_reverse_decoder = self.spectrogram_spec.is_reverse_decoder()
        self.run_name = f"{model_name}/{batch_size}/{precision}"

    def add_hparams_and_step(self, hparam_dict, metric_dict, hparam_domain_discrete=None,
                             run_name=None, global_step=None, epoch=None):
        """
         This a fix for tensorboard util to include proper step.
         Add a set of hyperparameters to be compared in TensorBoard.

        Args:

            hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
              contains names of the hyperparameters and all discrete values they can hold
            run_name (str): Name of the run, to be included as part of the logdir.
              If unspecified, will use current timestamp.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

        Expected result:

        .. image:: _static/img/tensorboard/add_hparam.png
           :param epoch:        If specified instead of time, run = epoch
           :param hparam_dict:  Each key-value pair in the dictionary is the name of the
                                hyperparameter, and it's corresponding value.
                                The type of the value can be one of `bool`, `string`, `float`,
                                `int`, or `None`.
           :param metric_dict:  (dict): Each key-value pair in the dictionary is the
                                name of the metric, and it's corresponding value. Note that the key used
                                here should be unique in the tensorboard record. Otherwise, the value
                                you added by ``add_scalar`` will be displayed in hparam plugin. In most
                                cases, this is unwanted.
           :param run_name:     overwrite default run
           :param global_step:
           :param hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
                                contains names of the hyperparameters
                                and all discrete values they can hold
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = torch.utils.tensorboard.summary.hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        if not run_name:
            if epoch is not None:
                run_name = epoch
            else:
                run_name = str(time.time())
        logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                if global_step is not None:
                    w_hp.add_scalar(k, v, global_step=global_step)
                else:
                    w_hp.add_scalar(k, v)

    def log_training(self, criterions: dict, step=None, lr=None, hparams=None, metrics=None, extra_data=None) -> None:
        """
        Log trainer result to tensorboard.
        :param metrics: metric all term attach to hparams.
        :param criterions: criterions all loss term in dict.
        :param step:  current step of in training loop
        :param hparams:  dict host hparams.
        :param lr: learning rate
        :param extra_data:  extra data key value pair
        :return: 
        """
        if self.update_rate == 0 or step % self.update_rate != 0:
            return

        # make sure key not overlap with validation.
        for k in criterions:
            self.add_scalar(k, criterions[k], global_step=step)

        self.add_scalar("learning.rate", lr, global_step=step)

        if hparams is not None and metrics is not None:
            self.add_hparams_and_step(hparams, metrics, run_name=self.run_name, global_step=step)

        if extra_data is not None:
            for k in extra_data:
                self.add_scalar(k, extra_data[k], global_step=step)

    def log_hparams(self, step, hp_dict, metrics) -> None:
        """
        Log tf hp params.

        :param metrics:  hyperparameter metrics
        :param hp_dict: hyperparameter dict
        :param step: current step of execution
        :return:
        """
        if self.update_rate == 0 or step % self.update_rate != 0:
            return
        self.add_hparams(hp_dict, metrics)

    def log_validation(self, criterions: dict, model: nn.Module, y, y_pred, step=None,
                       mel_filter=True, v3=True) -> None:
        """
        Log validation step, each step we serialize prediction spectrogram loss counter
        from a criterions' dict.

        :param criterions: dict that must hold all loss metric.
        :param model: nn.Module
        :param y:
        :param y_pred:
        :param step: current execution step.
        :param mel_filter:
        :param v3: For v3 DTC model we report STFT
        :return:
        """
        if self.update_rate == 0 or step % self.update_rate != 0:
            return

        alignments_rev = None
        gate_out_rev = None

        for k in criterions:
            self.add_scalar(k, criterions[k], step)

        # self.add_scalar("loss/validation", loss, step)

        if self.is_reverse_decoder:
            _, mel_outputs, gate_outputs, alignments, rev = y_pred
            mel_out_rev, gate_out_rev, alignments_rev = rev
            mel_targets, gate_targets, = y[0], y[1]
        else:
            _, mel_outputs, gate_outputs, alignments = y_pred
            mel_targets, gate_targets = y[0], y[1]

        if self.spectogram_spec.is_stft_loss_enabled():
            stft = y[2]

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), step)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)

        self.add_image(
                "alignment/alignments_left",
                plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
                step, dataformats='HWC')

        if self.is_reverse_decoder and alignments_rev is not None:
            self.add_image(
                    "alignment/alignments_right",
                    plot_alignment_to_numpy(alignments_rev[idx].data.cpu().numpy().T),
                    step, dataformats='HWC')

        self.add_image(
                "mel/mel_target",
                plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
                step, dataformats='HWC')

        self.add_image(
                "mel/mel_predicted",
                plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
                step, dataformats='HWC')

        # if v3:
        #     self.add_image(
        #             plot_spectrogram(stft[idx], title="Spectrogram", ylabel="freq", file_name=None),
        #             step, dataformats='HWC')

        # self.add_image(
        #         "mel/stft",
        #         plot_spectrogram_to_numpy(stft[idx].data.cpu().numpy()), step, dataformats='HWC')

        # self.add_image(
        #         "stft",
        #         plot_sft(stft[idx].data.cpu().numpy()), step, dataformats='HWC')

        self.add_image(
                "gate/gate",
                plot_gate_outputs_to_numpy(
                        gate_targets[idx].data.cpu().numpy(),
                        torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
                step, dataformats='HWC')

        if self.is_reverse_decoder and gate_out_rev is not None:
            self.add_image(
                    "gate/gate_rev",
                    plot_gate_outputs_to_numpy(
                            gate_targets[idx].data.cpu().numpy(),
                            torch.sigmoid(gate_out_rev[idx]).data.cpu().numpy()),
                    step, dataformats='HWC')
