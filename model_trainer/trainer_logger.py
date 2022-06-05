import random
from typing import Optional
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model_trainer.plotting_utils import plot_alignment_to_numpy
from model_trainer.plotting_utils import plot_spectrogram_to_numpy
from model_trainer.plotting_utils import plot_gate_outputs_to_numpy
from model_trainer.trainer_specs import ExperimentSpecs


class TensorboardTrainerLogger(SummaryWriter):
    """
    """

    def __init__(self, trainer_spec: ExperimentSpecs,
                 model_name: Optional[str] = "dts",
                 batch_size: Optional[int] = 32,
                 precision: Optional[str] = "fp32",
                 comments="", logdir=None, is_distributed=False):
        """
        :param logdir:
        :param is_distributed:
        """
        super(TensorboardTrainerLogger, self).__init__(f"results/tensorboard/{model_name}/{batch_size}/{precision}",
                                                       comment="dts",
                                                       filename_suffix="dts",
                                                       flush_secs=2)

        self.is_stft_loss = None
        self.trainer_spec = trainer_spec
        self.model_spec = trainer_spec.get_model_spec()
        self.spectogram_spec = self.model_spec.get_spectrogram()
        self.update_rate = trainer_spec.tensorboard_update_rate()
        self.spectrogram_spec = trainer_spec.get_model_spec().get_spectrogram()
        self.is_reverse_decoder = self.spectrogram_spec.is_reverse_decoder()

    def log_training(self, criterions: dict, step, lr, hparams=None, metrics=None, extra_data=None) -> None:
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
            self.add_scalar(k, criterions[k], step)

        self.add_scalar("learning.rate", lr, step)

        if hparams is not None:
            self.add_hparams(hparams, metrics)

        if extra_data is not None:
            for k in extra_data:
                self.add_scalar(k, extra_data[k])

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
        Log validation step.
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
                plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()), step, dataformats='HWC')

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
                            torch.sigmoid(gate_out_rev[idx]).data.cpu().numpy()), step, dataformats='HWC')
