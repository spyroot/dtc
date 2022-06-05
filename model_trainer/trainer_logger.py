import random

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model_trainer.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy


class TensorboardTrainerLogger(SummaryWriter):
    """
    """

    def __init__(self, tensorboard_update_rate=0, model_name="dts", batch_size=32, precision="fp32",
                 comments="", logdir=None, is_distributed=False):
        """
        :param tensorboard_update_rate:
        :param logdir:
        :param is_distributed:
        """
        super(TensorboardTrainerLogger, self).__init__(f"results/tensorboard/{model_name}/{batch_size}/{precision}",
                                                       comment="dts",
                                                       filename_suffix="dts",
                                                       flush_secs=2)
        self.update_rate = tensorboard_update_rate

    def log_training(self, criterions: dict, step, lr, hparams=None, metrics=None, extra_data=None) -> None:
        """

        :param metrics:
        :param criterions:
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
                       mel_filter=True, v3=True, is_reversed=True) -> None:
        """
        Log validation step.
        :param criterions: dict that must hold all loss metric.
        :param model: nn.Module
        :param y:
        :param y_pred:
        :param step: current execution step.
        :param mel_filter:
        :param is_reversed:   For dual decoder case we report additional metrics.
        :param v3:  For v3 DTC model we report STFT
        :return:
        """
        if self.update_rate == 0 or step % self.update_rate != 0:
            return

        alignments_rev = None
        gate_out_rev = None

        for k in criterions:
            self.add_scalar(k, criterions[k], step)

        # self.add_scalar("loss/validation", loss, step)

        if is_reversed:
            _, mel_outputs, gate_outputs, alignments, rev = y_pred
            mel_out_rev, gate_out_rev, alignments_rev = rev
            mel_targets, gate_targets, stft = y
        else:
            _, mel_outputs, gate_outputs, alignments, _ = y_pred
            mel_targets, gate_targets = y

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

        if is_reversed and alignments_rev is not None:
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
                "gate",
                plot_gate_outputs_to_numpy(
                        gate_targets[idx].data.cpu().numpy(),
                        torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
                step, dataformats='HWC')
        if is_reversed and gate_out_rev is not None:
            self.add_image(
                    "gate_rev",
                    plot_gate_outputs_to_numpy(
                            gate_targets[idx].data.cpu().numpy(),
                            torch.sigmoid(gate_out_rev[idx]).data.cpu().numpy()), step, dataformats='HWC')
