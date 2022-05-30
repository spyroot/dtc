from model_trainer.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy
import random
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn


class TensorboardTrainerLogger(SummaryWriter):
    """

    """
    def __init__(self, tensorboard_update_rate=0,  comments="", logdir=None, is_distributed=False):
        """

        :param tensorboard_update_rate:
        :param logdir:
        :param is_distributed:
        """
        super(TensorboardTrainerLogger, self).__init__("results/tensorboard", comment="dts", flush_secs=2)
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

        print(f"logging step {step} {criterions.keys()} {self.update_rate}")
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

    def log_hparams(self, step, tf_hp_dict) -> None:
        """
        Log tf hp params.

        :param tf_hp_dict: hyperparameter dict
        :param step:
        :return:
        """
        if self.update_rate == 0 or step % self.update_rate != 0:
            return
        self.add_hparams(tf_hp_dict)

    def log_validation(self, loss, model: nn.Module, y, y_pred, step=None, mel_filter=True) -> None:
        """
        Log validation step.
        :param loss:
        :param model:
        :param y:
        :param y_pred:
        :param step:
        :param mel_filter:
        :return:
        """
        self.add_scalar("loss/validation", loss, step)

        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), step)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        # if mel_filter:
        # plot_filter_bank()
        # img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)

        self.add_image(
                "alignment",
                plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
                step, dataformats='HWC')
        self.add_image(
                "mel_target",
                plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
                step, dataformats='HWC')
        self.add_image(
                "mel_predicted",
                plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
                step, dataformats='HWC')
        self.add_image(
                "gate",
                plot_gate_outputs_to_numpy(
                        gate_targets[idx].data.cpu().numpy(),
                        torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
                step, dataformats='HWC')
