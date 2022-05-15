from tacotron2.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy
import torch
import random
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardTrainerLogger(SummaryWriter):
    """

    """
    def __init__(self, tensorboard_update_rate=0, logdir=None, is_distributed=False):
        super(TensorboardTrainerLogger, self).__init__()
        self.update_rate = tensorboard_update_rate

    def log_training(self, step, reduced_loss, grad_norm, learning_rate, hparams=None):
        """

        Args:
            reduced_loss:
            grad_norm:
            learning_rate:
            step:

        Returns:
        :param hparams:

        """
        if self.update_rate == 0:
            return

        if step % self.update_rate != 0:
            return

        self.add_scalar("training.loss", reduced_loss, step)
        self.add_scalar("grad.norm", grad_norm, step)
        self.add_scalar("learning.rate", learning_rate, step)
        self.add_hparams(hparams)
        self.flush()

    def log_hparams(self, step, tf_hp_dict):
        """

        :param step:
        :param dict:
        :return:
        """
        if self.update_rate == 0:
            return

        if step % self.update_rate != 0:
            return

        self.add_hparams(tf_hp_dict)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        """

        :param reduced_loss:
        :param model:
        :param y:
        :param y_pred:
        :param iteration:
        :return:
        """
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.flush()

