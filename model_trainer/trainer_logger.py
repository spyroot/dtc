from tacotron2.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy
import torch
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import librosa


class TensorboardTrainerLogger(SummaryWriter):
    """

    """
    def __init__(self, tensorboard_update_rate=0, logdir=None, is_distributed=False):
        super(TensorboardTrainerLogger, self).__init__()
        self.update_rate = tensorboard_update_rate
        # self.mel_filters_librosa = librosa.filters.mel(
        #         sr=sampling_rate,
        #         n_fft=n_fft,
        #         fmin=mel_fmin,
        #         fmax=mel_fmax,
        #         norm="slaney",
        #         htk=True,
        # ).T

    def log_training(self, step, cluster_loss, grad_norm, lr, hparams=None, extra_data=None) -> None:
        """

        :param step:  current step of in training loop
        :param cluster_loss:  for distributed loss,
        :param grad_norm:  normalized gradient loss
        :param lr: learning rate
        :param hparams:  dict host hparams.
        :param extra_data:  extra data key value pair
        :return: 
        """
        if self.update_rate == 0:
            return

        if step % self.update_rate != 0:
            return

        self.add_scalar("training.loss", cluster_loss, step)
        self.add_scalar("grad.norm", grad_norm, step)
        self.add_scalar("learning.rate", lr, step)
        if hparams is not None:
            self.add_hparams(hparams)
        if extra_data is not None:
            for k in extra_data:
                self.add_scalar(k, extra_data[k])

        self.flush()

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

    def log_validation(self, reduced_loss, model, y, y_pred, iteration, mel_filter=True) -> None:
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
        # if mel_filter:
            # plot_filter_bank()
            # img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)

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
