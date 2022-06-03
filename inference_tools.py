import os
import librosa
import torchaudio
from PIL._imaging import display
from matplotlib import pyplot as plt
import torch
# from IPython.display import Audio, display
from playsound import playsound
from IPython.display import Audio, display

from model_trainer.plotting_utils import save_figure_to_numpy


def play_audio_file(file_path, is_notebook=False, sample_rate=22050):
    """

    :param waveform:
    :param is_notebook:
    :param sample_rate:
    :return:
    """
    # waveform = waveform.numpy(
    playsound(file_path)


def play_audio(waveform, is_notebook=False, sample_rate=22050):
    """

    :param waveform:
    :param sample_rate:
    :return:
    """
    waveform = waveform.numpy()
    playsound('/path/to/a/sound/file/you/want/to/play.wav')

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")


def show_img(img, title=""):
    """

    :param img:
    :param title:
    :return:
    """
    plt.imshow(img)
    plt.title(title)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+100+100")
    plt.show()


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None, file_name="default.png"):
    """

    :param file_name:
    :param spec:
    :param title:
    :param ylabel:
    :param aspect:
    :param xmax:
    :return:
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

    if len(file_name) > 0:
        plt.savefig(file_name)
        plt.close()
        return

    data = save_figure_to_numpy(fig)
    plt.close()

    return data


def plot_mel_fbank(fbank, title=None):
    """

    :param fbank:
    :param title:
    :return:
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Filter bank')
    axs.imshow(fbank, aspect='auto')
    axs.set_ylabel('frequency bin')
    axs.set_xlabel('mel bin')
    plt.show(block=False)


def plot_waveform(waveform, sample_rate=22050, title="Spectrogram",
                  xlim=None, file_name="default_spect.png"):
    """
    :param file_name:
    :param waveform:
    :param sample_rate:
    :param title:
    :param xlim:
    :return:
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)

    figure.suptitle(title)
    plt.show(block=False)
    if len(file_name) > 0:
        plt.savefig(file_name)
        plt.close()
        return

    data = save_figure_to_numpy(figure)
    plt.close()
    return data


def plot_pitch(waveform, pitch, title="Pitch", sample_rate=22050, file_name="default_pitch.png"):
    """

    :param title:
    :param waveform:
    :param pitch:
    :param sample_rate:
    :param file_name:
    :return:
    """
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    axis2.legend(loc=0)
    plt.show(block=False)

    figure.suptitle(title)
    plt.show(block=False)
    if len(file_name) > 0:
        plt.savefig(file_name)
        plt.close()
        return

    data = save_figure_to_numpy(figure)
    plt.close()
    return data
