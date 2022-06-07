import time

import librosa
import torch
from torch import nn
from torch.nn import functional as F

from model_loader.stft_dataloader import SFTFDataloader
from model_trainer.trainer_specs import ExperimentSpecs
from models.lbfs_mel_Inverse import DTCInverseSTFS
from models.torch_mel_inverse import LibrosaInverseMelScale


def _get_ratio(mat):
    return (mat.sum() / mat.numel()).item()


def inverse_test(dataset_name=""):
    """

    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    model_spec = trainer_spec.get_model_spec().get_spec('spectrogram_layer')
    pk_dataset = trainer_spec.get_audio_dataset(dataset_name)
    assert 'train_set' in pk_dataset
    assert 'validation_set' in pk_dataset
    assert 'test_set' in pk_dataset

    dataloader = SFTFDataloader(trainer_spec, batch_size=2, verbose=True)
    loaders, collate = dataloader.get_loader()

    # get all
    data_loaders, collate_fn = dataloader.get_all()
    _train_loader = data_loaders['train_set']
    print(f"Train datasize {dataloader.get_train_dataset_size()}")

    n_fft = 1024
    n_stft = n_fft // 2 + 1

    inverse_mel = LibrosaInverseMelScale(n_stft, f_max=8000.0)
    for bidx, batch in enumerate(_train_loader):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, stft_padded = batch

        epsilon = 1e-60

        start_time = time.time()
        torch_mel_inverse = inverse_mel(mel_padded)

        print(f"MEL shape         {mel_padded.shape},  "
              f"output lengths    {output_lengths}  "
              f"stft target shape {stft_padded.shape}  "
              f"inverse shape     {torch_mel_inverse.shape}")
        print(f"Train datasize {dataloader.get_train_dataset_size()}")
        print("--- %s load time, seconds ---" % (time.time() - start_time))

        torch_mel_inverse = F.pad(torch_mel_inverse, (1, 1), "constant", 0)
        mel_padded = F.pad(mel_padded, (1, 1), "constant", 0)
        torch_inverse_mel_padded = inverse_mel(mel_padded)

        # diff between torch mel and original STFT
        torch_relative_diff = torch.abs((torch_mel_inverse - stft_padded) / (torch_mel_inverse + epsilon))
        relative_diff_padded = torch.abs(
                (torch_inverse_mel_padded - stft_padded) / (torch_inverse_mel_padded + epsilon))
        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(f"Ratio of relative diff smaller than {tol:e} is " f"{_get_ratio(torch_relative_diff < tol)}")
        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(f"Ratio of relative diff padded than {tol:e} is " f"{_get_ratio(relative_diff_padded < tol)}")
        print("-------------------------------------------------------------------------------------------------")

        # librosa version numpy
        librosa_inverse = librosa.feature.inverse.mel_to_stft(mel_padded.detach().cpu().numpy(), n_fft=1024, sr=22050)
        librosa_inverse = torch.from_numpy(librosa_inverse)
        diff_librosa = torch.abs((librosa_inverse - stft_padded.numpy()) / (librosa_inverse + epsilon))
        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(f"Ratio of relative diff librosa than {tol:e} is " f"{_get_ratio(diff_librosa < tol)}")

        print("-------------------------------------------------------------------------------------------------")
        # mine
        librosa_module = DTCInverseSTFS(1024 // 2 + 1, f_max=8000.0)
        x = librosa_module(mel_padded)
        assert x.shape == stft_padded.shape == librosa_inverse.shape

        my_inverse = torch.abs((x - stft_padded) / (x + epsilon))
        for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
            print(f"Ratio of relative diff my impl {tol:e} is " f"{_get_ratio(my_inverse < tol)}")

        print("my inverse dtype", my_inverse.dtype)

        # assert _get_ratio(my_inverse < 1e-1) > 0.2
        # assert _get_ratio(my_inverse < 1e-3) > 5e-3
        # assert _get_ratio(my_inverse < 1e-5) > 1e-5

        # spec = inverse_mel(mel_padded)
        break

    # # example = train_dataset[1]
    # mse = torch.square(melspec - melspec_librosa).mean().item()
    # print("Mean Square Difference: ", mse)


def compute_epsilon_deltas(computed_inverse):
    for tol in [1e-1, 1e-3, 1e-5, 1e-10]:
        print(f"Ratio of relative diff my impl {tol:e} is " f"{_get_ratio(computed_inverse < tol)}")


def inverse_test_combine_error(dataset_name="", epsilon=1e-60, max_iteration=100, verbose=False):
    """
    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config='../config.yaml')
    pk_dataset = trainer_spec.get_audio_dataset(dataset_name)
    assert 'train_set' in pk_dataset
    assert 'validation_set' in pk_dataset
    assert 'test_set' in pk_dataset

    dataloader = SFTFDataloader(trainer_spec, batch_size=2, verbose=True)
    loaders, collate = dataloader.get_loader()

    # get all
    data_loaders, collate_fn = dataloader.get_all()
    _train_loader = data_loaders['train_set']
    print(f"Train datasize {dataloader.get_train_dataset_size()}")

    n_fft = 1024
    n_stft = n_fft // 2 + 1

    inverse_mel = LibrosaInverseMelScale(n_stft, f_max=8000.0)

    torch_abs_error = 0
    mine_abs_error = 0
    librosa_abs_error = 0
    torch_abs_error_padded = 0
    for bidx, batch in enumerate(_train_loader):
        if bidx == max_iteration:
            break

        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, stft_padded = batch

        # start_time = time.time()
        torch_mel_inverse = inverse_mel(mel_padded)
        torch_mel_inverse = F.pad(torch_mel_inverse, (1, 1), "constant", 0)
        mel_padded = F.pad(mel_padded, (1, 1), "constant", 0)
        torch_inverse_mel_padded = inverse_mel(mel_padded)

        # diff between torch mel and original STFT
        torch_relative_diff = torch.abs((torch_mel_inverse - stft_padded) / (torch_mel_inverse + epsilon))
        relative_diff_padded = torch.abs(
                (torch_inverse_mel_padded - stft_padded) / (torch_inverse_mel_padded + epsilon))
        if verbose:
            compute_epsilon_deltas(torch_relative_diff)
            compute_epsilon_deltas(relative_diff_padded)

        torch_abs_error += nn.L1Loss()(torch_mel_inverse, stft_padded).item()
        torch_abs_error_padded += nn.L1Loss()(torch_inverse_mel_padded, stft_padded).item()

        # librosa version numpy
        librosa_inverse = torch.from_numpy(
                librosa.feature.inverse.mel_to_stft(
                        mel_padded.detach().cpu().numpy(), n_fft=1024, sr=22050))
        diff_librosa = torch.abs((librosa_inverse - stft_padded.numpy()) / (librosa_inverse + epsilon))
        if verbose:
            compute_epsilon_deltas(diff_librosa)
        librosa_abs_error += nn.L1Loss()(librosa_inverse, stft_padded).item()

        # mine
        librosa_module = DTCInverseSTFS(1024 // 2 + 1, f_max=8000.0)
        x = librosa_module(mel_padded)
        assert x.shape == stft_padded.shape == librosa_inverse.shape
        if verbose:
            compute_epsilon_deltas(x)
        mine_abs_error += nn.L1Loss()(x, stft_padded).item()

    print(f"torch {torch_abs_error}, torch padded {torch_abs_error_padded}, "
          f"librosa {librosa_abs_error}, dtc {mine_abs_error}")


def inverse_test_gpu(dataset_name="", config='config.yaml',
                     epsilon=1e-60, max_iteration=100,
                     batch_size=1, verbose=False):
    """
    :return:
    """
    trainer_spec = ExperimentSpecs(spec_config=config)
    pk_dataset = trainer_spec.get_audio_dataset(dataset_name)
    assert 'train_set' in pk_dataset
    assert 'validation_set' in pk_dataset
    assert 'test_set' in pk_dataset

    dataloader = SFTFDataloader(trainer_spec, batch_size=batch_size, verbose=True)
    data_loaders, collate_fn = dataloader.get_all()
    _train_loader = data_loaders['train_set']

    n_stft = 1024 // 2 + 1
    dts_inverse = DTCInverseSTFS(n_stft, f_max=8000.0).to("cuda")

    abs_error = 0
    start_time = time.time()
    for bidx, batch in enumerate(_train_loader):
        if bidx == max_iteration:
            break
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, stft_padded = batch

        # mine
        mel_padded = mel_padded.contiguous().cuda(non_blocking=True)
        stft_padded = stft_padded.contiguous().cuda(non_blocking=True)
        mel_padded = F.pad(mel_padded, (1, 1), "constant", 0)
        x = dts_inverse(mel_padded)
        abs_error += nn.L1Loss()(x, stft_padded).item()

    print("--- %s load time, seconds ---" % (time.time() - start_time))
    print(f"torch {abs_error}")


if __name__ == '__main__':
    """
    """
    # test_download()
    # test_create_from_numpy_in_memory()
    # test_create_from_numpy_and_iterator()
    # inverse_test('lj_speech_1k_raw')
    # inverse_test_combine_error('lj_speech_1k_raw')

   # inverse_test_gpu('lj_speech_1k_raw')
    inverse_test_gpu('LJSpeech')
