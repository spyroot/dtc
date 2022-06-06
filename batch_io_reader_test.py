import time
import torch
from model_trainer.utils import to_gpu
from tqdm import tqdm
from model_loader.stft_dataloader import SFTFDataloader
from model_trainer.trainer_specs import ExperimentSpecs


def batch_reader(batch, device, version=3):
    """
    Batch parser for dtc.
    :param version:
    :param device:
    :param batch:
    :return:
    """
    if version == 3:
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, stft = batch
        text_padded = to_gpu(text_padded, device).long()
        input_lengths = to_gpu(input_lengths, device).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded, device).float()
        gate_padded = to_gpu(gate_padded, device).float()
        output_lengths = to_gpu(output_lengths, device).long()

        sf = stft.contiguous()
        if torch.cuda.is_available():
            sf = sf.cuda(non_blocking=True)
            sf.requires_grad = False

        return (text_padded, input_lengths,
                mel_padded, max_len,
                output_lengths, stft), \
               (mel_padded, gate_padded, stft)
    else:
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded, device).long()
        input_lengths = to_gpu(input_lengths, device).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded, device).float()
        gate_padded = to_gpu(gate_padded, device).float()
        output_lengths = to_gpu(output_lengths, device).long()

        return (text_padded, input_lengths,
                mel_padded, max_len,
                output_lengths), \
               (mel_padded, gate_padded)


def v3_dataloader_audio_test(config="config.yaml"):
    """

    :return:
    """
    spec = ExperimentSpecs(spec_config=config)
    start_time = time.time()
    dataloader = SFTFDataloader(spec, verbose=True)
    print("--- %s SFTFDataloader create batch , load time, seconds ---" % (time.time() - start_time))
    _device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # get all
    data_loaders, collate_fn = dataloader.get_all()
    _train_loader = data_loaders['train_set']

    iters = dataloader.get_train_dataset_size() // dataloader.get_batch_size()
    print("Total iters", iters)

    # full GPU pass
    start_time = time.time()
    for batch_idx, (batch) in tqdm(enumerate(_train_loader), total=iters):
        x, y = batch_reader(batch, device=_device, version=3)
    print("--- %s SFTFDataloader entire dataset pass, load time, seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    """
    """
    v3_dataloader_audio_test()
