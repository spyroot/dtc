import argparse
from pathlib import Path

from tqdm import tqdm
import time

# for i in tqdm(range(20), desc='tqdm() Progress Bar', leave_empty=True):
#     time.sleep(0.5)
#
# training_meta: ljs_audio_text_train_filelist.txt
# validation_meta: ljs_audio_text_val_filelist.txt
# test_meta: ljs_audio_text_test_filelist.txt
from model_trainer.trainer_specs import ExperimentSpecs


def generator_read(file_name):
    file = open(file_name, 'r', encoding="utf8")
    while True:
        line = file.readline()
        if not line:
            file.close()
            break
        yield line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', type=str,
                        help='Path to a pre-trained model.')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints.')
    parser.add_argument('--debug', action="store_true",
                        required=False, help='set debug output.')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('--warm', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--world_size', type=int, default=0,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='run trainer in distributed or standalone',
                        required=False)

    parser.add_argument('--config', type=str, help='set config file',
                        default='config.yaml',
                        required=False)

    spec = ExperimentSpecs()
    ds_spec = spec.get_dataset_spec(dataset_name='LJSpeech')
    files = [
        Path(ds_spec['dir']).expanduser() / ds_spec['training_meta'],
        Path(ds_spec['dir']).expanduser() / ds_spec['validation_meta'],
        Path(ds_spec['dir']).expanduser() / ds_spec['test_meta']
    ]

    num_ent = []
    for i, f in enumerate(files):
        resolved = f.resolve()
        num_lines = 0
        if resolved.exists():
            for j, line in enumerate(generator_read(resolved)):
                num_lines += 1
        num_ent.append(num_lines)

    splits = [500, 1000, 2000, 5000]
    for split in splits:
        resolved = files[0].resolve()
        if not resolved.exists():
            print("file not found.")

        new_file = Path(f"{str(resolved)[:-4]}_{split}.txt").resolve()
        with open(str(new_file), "w", encoding="utf8") as new_file:
            print("Writing", f"{str(resolved)[:-4]}_{split}.txt")
            for j, line in enumerate(generator_read(resolved)):
                new_file.write(line)
                if j == split:
                    break
