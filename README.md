# dtc Dueling Turing Classifier

The idea for this project was motivated primarily by Turing’s influential
paper "Computing Machinery and Intelligence" [1] 

First of all, I think most researchers who use datasets must and should 
acknowledge this LJ for this fantastic dataset.

 [https://keithito.com/LJ-Speech-Dataset/]
 

 [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
 [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions] (https://arxiv.org/abs/1712.05884)
 [https://keithito.com/LJ-Speech-Dataset/]

This repository is the PyTorch implementation. 

###

This repository is the PyTorch implementation and improvements and some new ideas
focused on Tacotron 2 and WaveGlow


## Installation

The code has been tested over PyTorch latest version 1.10 and Python 3.9 and 3.10
 - Install PyTorch following the instructions on the [official website](https://pytorch.org/).
 - Check requirement files.
 - 
```bash
conda env create -f environment.yml
conda activate dtc
conda install pytorch torchvision torchaudio  -c pytorch
```

Note Trainer and code base ready for hyperparameter tunning , hence you want install Ray.

```bash
pip install ray  # minimal install
```

We can run code in colab, jupiter or standalone app.
```bash
 For Colab you need follow colab notebook.
```
Then install the other dependencies.

```bash
pip install -r requirements.txt
```

First create config.yaml file

## Download Datasets

Note Datasets abstraction provide option to download a dataset, either as raw , torch tensor or numpy.


## Convert and generate datasets

You want to do that only if you want to serialize the torch tensor on disk.

```shell
main.py --convert --dataset_name LJSpeech
Going to write to dir C:\Users\neros\Dropbox\Datasets\LJSpeech-1.1
Converting:   0%|          | 0/12500 [00:00<?, ?it/s]
Converting: 100%|██████████| 12500/12500 [06:32<00:00, 31.86it/s]
Saving  ~Dropbox\Datasets\LJSpeech-1.1\LJSpeech_train_num_sam_12500_filter_80_3.pt
MD5 checksum ae7379fcfa79f47feac81ff1e044a056
```

```docker
docker run -it --gpus=all --rm nvidia/cuda:11.4.2-base-ubuntu20.04 nvidia-smi
docker run -it --gpus=all --rm nvidia/cuda:11.6.0-base-ubuntu20.04 nvidia-smi
```

# For ubuntu based

11.6.0-base-ubuntu20.04

```shell
docker build -t dtc_rt:v1 .
docker run --privileged --name dtc_win_build --rm -i -t dtc_win_build bash
docker run -it --mount src="$(pwd)",target=/test_container,type=bind k3_s3
docker run -t -i -v <host_dir>:<container_dir>  ubuntu /bin/bash
```

Alternative build in case you want expose DDP.

```shell
docker build -t dtc_rt:v1 .
docker run -it --gpus=all --rm -p 2222:2222 -p 22:22 -p 54321:54321 -p 54321:54321 dtc_rt:v1 /bin/bash
docker run -it --gpus=all --rm dtc_rt:v1 /bin/bash
docker run --gpus=all --rm -p 2222:2222 -p 22:22 -p 54322:54322 -p 54321:54321 dtc_rt:v1
docker run --gpus=all --rm -p 2222:22 -p 54321:54321 -v ~\Dropbox\Datasets:/datasets dtc_rt:v1
docker run --privileged --gpus=all --rm -p 2222:22 -p 54321:54321 -v ${PWD}:/datasets --workdir=/datasets dtc_rt:v1
```

## Configuration

All baseline configs in source repo,  check config.yaml

Global settings 

```yaml
train: True                 # train or not,  default is True for generation we only need load pre-trained model
use_dataset: 'LJSpeech'     # dataset set generated.
use_model: 'dtc'            # model to use , it must be defined in models section.
draw_prediction: True       # this nop for now.
load_model: True            # load model or not, and what
load_epoch: 500             # load model,  last epoch
save_model: True            # save model,
regenerate: True            # regenerated,  factor when indicated by epochs_save
active_setting: small       # indicate what setting to use, so we can switch from debug to production
evaluate: True              # will run evaluation
```

## Workspace

By default trainer will create results folder, for example if you will run python main.py
it will create results folder and all sub folders

```yaml
root_dir: "."
log_dir: "logs"
nil_dir: "timing"
graph_dir: "graphs"
results_dir: "results"
figures_dir: "figures"
prediction_dir: "prediction"               # where we save prediction
model_save_dir: "model"                    # where we save model
metrics_dir: "metrics"                     # metrics dir in case we need store separate metrics
generated_dir: "generated"                 # inference time generated here.
```

Datasets defined in datasets section.  The global parameter use_dataset dictates what dataset 
active at given moment. 

The idea here we can have different type dataset, each dataset has different type.
For raw audio dataset. 

The idea here is that we can have a different type of dataset. Each dataset has a different style and structure.
For the raw audio dataset. 

ds_type: "audio"
file_type: "wav"

FOr example if you are using LJSpeech-1.1

```yaml

datasets:
  LJSpeech:
    format: raw
    ds_type: "audio"
    file_type: "wav"
    dir: "~/Dropbox/Datasets/LJSpeech-1.1"
    training_meta: ljs_audio_text_train_filelist.txt
    validation_meta: ljs_audio_text_val_filelist.txt
    test_meta: ljs_audio_text_test_filelist.txt
    meta: metadata.csv
    recursive: False
  lj_speech_full:
    format: tensor_mel
    ds_type: "audio"
    dir: "~/Dropbox/Datasets/LJSpeech-1.1"
    dataset_files:
      - "LJSpeech_train_num_sam_12500_filter_80_3.pt"
      - "LJSpeech_validate_num_sam_100_filter_80_3.pt"
      - "LJSpeech_test_num_sam_500_filter_80_3.pt"
    training_meta: LJSpeech_train_num_sam_12500_filter_80_3.pt
    validation_meta: LJSpeech_validate_num_sam_100_filter_80_3.pt
    test_meta: LJSpeech_test_num_sam_500_filter_80_3.pt
    checksums:
      - "7bc0f1bac289cfd1ba8ea1c390bddf8f"
      - "5d3b94b131c08afcca993dfbec54c63a"
      - "3291d802351928da7c5d0d9c917c2663"
    meta: metadata.csv
    recursive: False
    file_type: "torch"
```

Second section. lj_speech_full Converted dataset to torch.tensor format. So all spectrograms, 
SFTS and text serialized a tensor. 

``
``

