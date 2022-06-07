# dtc Dueling Turing Classifier

The idea for this project was motivated primarily by Turing’s influential
paper "Computing Machinery and Intelligence" [1] 

First of all, I think most researchers who use datasets must and should 
acknowledge this LJ for this fantastic dataset.

- [https://keithito.com/LJ-Speech-Dataset/]
- [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions] (https://arxiv.org/abs/1712.05884)
- [https://keithito.com/LJ-Speech-Dataset/]

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

- ds_type: "audio"
- file_type: "wav"

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

Second section. lj_speech_full already converted dataset to torch.tensor format. So all Spectrogram, 
SFTS and text serialized a tensor.


## Settings

Setting mainly describes the behavior of a trainer. 

```yaml
settings:
  # debug mode
  debug:
    early_stopping: True
    epochs_log:  1000
    start_test:  10
    epochs_test: 10
    epochs_save: 10
    tensorboard_update: 10
  baseline:
    epochs: 500                  #  total epochs
    batch_size: 32
    grad_clipping: True           # enables grad clipping and max norm
    grad_max_norm : 1.0
    tensorboard_update: 100       # update rate for tensorboard, when we log to tb
    early_stopping: True          # early stopping
    console_log_rate: 5           # when to log to console
    start_test: 20                # this no op for now,  sometime we want wait before we run validation.
    predict: 50                   # will do validation , prediction every 10 iteration .(re-scale based on ds/batch size)
    # use epoch or iteration counter, in large batch size we might want to see progress.
    predict_per_iteration: False
    # when to save
    epochs_save: 2
    save_per_iteration: False
    seed: 1234                       # fix seed, importantly it must for DDP
    fp16: False                      # fp16 run
    distributed: False               # what distributed backend uses.
    backend: "nccl"                  # backend: "gloo"
    url: "tcp://localhost:54321"
    master_address: localhost
    master_port: 54321
    cudnn_enabled: True
    cudnn_benchmark: False
    workers:
      - 192.168.254.205
    dataloader:
      train_set:
        num_workers: 1
        drop_last: True
        pin_memory: True
        shuffle: True
      validation_set:
        num_workers: 1
        drop_last: False
        pin_memory: True
        shuffle: False
```
## Model and Optimizer settings.

All model specs defined in models , you can also define different optimizer settings and bind
specific optimizer for a model.  

Note that model are just symbolic names, internally trainer uses model: to dispatch to
correct factory method.

```yaml

# lr_schedulers definition
lr_schedulers:
    - type: multistep
      milestones: [ 400, 1000 ]
      name: main_lr_scheduler
    - type: ReduceLROnPlateau
      mode: 'min'   #min, max
      patience: 10
      name: dtc_scheduler
#
# Training strategy
strategy:
  tacotron25:
    type: sequential
    order:
     - spectrogram_layer
     - wav2vec
  dtc:
    type: sequential
    order:
     - spectrogram_layer
     - wav2vec

# Model definition
models:
  # this pure model specific, single model can describe both edges and nodes
  # in case we need use single model for edge and node prediction task ,
  # use keyword single_model: model_name
  tacotron25:
    spectrogram_layer:
      model: tacotron25
      optimizer: dtc_optimizer
#      lr_scheduler: main_lr_scheduler
      reverse_decoder: False
      has_input: True
      has_output: True
      max_wav_value: 32768.0
      frames_per_step: 1
      sampling_rate: 22050
      filter_length: 1024   # length of the FFT window
      win_length: 1024      # each frame of audio is windowed by
      hop_length: 256
      n_mel_channels: 80
      mel_fmin: 0.0
      mel_fmax: 8000.0
      symbols_embedding_dim: 512
      encoder:
        desc: "Encoder parameters"
        dropout_rate: 0.5
        num_convolutions: 3
        embedding_dim: 512
        kernel_size: 5
      decoder:
        desc: "Decoder layer and parameters"
        fps: 1
        max_decoder_steps: 1000
        gate_threshold: 0.5
        attention_dropout: 0.1
        decoder_dropout: 0.1
        rnn_dim: 1024
        pre_net_dim: 256
      attention:
        desc: "Attention layer and parameters"
        rnn_dim: 1024
        attention_dim: 128
      attention_location:
        desc: "Location Layer parameters"
        num_filters: 32
        kernel_size: 31
      post_net:
        desc: "Mel post-processing network"
        embedding_dim: 512
        kernel_size: 5
        num_convolutions: 5
    vocoder:
      state: disabled
      name: Test
      model: GraphLSTM
      optimizer: edge_optimizer
      lr_scheduler: main_lr_scheduler
      input_size: 1
  dtc:
    spectrogram_layer:
      model: tacotron3
      optimizer: dtc_optimizer
      #      lr_scheduler: main_lr_scheduler
      reverse_decoder: False
      enable_stft_loss: True
      has_input: True
      has_output: True
      max_wav_value: 32768.0
      frames_per_step: 1
      sampling_rate: 22050
      filter_length: 1024   # length of the FFT window
      win_length: 1024      # each frame of audio is windowed by
      hop_length: 256
      n_mel_channels: 80
      mel_fmin: 0.0
      mel_fmax: 8000.0
      symbols_embedding_dim: 512
      encoder:
        desc: "Encoder parameters"
        dropout_rate: 0.5
        num_convolutions: 3
        embedding_dim: 512
        kernel_size: 5
      decoder:
        desc: "Decoder layer and parameters"
        fps: 1
        max_decoder_steps: 1000
        gate_threshold: 0.5
        attention_dropout: 0.1
        decoder_dropout: 0.1
        rnn_dim: 1024
        pre_net_dim: 256
      attention:
        desc:  "Attention layer and parameters"
        rnn_dim: 1024
        attention_dim: 128
      attention_location:
        desc: "Location Layer parameters"
        num_filters: 32
        kernel_size: 31
      post_net:
        desc: "Mel post-processing network"
        embedding_dim: 512
        kernel_size: 5
        num_convolutions: 5
    vocoder:
      name: Test
      state: disabled
      model: SomeModel
      optimizer: SomeOptimizer
      lr_scheduler: SomeLar
      input_size: 1
```

Hyperparameters tunner

```yaml
ray:
  batch_size: [32, 64]
  lr_min: 1e-4
  lr_max: 1e-1
  num_samples: 10
  checkpoint_freq: 4
  resources:
    cpu: 4
    gpu: 1
  attention_location_filters: 32
  attention_kernel_size: 31
  grad_clip:
    min: 0.5
    max: 1.0
```


## Trainer logic
All trainer logic is abstracted in the generic trainer.
During the initial start, the trainer takes specs i.e., the yaml file invokes 
the factory method.  That creates a model, a model-specific optimizer, and a scheduler.
Note my assumption that the model can be stacked. Hence internally, it queues.
So, for example, if you have two models and you train, you can define two sub-layers.

A good example is if you want to train DTC with a different Vocoder, or, 
for instance, Tacotron 2 and WaveGlow
 
Note right trainer is logically executed in a sequence generally backward in torch 
implementation and can not be executed in parallel anyway. Hence queue is just FIFO.


