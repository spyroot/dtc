# dtc


## Convert and generate datasets

```shell
main.py --convert --dataset_name LJSpeech
Going to write to dir C:\Users\neros\Dropbox\Datasets\LJSpeech-1.1
Converting:   0%|          | 0/12500 [00:00<?, ?it/s]
Converting: 100%|██████████| 12500/12500 [06:32<00:00, 31.86it/s]
Saving  ~Dropbox\Datasets\LJSpeech-1.1\LJSpeech_train_num_sam_12500_filter_80_3.pt
MD5 checksum ae7379fcfa79f47feac81ff1e044a056
```

```
docker run -it --gpus=all --rm nvidia/cuda:11.4.2-base-ubuntu20.04 nvidia-smi
docker run -it --gpus=all --rm nvidia/cuda:11.6.0-base-ubuntu20.04 nvidia-smi
```

#11.6.0-base-ubuntu20.04

```shell
docker build -t dtc_rt:v1 .
docker run --privileged --name dtc_win_build --rm -i -t dtc_win_build bash
    docker run -it --mount src="$(pwd)",target=/test_container,type=bind k3_s3
docker run -t -i -v <host_dir>:<container_dir>  ubuntu /bin/bash

docker build -t dtc_rt:v1 .
docker run -it --gpus=all --rm -p 2222:2222 -p 22:22 -p 54321:54321 -p 54321:54321 dtc_rt:v1 /bin/bash
docker run -it --gpus=all --rm dtc_rt:v1 /bin/bash

docker run --gpus=all --rm -p 2222:2222 -p 22:22 -p 54322:54322 -p 54321:54321 dtc_rt:v1

docker run --gpus=all --rm -p 2222:22 -p 54321:54321 -v ~\Dropbox\Datasets:/datasets dtc_rt:v1

docker run --privileged --gpus=all --rm -p 2222:22 -p 54321:54321 -v ${PWD}:/datasets --workdir=/datasets dtc_rt:v1
```





