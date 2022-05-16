# dtc


docker run -it --gpus=all --rm nvidia/cuda:11.4.2-base-ubuntu20.04 nvidia-smi

docker run -it --gpus=all --rm nvidia/cuda:11.6.0-base-ubuntu20.04 nvidia-smi

#11.6.0-base-ubuntu20.04

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


=====================
== NVIDIA TensorRT ==
=====================

NVIDIA Release 22.04 (build 34850664)
NVIDIA TensorRT Version 8.2.4
Copyright (c) 2016-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Container image Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

https://developer.nvidia.com/tensorrt

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

To install Python sample dependencies, run /opt/tensorrt/python/python_setup.sh

To install the open-source samples corresponding to this TensorRT release version
run /opt/tensorrt/install_opensource.sh.  To build the open source parsers,
plugins, and samples for current top-of-tree on master or a different branch,
run /opt/tensorrt/install_opensource.sh -b <branch>
See https://github.com/NVIDIA/TensorRT for more information.







