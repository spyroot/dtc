FROM  nvidia/cuda:11.6.0-base-ubuntu20.04
RUN apt-get update
RUN apt-get install -y tzdata
RUN apt-get install -y build-essential devscripts debhelper fakeroot
RUN apt-get install -y gcc meson git wget numactl make curl ninja-build python3-pip unzip zip gzip
RUN apt-get install -y libnuma-dev vim openssh-server zsh fzf \
    cuda-cudart-dev-11-6 cuda-libraries-dev-11-6 cuda-libraries-dev-11-6 libcusparse-dev-11-6 \
    install nvidia-cuda-toolkit

RUN a
 python3-wheel python3-libnvinfer
#RUN make pkg.debian.build
EXPOSE 22/tcp
EXPOSE 54321/tcp

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
