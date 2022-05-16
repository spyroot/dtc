FROM nvcr.io/nvidia/tensorrt:22.04-py3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get upgrade -y
RUN apt-get install -y build-essential
RUN apt-get install -y libnuma-dev vim openssh-server zsh fzf
RUN apt-get install -y gcc meson git wget numactl make curl ninja-build python3-pip unzip zip gzip
RUN apt-get install -y libnuma-dev vim openssh-server zsh fzf net-tools nmap

RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i "/Port/s/^#//g" /etc/ssh/sshd_config
RUN sed -i "/AddressFamily/s/^#//g" /etc/ssh/sshd_config
RUN sed -i "/PermitEmptyPasswords/s/^#//g" /etc/ssh/sshd_config

RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN mkdir /var/run/sshd
RUN bash -c 'install -m755 <(printf "#!/bin/sh\nexit 0") /usr/sbin/policy-rc.d'
RUN ex +'%s/^#\zeListenAddress/\1/g' -scwq /etc/ssh/sshd_config
RUN ex +'%s/^#\zeHostKey .*ssh_host_.*_key/\1/g' -scwq /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

RUN RUNLEVEL=1 dpkg-reconfigure openssh-server
RUN ssh-keygen -A -v
RUN update-rc.d ssh defaults

# Configure sudo.
#RUN ex +"%s/^%sudo.*$/%sudo ALL=(ALL:ALL) NOPASSWD:ALL/g" -scwq! /etc/sudoers

# Generate and configure user keys.
#USER ubuntu
RUN ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
#COPY --chown=ubuntu:root "./files/authorized_keys" /home/ubuntu/.ssh/authorized_keys
# Setup default command and/or parameters.
RUN apt-get install -y ffmpeg
RUN apt-get install -y tzdata ffmpeg libsndfile1 libtiff-dev libpng-dev meson ninja-build cmake yamllint
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115 \
RUN pip install loguru librosa pip install unidecode
RUN pip install inflect tensorboard ray matplotlib opencv-python tqdm pandas tabulate dill

EXPOSE 22
EXPOSE 54321/tcp
CMD ["/usr/sbin/sshd", "-D", "-o", "ListenAddress=0.0.0.0"]
