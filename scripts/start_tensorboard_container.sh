docker run -it -p 8888:8888 -p 6006:6006 \
tensorflow/tensorflow:nightly-py3-jupyter
# tensorboard --logdir=runs
tensorboard --logdir=tensorboard --bind_all