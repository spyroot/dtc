git pull
cp config_worker.yaml config.yaml
python main.py --rank 1 --world_size 2
