git pull
rm results/model/.*dat
python main.py --rank 1 --world_size 1 --local_rank 0 --device_id 0 --config config_worker.yaml