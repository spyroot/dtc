git pull
rm results/model/.*dat
python main.py --rank 0 --world_size 2 --local_rank 0 --device_id 0 --config config_master.yaml