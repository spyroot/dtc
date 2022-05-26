from tqdm import tqdm
import time

for i in tqdm(range(20), desc='tqdm() Progress Bar', leave_empty=True):
    time.sleep(0.5)
