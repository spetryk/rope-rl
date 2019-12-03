import os
import re
import subprocess

data_dir = './data/train'

# filter filenames that are incomplete trajectories
files = os.listdir(os.path.join(data_dir, "json"))

# extract timestamp
exp = "(.+)_\d\.json"
tags = [re.match(exp, f) for f in files]
tags = list(filter(lambda x: x is not None, tags))
tags = [x.groups()[0] for x in tags]
timestamps = set()
depth_list = []
json_list = []
mask_list = []
npy_list = []

for ts in tags:
    if tags.count(ts) == 3 and ts not in timestamps:
        timestamps.add(ts)
        for i in range(3):
            ob_idx = 'start' if i == 0 else str(i-1)
            ac_idx = str(i)
            depth_list.append(os.path.join(data_dir, "depth/", '{}_segdepth_{}.png'.format(ts, ob_idx)))
            json_list.append(os.path.join(data_dir, "json/", '{}_{}.json'.format(ts, ac_idx)))
            mask_list.append(os.path.join(data_dir, "mask/", '{}_segmask_{}.png'.format(ts, ob_idx)))
            npy_list.append(os.path.join(data_dir, "npy/", '{}_raw_depth_{}.npy'.format(ts, ob_idx)))

print("dataset_size", len(depth_list))

# 30% in val
num_val = int(len(depth_list)*.3)

# move first num_val samples into val folder
for i in range(num_val):
    subprocess.run(["mv", depth_list[i], 'data/val/depth/'])
    subprocess.run(["mv", json_list[i], 'data/val/json/'])
    subprocess.run(["mv", mask_list[i], 'data/val/mask/'])
    subprocess.run(["mv", npy_list[i], 'data/val/npy/'])


