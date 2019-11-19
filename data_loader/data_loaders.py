from torchvision import datasets, transforms
#from base import BaseDataLoader
import os
import numpy as np
import re
import cv2
import json

from torch.utils.data import Dataset, DataLoader

class RopeTrajectoryDataset(Dataset):
    """ Rope trajectory dataset """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.timestamps = self.filter_trajectories() # list of all valid timestamps

    def __getitem__(self, idx):
        # returns depth and json for random depth/action pair in given idx sequence
        ts = self.timestamps[idx]
        # choose random sample in trajectory
        i = np.random.choice(np.arange(3))

        idx_prev = 'start' if i == 0 else str(i-1)
        idx_after = str(i)
        depth_path = os.path.join(self.data_dir, "depth/", '{}_segdepth_{}.png'.format(ts, idx_prev))
        json_path = os.path.join(self.data_dir, "json/", '{}_{}.json'.format(ts, idx_after))
        mask_path = os.path.join(self.data_dir, "mask/", '{}_segmask_{}.png'.format(ts, idx_prev))
        npy_path = os.path.join(self.data_dir, "npy/", '{}_raw_depth_{}.npy'.format(ts, idx_prev))

        depth_image = cv2.imread(depth_path) # note: rn RGB, change this to grayscale
        with open(json_path) as f:
            actions = json.load(f)

        return {'depth': depth_image, 'actions': actions}

    def __len__(self):
        return len(self.timestamps)

    def filter_trajectories(self):
        # filter filenames that are incomplete trajectories
        files = os.listdir(os.path.join(self.data_dir, "json"))
        # extract timestamp
        exp = "(.+)_\d\.json"
        tags = [re.match(exp, f) for f in files]
        tags = list(filter(lambda x: x is not None, tags))
        tags = [x.groups()[0] for x in tags]
        timestamps = []
        for ts in tags:
            if tags.count(ts) == 3:
                timestamps.append(ts)
        return list(set(timestamps))


# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
