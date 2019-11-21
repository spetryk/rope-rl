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
        self.timestamps, self.depth_list, self.json_list, self.mask_list, self.npy_list = self.filter_trajectories() 

    def __getitem__(self, idx):
        depth_path = self.depth_list(idx)
        json_path = self.json_list(idx)
        mask_path = self.mask_list(idx)
        npy_path = self.npy_list(idx)

        depth_image = cv2.imread(depth_path)
        with open(json_path) as f:
            actions = json.load(f)

        return {'depth': depth_image, 'actions': actions}

    def __len__(self):
        return len(self.depth_list)

    def filter_trajectories(self):
        # filter filenames that are incomplete trajectories
        files = os.listdir(os.path.join(self.data_dir, "json"))
        # extract timestamp
        exp = "(.+)_\d\.json"
        tags = [re.match(exp, f) for f in files]
        tags = list(filter(lambda x: x is not None, tags))
        tags = [x.groups()[0] for x in tags]
        timestamps = []
        depth_list = []
        json_list = []
        mask_list = []
        npy_list = []

        for ts in tags:
            if tags.count(ts) == 3:
                timestamps.append(ts)
                for i in range(3):
                    ob_idx = 'start' if i == 0 else str(i-1)
                    ac_idx = str(i)
                    depth_list.append(os.path.join(self.data_dir, "depth/", '{}_segdepth_{}.png'.format(ts, ob_idx)))
                    json_list.append(os.path.join(self.data_dir, "json/", '{}_{}.json'.format(ts, ac_idx)))
                    mask_list.append(os.path.join(self.data_dir, "mask/", '{}_segmask_{}.png'.format(ts, ob_idx)))
                    npy_list.append(os.path.join(self.data_dir, "npy/", '{}_raw_depth_{}.npy'.format(ts, ob_idx)))

        return list(set(timestamps)), depth_list, json_list, mask_list, np_list


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
