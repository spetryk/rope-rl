from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader

class RopeTrajectoryDataset(Dataset):
    """ Rope trajectory dataset """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.timestamps = self.filter_trajectories().sort() # sorted list of all valid timestamps

    def __getitem__(self, idx):
        # returns depth, json, mask, and npy configs for given idx item
        ts = self.timestamps[idx]
        traj = []
        for i in range(4): 
            idx = 'start' if i == 0 else str(i-1)
            depth_path = os.path.join(self.data_dir, "depth/", '{}_segdepth_{}.png'.format(ts, idx))
            json_path = os.path.join(self.data_dir, "json/", '{}_{}.json'.format(ts, idx))
            mask_path = os.path.join(self.data_dir, "mask/", '{}_segmask_{}.png'.format(ts, idx))
            npy_path = os.path.join(self.data_dir, "npy/", '{}_raw_depth_{}.npy'.format(ts, idx))
            # TODO: figure out how to load them properly.... rn only loading path names
            traj.append((depth_path, json_path, mask_path, npy_path))
        assert len(traj) == 4

        return traj

    def __len__(self):
        return len(self.timestamps)

    def filter_trajectories(self):
        # filter filenames that are incomplete trajectories
        files = os.listdir(os.path.join(self.data_dir), "json/")
        tags = files.split('_')
        return list(set([ts for ts in tags if tags.count(ts) == 4]))


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
