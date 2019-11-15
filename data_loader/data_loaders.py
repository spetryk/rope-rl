from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import numpy as np

class RopeDataLoader(BaseDataLoader):
    """
    Data loader for rope data. Currently just depth images -> actions.
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


    def __len__(self):
        pass

    def __get_item__(self):
        pass

    def find_groups(self):
        """
        Find groups of full demonstrations with all data available
        Return list of timestamps where all segdepths and json present
        """
        timestamps = []



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
