from torchvision import datasets, transforms
#from base import BaseDataLoader
import os
import numpy as np
import re
import cv2
import json
import yaml
import matplotlib.pyplot as plt

import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tools.dense_correspondence_network import DenseCorrespondenceNetwork
from tools.find_correspondences import CorrespondenceFinder

class RopeTrajectoryDataset(Dataset):
    """ Rope trajectory dataset """
    def __init__(self, data_dir, network_dir, network, cfg_dir='../cfg', transform=None, features='priya',  dataset_fraction=1, save_im=False):
        self.data_dir = data_dir
        self.timestamps, self.depth_list, self.json_list, self.mask_list, self.npy_list = self.filter_trajectories(dataset_fraction)

        # Path to network
        network_path = os.path.join(network_dir, network)
        self.dcn = DenseCorrespondenceNetwork.from_model_folder(network_path, model_param_file=os.path.join(network_path, '003501.pth'))
        self.dcn.eval()
        with open(os.path.join(cfg_dir, 'dataset_info.json'), 'r') as f:
            dataset_stats = json.load(f)
        dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
        self.cf = CorrespondenceFinder(self.dcn, dataset_mean, dataset_std_dev)
        self.descriptor_stats_config = os.path.join(network_path, 'descriptor_statistics.yaml')
        self.transform = transform
        self.features = features
        self.save_im = save_im

    def __getitem__(self, idx):
        depth_path = self.depth_list[idx]
        json_path = self.json_list[idx]
        mask_path = self.mask_list[idx]
        npy_path = self.npy_list[idx]

        save_file_name = ''
        if self.features == 'priya':
            image = self.make_descriptors_images(depth_path)
        else:
            image = Image.open(depth_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image = image.transform(image)

        return image


    def __len__(self):
        return len(self.depth_list)

    def filter_trajectories(self, dataset_fraction):
        # filter filenames that are incomplete trajectories
        files = os.listdir(os.path.join(self.data_dir, "json"))
        print("data", self.data_dir, "dataset_fraction", dataset_fraction)
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
                    depth_list.append(os.path.join(self.data_dir, "depth/", '{}_segdepth_{}.png'.format(ts, ob_idx)))
                    json_list.append(os.path.join(self.data_dir, "json/", '{}_{}.json'.format(ts, ac_idx)))
                    mask_list.append(os.path.join(self.data_dir, "mask/", '{}_segmask_{}.png'.format(ts, ob_idx)))
                    npy_list.append(os.path.join(self.data_dir, "npy/", '{}_raw_depth_{}.npy'.format(ts, ob_idx)))

        print("dataset_size", len(depth_list))
        dataset_size = int(round(len(depth_list) * dataset_fraction))

        return list(set(timestamps))[:dataset_size], depth_list[:dataset_size], json_list[:dataset_size], mask_list[:dataset_size], npy_list[:dataset_size]


    def make_descriptors_images(self, image_path):
        rgb_a = Image.open(image_path).convert('RGB').resize((640,480))
        # compute dense descriptors
        # This takes in a PIL image!
        rgb_a_tensor = self.cf.rgb_image_to_tensor(rgb_a)

        # these are Variables holding torch.FloatTensors, convert to numpy
        res_a = self.cf.dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()

        return res_a
