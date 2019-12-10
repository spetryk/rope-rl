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

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

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

        ForkedPdb().set_trace()

        if self.features is 'priya':
            image = self.make_descriptors_images(depth_path)
            if self.save_im:
                save_file_name = os.path.join('data/res/', '{}_res_priya.png'.format(os.path.basename(depth_path)))
                print('saving feat. to: {}', save_file_name)
                plt.imsave(save_file_name, image)
        else:
            image = Image.open(depth_path).convert('RGB')
            if self.save_im:
                save_file_name = os.path.join('data/res/', '{}_res_none.png'.format(os.path.basename(depth_path)))
                print('saving feat. to: {}', save_file_name)
                plt.imsave(save_file_name, image)

            #image = self.cf.rgb_image_to_tensor(image)
            #desc_image = self.normalize_descriptor(desc_image)

        # image must be in range [0,1] by this pt, and must be PIL Image
        assert(torch.max(transforms.ToTensor()(image)) <= 1.)
        assert(torch.min(transforms.ToTensor()(image)) >= 0.)
        #assert(type(image)==PIL.PngImagePlugin.PngImageFile)

        image = self.transform(image)

        actions = []
        with open(json_path) as f:
            js = json.load(f)
            actions.append(np.array(js['grasp']))
            actions.append(np.array(js['drop']))
            actions.append(np.array(js['orientation']))

        actions = np.hstack(actions)

        return image, list(actions), save_file_name

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
        rgb_a = Image.open(image_path).convert('RGB')

        # compute dense descriptors
        # This takes in a PIL image!
        rgb_a_tensor = self.cf.rgb_image_to_tensor(rgb_a)

        # these are Variables holding torch.FloatTensors, convert to numpy
        res_a = self.cf.dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        descriptor_image_stats = yaml.load(file(descriptor_stats_config), Loader=CLoader)
        descriptor_image_stats = yaml.load(file(self.descriptor_stats_config))
        res_a = self.normalize_descriptor(res_a, descriptor_image_stats["entire_image"])
        #print("make_descriptors", self.save_im)
        return res_a

    def normalize_descriptor(self, res, stats=None):
        """
        Normalizes the descriptor into RGB color space
        :param res: numpy.array [H,W,D]
            Output of the network, per-pixel dense descriptor
        :param stats: dict, with fields ['min', 'max', 'mean'], which are used to normalize descriptor
        :return: numpy.array
            normalized descriptor
        """
        if stats is None:
            res_min = res.min()
            res_max = res.max()
        else:
            res_min = np.array(stats['min'])
            res_max = np.array(stats['max'])

        normed_res = np.clip(res, res_min, res_max)
        eps = 1e-10
        scale = (res_max - res_min) + eps
        normed_res = (normed_res - res_min) / scale
        return normed_res
