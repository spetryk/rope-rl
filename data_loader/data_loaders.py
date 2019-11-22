from torchvision import datasets, transforms
#from base import BaseDataLoader
import os
import numpy as np
import re
import cv2
import json

from torch.utils.data import Dataset, DataLoader
from dense_correspondence_network import DenseCorrespondenceNetwork
from find_correspondences import CorrespondenceFinder

class RopeTrajectoryDataset(Dataset):
    """ Rope trajectory dataset """
    def __init__(self, data_dir, network_dir, network, cfg_dir='../cfg', transform=None):
        self.data_dir = data_dir
        self.timestamps, self.depth_list, self.json_list, self.mask_list, self.npy_list = self.filter_trajectories() 

        # Path to network 
        network_path = os.path.join(network_dir, network)
        self.dcn = DenseCorrespondenceNetwork.from_model_folder(network_path, model_param_file=os.path.join(network_path, '003501.pth'))
        self.dcn.eval()
        with open(os.path.join(cfg_dir, 'dataset_info.json'), 'r') as f:
            dataset_stats = json.load(f)
        dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
        self.cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev)
        self.descriptor_stats_config = os.path.join(network_path, 'descriptor_statistics.yaml')


    def __getitem__(self, idx):
        depth_path = self.depth_list(idx)
        json_path = self.json_list(idx)
        mask_path = self.mask_list(idx)
        npy_path = self.npy_list(idx)

        desc_image = make_descriptors_images(self.cf, args.image_dir, args.save_dir, descriptor_stats_config, 
                                             make_masked_video=args.mask, mask_folder=args.mask_dir)

        with open(json_path) as f:
            actions = json.load(f)

        return desc_image, actions

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


    def make_descriptors_images(self, image_path):
        rgb_a = Image.open(image_path).convert('RGB')

        # compute dense descriptors
        # This takes in a PIL image!
        rgb_a_tensor = self.cf.rgb_image_to_tensor(rgb_a)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = self.cf.dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        #descriptor_image_stats = yaml.load(file(descriptor_stats_config), Loader=CLoader)
        descriptor_image_stats = yaml.load(file(self.descriptor_stats_config))
        res_a = self.normalize_descriptor(res_a, descriptor_image_stats["mask_image"])
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