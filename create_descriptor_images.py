from torchvision import datasets, transforms
import argparse
import os
import numpy as np
import re
import cv2
import json
import yaml
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tools.dense_correspondence_network import DenseCorrespondenceNetwork
from tools.find_correspondences import CorrespondenceFinder


def main(args):
    network_path = os.path.join(args.network_dir, args.network)
    dcn = DenseCorrespondenceNetwork.from_model_folder(network_path, model_param_file=os.path.join(network_path, '003501.pth'))
    dcn.eval()
    with open(os.path.join(args.config, 'dataset_info.json'), 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev)
    descriptor_stats_config = "stats_pre_priya.json"

    for filename in os.listdir(args.depth_image_dir):
        depth_path = os.path.join(args.depth_image_dir, filename)
        name_parts = filename.split("_")
        name_parts[2] = "priya_descriptors"
        output_filename = "_".join(name_parts)
        output_path = os.path.join(args.descriptor_image_dir, output_filename)
        desc_image = make_descriptors_images(cf, depth_path, descriptor_stats_config)
        plt.imsave(output_path, desc_image)



def make_descriptors_images(cf, image_path, descriptor_stats_config):
    rgb_a = Image.open(image_path).convert('RGB').resize((640,480))

    # compute dense descriptors
    # This takes in a PIL image!
    rgb_a_tensor = cf.rgb_image_to_tensor(rgb_a)

    # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
    res_a = cf.dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
    with open(descriptor_stats_config) as f:
        descriptor_image_stats = json.load(f)
    res_a = normalize_descriptor(res_a, descriptor_image_stats)

    # Convert to range [0,255] and float32
    res_a = (res_a * 255.).astype(np.uint8)

    # Convert to PIL Image
    res_a =  transforms.ToPILImage()(res_a)
    return res_a


def normalize_descriptor(res, stats=None):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create descriptor Images')
    # vis-descriptor specific file paths
    parser.add_argument('--config', default='cfg', type=str,
                      help='path to config')

    parser.add_argument('--depth_image_dir', default='data/train/depth', type=str,
                      help='directory to training data')
    
    parser.add_argument('--descriptor_image_dir',  default='data/train/descriptor_images', type=str,
                      help='directory to output descriptor images into')
    
    parser.add_argument('--network_dir', default='/nfs/diskstation/priya/rope_networks/', type=str,
                      help='directory to vis_descriptor network')
    
    parser.add_argument('--network', default='rope_noisy_1400_depth_norm_3', type=str,
                      help='filename of vis_descriptor network')
    

    args = parser.parse_args()
    main(args)