import argparse
import collections
import torch
import numpy as np
import os
import json

from behavioral_cloning.data_loader.get_stats_dataloader import RopeTrajectoryDataset
from behavioral_cloning.model.model import BasicModel

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

def main(args):
    dataset = RopeTrajectoryDataset(args.train_dir, args.network_dir, args.network, 
                                         cfg_dir=args.config, transform=None, features='priya', postprocess=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    print("batch size", len(dataset))
    js = {}
    for idx, obs in enumerate(dataloader):
        obs = obs.view(-1, 3)
        mean = obs.mean(0)
        print(mean)
        std = obs.std(0)
        print(std)
        js['mean'] = mean.item()
        js['std'] = std.item()
        print(obs.shape)
        mins, _ = torch.min(obs, 0)
        maxs, _ = torch.max(obs, 0)
        js['min'] = mins.detach().numpy().tolist()
        js['max'] = maxs.detach().numpy().tolist()
    with open(args.save_file, 'w') as outfile:
        json.dump(js, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    # vis-descriptor specific file paths
    parser.add_argument('--config', default='cfg', type=str,
                      help='path to config')
    parser.add_argument('--train_dir', default='data/train/', type=str,
                      help='directory to validation data')
    parser.add_argument('--network_dir', default='/nfs/diskstation/priya/rope_networks/', type=str,
                      help='directory to vis_descriptor network')
    parser.add_argument('--network', default='rope_noisy_1400_depth_norm_3', type=str,
                      help='filename of vis_descriptor network')
    parser.add_argument('--save_file', default='vis_descriptor_stats.json', type=str,
                      help='directory to validation data')
    args = parser.parse_args()
    main(args)
