import argparse
import collections
import torch
import numpy as np
import os

from behavioral_cloning.data_loader.data_loaders import RopeTrajectoryDataset
from behavioral_cloning.model.model import BasicModel

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def main(args):
    test_dataset_none = RopeTrajectoryDataset(args.test_dir, args.network_dir, args.network, 
                                         cfg_dir=args.config, transform=None, features='none')
    test_dataloader_none = DataLoader(test_dataset_none, batch_size=32, shuffle=False)

    test_dataset_priya = RopeTrajectoryDataset(args.test_dir, args.network_dir, args.network, 
                                         cfg_dir=args.config, transform=None, features='priya')
    test_dataloader_priya = DataLoader(test_dataset_priya, batch_size=32, shuffle=False)

    model_paths = []
    for size in ['high', 'med', 'low']:
        info = []
        for feat in ['none', 'priya']:
            mdir = os.path.join(args.model_dir, feat, size)
            files = os.listdir(mdir)
            files.sort(key=lambda f: int(filter(str.isdigit, f)))
            best_model = os.path.join(mdir, files[-1])
            print('...processing: {}'.format(best_model))
            if feat is 'none':
                info.append(eval_model(test_dataloader_none, best_model))
            else:
                info.append(eval_model(test_dataloader_priya, best_model))
        print('priya: {}'.format(info[0]))
        print('none-: {}'.format(info[1]))

def eval_model(dataloader, model_path):
    model = BasicModel().float()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.MSELoss()
    test_loss = []
    for idx, (obs, targets) in enumerate(dataloader):
        pred = model(obs.float())
        t = torch.zeros(pred.shape)
        for i in range(0, len(targets)):
            t[:,i] = targets[i]
        targets = t.float()
        loss = criterion(pred, targets)
        test_loss.append(loss.item())
    return (np.sum(test_loss), np.mean(test_loss), np.std(test_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    # vis-descriptor specific file paths
    parser.add_argument('--config', default='cfg', type=str,
                      help='path to config')
    parser.add_argument('--test_dir', default='data/val/', type=str,
                      help='directory to validation data')
    parser.add_argument('--network_dir', default='/nfs/diskstation/priya/rope_networks/', type=str,
                      help='directory to vis_descriptor network')
    parser.add_argument('--network', default='rope_noisy_1400_depth_norm_3', type=str,
                      help='filename of vis_descriptor network')
    
    # training specific arguments
    parser.add_argument('--model_dir', default='bc_model/', type=str,
                      help='path to the pretrained model')
    args = parser.parse_args()
    main(args)
