import matplotlib
matplotlib.use('Agg')
import argparse
import collections
import torch
import numpy as np
import os

from behavioral_cloning_roperl.data_loader.data_loaders import RopeTrajectoryDataset
from behavioral_cloning.model.model import BasicModel, ResNet18

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


import matplotlib.pyplot as plt

def main(args):

    if args.pretrained:
        # Use ImageNet stats, for both feature desc and depth image networks
        none_stats = {'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]}
        priya_stats = none_stats


    else:
        # Using depth images and not pretrained: single channel input
        # Mean and std dev of depth image train dataset
        with open('stats_pre_none.json') as f:
            none_stats = json.load(f)
            

        # Feature descriptors and not pretrained
        with open('stats_POST_priya.json') as f:
            priya_stats = json.load(f)
            
    none_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(none_stats['mean'], none_stats['std'])
    ])
    priya_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(priya_stats['mean'], priya_stats['std'])
    ])


    test_dataset_none = RopeTrajectoryDataset(args.test_dir, args.network_dir, args.network, 
                                         cfg_dir=args.config, transform=none_transform, features='none', save_im=False, dataset_fraction=1, pretrained=args.pretrained)
    test_dataloader_none = DataLoader(test_dataset_none, batch_size=32, shuffle=False)

    test_dataset_priya = RopeTrajectoryDataset(args.test_dir, args.network_dir, args.network, 
                                         cfg_dir=args.config, transform=priya_transform, features='priya', save_im=False, dataset_fraction=1, pretrained=args.pretrained)
    test_dataloader_priya = DataLoader(test_dataset_priya, batch_size=32, shuffle=False)


    model_paths = []
    for size in ['high', 'med', 'low']:
        info = {}
        for feat in ['priya', 'none']:
            mdir = os.path.join(args.model_dir, feat, size)
            files = os.listdir(mdir)
            files.sort(key=lambda f: int(filter(str.isdigit, f)))
            best_model = os.path.join(mdir, files[-1])
            print('...processing: {}'.format(best_model))
            if feat is 'none':
                info[feat] = (eval_model(test_dataloader_none, best_model, feat, args.pretrained, args.save_dir, size))
            else:
                info[feat] = (eval_model(test_dataloader_priya, best_model, feat, args.pretrained, args.save_dir, size))

        print('priya: {}'.format(info["none"]))
        print('none-: {}'.format(info["priya"]))

def eval_model(dataloader, model_path, feat, pretrained, save_dir, size):
    model = ResNet18(args.pretrained, channels=1 if not pretrained and feat == 'none' else 3).float()
    # model = BasicModel().float()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.MSELoss()
    test_loss = []
    for idx, (obs, targets, filenames) in enumerate(dataloader):
        pred = model(obs.float())
        t = torch.zeros(pred.shape)
        for i in range(0, len(targets)):
            t[:,i] = targets[i]
        targets = t.float()
        loss = criterion(pred, targets)
        test_loss.append(loss.item())
        
        # Save the image outputs
        for t, p, f in zip(targets, pred, filenames):
            
            
            if pretrained and feat == "none":
                plt.figure(1)
                plt.title("ResNet18 pretrained on ImageNet finetuned on depth images")
            elif pretrained and feat == "priya":
                plt.figure(2)
                plt.title("ResNet18 pretrained on ImageNet finetuned on descriptor images")
            elif not pretrained and feat == "none":
                plt.figure(3)
                plt.title("ResNet18 trained on depth images")
            else:
                plt.figure(4)
                plt.title("ResNet18 trained on descriptor images")

            plt.scatter([t[0].item()], [t[1].item()], c='r', marker='o', label="ground truth grasp") # grasp [target]
            plt.scatter([t[3].item()], [t[4].item()], c='r', marker='x', label="ground truth drop") # drop [target]
            plt.scatter([p[0].item()], [p[1].item()], c='b', marker='o', label="predicted grasp") # grasp [pred]
            plt.scatter([p[3].item()], [p[4].item()], c='b', marker='x', label="predicted drop") # drop [pred]

            plt.figure()
            plt.scatter([t[0].item()], [t[1].item()], c='r', marker='o', label="ground truth grasp") # grasp [target]
            plt.scatter([t[3].item()], [t[4].item()], c='r', marker='x', label="ground truth drop") # drop [target]
            plt.scatter([p[0].item()], [p[1].item()], c='b', marker='o', label="predicted grasp") # grasp [pred]
            plt.scatter([p[3].item()], [p[4].item()], c='b', marker='x', label="predicted drop") # drop [pred]
            plt.legend()
            fn = os.path.join(save_dir, f)
            plt.savefig('{}_points.png'.format(fn))
            print('saving plot to:', '{}_points.png'.format(fn))
    for i in range(1,5):
        plt.figure(i)
        plt.legend()
        fn = os.path.join(save_dir, "plt_figure_" + str(i))
        plt.savefig('{}.png'.format(fn))

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
    parser.add_argument('--save_dir', default='visualizations/test/', type=str,
                      help='directory to validation data')
    
    # training specific arguments
    parser.add_argument('--model_dir', default='bc_model/', type=str,
                      help='path to the pretrained model')

    parser.add_argument('--pretrained', action='store_true',
                    help='If using depth images, flag to use pretrained model')

    args = parser.parse_args()
    main(args)
