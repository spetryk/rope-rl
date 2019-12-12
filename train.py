import argparse
import collections
import torch
import numpy as np
import os
import json

from behavioral_cloning.data_loader.data_loaders import RopeTrajectoryDataset
from behavioral_cloning.model.model import BasicModel, ResNet18

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def mse_loss(pred, targets):
    """
    MSE loss, accounting for real angle differences in orientation
    """
    diff = pred - targets
    orientation_diff = torch.abs(diff[:,-1])
    pi_mask     = ((orientation_diff >=   np.pi/2.) * (orientation_diff < 3*np.pi/2.)).float()
    pi_mask     = pi_mask * -np.pi
    two_pi_mask = ((orientation_diff >= 3*np.pi/2.) * (orientation_diff <    2*np.pi)).float()
    two_pi_mask = two_pi_mask * (-2.*np.pi)
    orientation_diff = orientation_diff + pi_mask + two_pi_mask
    mask = torch.ones(diff.shape).float().cuda()
    mask[:,-1] = orientation_diff
    diff = diff * mask
    mse_loss = (diff**2).mean()
    return mse_loss



def mse_loss_separated(pred, targets):
    """
    MSE loss, accounting for real angle differences in orientation
    """
    diff = pred - targets
    orientation_diff = torch.abs(diff[:,-1])
    pi_mask     = ((orientation_diff >=   np.pi/2.) * (orientation_diff < 3*np.pi/2.)).float()
    pi_mask     = pi_mask * -np.pi
    two_pi_mask = ((orientation_diff >= 3*np.pi/2.) * (orientation_diff <    2*np.pi)).float()
    two_pi_mask = two_pi_mask * (-2.*np.pi)
    orientation_diff = orientation_diff + pi_mask + two_pi_mask
    mask = torch.ones(diff.shape).float()
    mask[:,-1] = orientation_diff
    diff = diff * mask
    mse_loss = (diff**2).detach().cpu().numpy()
    return np.mean(mse_loss, axis=0)


def main(args):

    writer = SummaryWriter(comment="_{}_{}".format(args.features, args.training_set_size))

    if args.training_set_size == "low":
        dataset_fraction = 1/3.0
    elif args.training_set_size == "medium":
        dataset_fraction = 2/3.0
    elif args.training_set_size == "high":
        dataset_fraction = 1
    else:
        print("training_set_size specified is not one of (low, medium, high)... using all training data")
        dataset_fraction = 1

    if args.pretrained:
        # Use ImageNet stats, for both feature desc and depth image networks
        stats = {'mean': [0.485, 0.456, 0.406],
                 'std': [0.229, 0.224, 0.225]}
    elif args.features != 'priya':
        # Using depth images and not pretrained: single channel input
        # Mean and std dev of depth image train dataset
        with open('stats_pre_none.json') as f:
            stats = json.load(f)
    else:
        # Feature descriptors and not pretrained
        with open('stats_POST_priya.json') as f:
            stats = json.load(f)

    # before this transform, image needs to be PIL Image with values in between [0, 255]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(stats['mean'], stats['std'])
    ])

    rope_dataset = RopeTrajectoryDataset(args.data_dir, args.network_dir, args.network,
                                         cfg_dir=args.config, transform=transform, features=args.features, dataset_fraction=dataset_fraction, pretrained=args.pretrained)

    val_dataset = RopeTrajectoryDataset(args.val_dir, args.network_dir, args.network,
                                         cfg_dir=args.config, transform=transform, features=args.features, pretrained=args.pretrained)
    dataloader = DataLoader(rope_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers)

    model = ResNet18(args.pretrained, channels=1 if not args.pretrained and args.features == 'none' else 3)
    model = model.float().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #criterion = nn.MSELoss()
    criterion = mse_loss

    if args.weights is not None:
        # Load in weights to resume training
        model.load_state_dict(torch.load(args.weights))

    device = get_device(args.cuda)
    model.to(device)

    total_step = len(dataloader)
    print("total step:", total_step)
    best_validation_loss = None
    train_counter = 0
    validation_counter = 0
    split_losses_train = None
    split_losses_val = None
    for epoch in range(args.num_epoch):
        train_loss = 0.0
        for train_idx, (obs, targets, _) in enumerate(dataloader):
            # TODO: split into training and validation sets if dataset is big enough....
            model.train()
            optimizer.zero_grad()
            obs = obs.float()
            obs.to(device)

            pred = model(obs.cuda())
            t = torch.zeros(pred.shape)
            for i in range(0, len(targets)):
                t[:,i] = targets[i]
            targets = t.float()
            targets.to(device)

            loss = criterion(pred, targets.cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            manual_train_loss = mse_loss_separated(pred, targets.cuda())
            if split_losses_train is None:
                split_losses_train = manual_train_loss
            else:
                split_losses_train = np.vstack((split_losses_train, manual_train_loss))


            if train_idx % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epoch, train_idx, total_step, loss.item(), np.exp(loss.item())))

            if args.model_path is not None and (((train_idx+1) % args.save_step == 0) or (train_idx == total_step-1)):
                model.eval()
                validation_loss = 0.0
                for val_idx, (obs, targets, _) in enumerate(val_dataloader):
                    obs = obs.float()
                    obs.to(device)
                    pred = model(obs.cuda())
                    t = torch.zeros(pred.shape)
                    for i in range(0, len(targets)):
                        t[:,i] = targets[i]
                    targets = t.float()
                    targets.to(device)
                    loss = criterion(pred, targets.cuda())
                    validation_loss += loss.item()
                    manual_val_loss = mse_loss_separated(pred, targets.cuda())
                    if split_losses_val is None:
                        split_losses_val = manual_val_loss
                    else:
                        split_losses_val = np.vstack((split_losses_val, manual_val_loss))

                writer.add_scalar('Loss/validation', validation_loss / (val_idx+1), validation_counter)
                validation_counter += 1

                if best_validation_loss is None or (validation_loss / (val_idx+1)) < best_validation_loss:
                    print("Validation Loss at epoch {} and step {}: {}... Previous Best Validation Loss: {}... saving model".format(epoch,
                                                                                                                                    train_idx,
                                                                                                                                    validation_loss / (val_idx+1),
                                                                                                                                    best_validation_loss,
                                                                                                                                    ))
                    save_folder = os.path.join(args.model_path, args.features, args.training_set_size)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_path = os.path.join(save_folder, 'bc_model-{}-{}.ckpt'.format(epoch+1, train_idx+1))
                    torch.save(model.state_dict(), save_path)
                    best_validation_loss = validation_loss / (val_idx+1)

        writer.add_scalar('Loss/train', train_loss / (train_idx+1), train_counter)
        train_counter += 1
        print('Train loss: -----epoch {}----- : {}'.format(epoch, train_loss / (train_idx+1)))
        print('Validation loss:               : {}'.format(validation_loss / (val_idx+1)))
    np.save(os.path.join(vars(writer)['log_dir'], 'train_losses.npy'), split_losses_train)
    np.save(os.path.join(vars(writer)['log_dir'], 'val_losses.npy'), split_losses_val)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    # vis-descriptor specific file paths
    parser.add_argument('--config', default='cfg', type=str,
                      help='path to config')
    parser.add_argument('--data_dir', default='data/train/', type=str,
                      help='directory to training data')
    parser.add_argument('--val_dir', default='data/val/', type=str,
                      help='directory to validation data')
    parser.add_argument('--network_dir', default='/nfs/diskstation/priya/rope_networks/', type=str,
                      help='directory to vis_descriptor network')
    parser.add_argument('--network', default='rope_noisy_1400_depth_norm_3', type=str,
                      help='filename of vis_descriptor network')
    
    # network-specific arguments
    parser.add_argument('--num_workers', default=4, type=int,
                  help='number of workers')
    parser.add_argument('--batch_size', default=32, type=int,
                  help='batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                  help='number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                  help='learning rate')
    parser.add_argument('--model_path', default='bc_model/', type=str,
                  help='save model location')
    parser.add_argument('--pretrained', action='store_true',
                    help='If using depth images, flag to use pretrained model')

    # logging arguments
    parser.add_argument('--log_step', default=2, type=int,
                  help='frequency to log')
    parser.add_argument('--save_step', default=2, type=int,
                  help='frequency to save')
    
    # training specific arguments
    parser.add_argument('--features', default='priya', type=str,
                      help='what feature type goes into the pipeline')
    parser.add_argument('--training_set_size', default='high', type=str,
                      help='size of training set (low, medium, or high)')
    parser.add_argument('--weights', default=None, type=str,
                        help='path to weights file to resume training from. \
                        None if start from scratch')
    parser.add_argument('--cuda', action='store_true',
                        help='use this flag to run on GPU')

    args = parser.parse_args()
    main(args)
