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
from torch.utils.tensorboard import SummaryWriter

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


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

    rope_dataset = RopeTrajectoryDataset(args.data_dir, args.network_dir, args.network, 
                                         cfg_dir=args.config, transform=None, features=args.features, dataset_fraction=dataset_fraction)

    val_dataset = RopeTrajectoryDataset(args.val_dir, args.network_dir, args.network, 
                                         cfg_dir=args.config, transform=None, features=args.features)
    dataloader = DataLoader(rope_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.num_workers)

    model = BasicModel()
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    if args.weights is not None:
        # Load in weights to resume training
        model.load_state_dict(torch.load(args.weights))


    total_step = len(dataloader)
    print("total step:", total_step)
    best_validation_loss = None
    train_counter = 0
    validation_counter = 0
    for epoch in range(args.num_epoch):
        train_loss = 0
        for idx, (obs, targets) in enumerate(dataloader):
            # TODO: split into training and validation sets if dataset is big enough....
            model.train()
            optimizer.zero_grad()
            pred = model(obs.float())
            t = torch.zeros(pred.shape)
            for i in range(0, len(targets)):
                t[:,i] = targets[i]
            targets = t.float()

            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if idx % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epoch, idx, total_step, loss.item(), np.exp(loss.item())))
            
            if args.model_path is not None and (((idx+1) % args.save_step == 0) or (idx == total_step-1)): 
                model.eval()
                validation_loss = 0
                for idx, (obs, targets) in enumerate(val_dataloader):
                    pred = model(obs.float())
                    t = torch.zeros(pred.shape)
                    for i in range(0, len(targets)):
                        t[:,i] = targets[i]
                    targets = t.float()
                    loss = criterion(pred, targets)
                    validation_loss += loss.item()
                writer.add_scalar('Loss/validation', validation_loss, validation_counter)
                validation_counter += 1

                if best_validation_loss is None or validation_loss < best_validation_loss:
                    print("Validation Loss at epoch {} and step {}: {}... Previous Best Validation Loss: {}... saving model".format(epoch,
                                                                                                                                    idx,
                                                                                                                                    validation_loss,
                                                                                                                                    best_validation_loss,
                                                                                                                                    ))
                    save_path = os.path.join(args.model_path, args.features, args.training_set_size, 'bc_model-{}-{}.ckpt'.format(epoch+1, idx+1))
                    torch.save(model.state_dict(), save_path)
                    best_validation_loss = validation_loss     

        writer.add_scalar('Loss/train', train_loss, train_counter)
        train_counter += 1
        print('Train loss: -----epoch {}----- : {}'.format(epoch, train_loss))    
    
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
    parser.add_argument('--learning_rate', default=0.0001, type=int,
                  help='learning rate')
    parser.add_argument('--model_path', default='bc_model/', type=str,
                  help='save model location')

    # logging arguments
    parser.add_argument('--log_step', default=2, type=int,
                  help='frequency to log')
    parser.add_argument('--save_step', default=2, type=int,
                  help='frequency to save')
    
    # training specific arguments
    parser.add_argument('--features', default='priya', type=str,
                      help='what feature type goes into the pipeline')
    parser.add_argument('--training_set_size', default='high', type=str,
                      help='size of training set (low, med, or high)')
    parser.add_argument('--weights', default=None, type=str,
                        help='path to weights file to resume training from. \
                        None if start from scratch')

    args = parser.parse_args()
    main(args)
