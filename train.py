import argparse
import collections
import torch
import numpy as np

from behavioral_cloning.data_loader.data_loaders import RopeTrajectoryDataset
from behavioral_cloning.model.model import BasicModel

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args):
    rope_dataset = RopeTrajectoryDataset(args.data_dir, args.network_dir, args.network, 
                                         cfg_dir=args.config, transform=None)
    dataloader = DataLoader(rope_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers)
    model = BasicModel()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(args.num_epoch):
        model.train()
        train_loss = 0
        for idx, (obs, target) in enumerate(dataloader):
            # TODO: split into training and validation sets if dataset is big enough....
            optimizer.zero_grad()
            pred = model(obs)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]

            if idx % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
            # TODO: save on validation
            if args.model_path is not None and (((i+1) % args.save_step == 0) or (i == total_step-1)): 
                torch.save(model.state_dict(), os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch+1, i+1)))
        print('Train loss: -----epoch {}----- : {}'.format(epoch, train_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    # vis-descriptor specific file paths
    parser.add_argument('--config', default='cfg', type=str,
                      help='path to config')
    parser.add_argument('--data_dir', default='data/', type=str,
                      help='directory to data')
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
    parser.add_argument('--model_path', default=None, type=int,
                  help='save model location')

    # logging arguments
    parser.add_argument('--log_step', default=None, type=int,
                  help='frequency to log')
    parser.add_argument('--save_step', default=None, type=int,
                  help='frequency to save')
    args = parser.parse_args()
    main(args)
