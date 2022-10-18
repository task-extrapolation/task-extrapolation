import argparse
import os
import random
import string
import time
import numpy as np
import json
import sys

import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.functional as F

from utils_eval import *
from learning_module import *
from models.loc_rnn_eval import *

parser = argparse.ArgumentParser(description='Pathfinder zeroshot evaluation')
parser.add_argument('--test_maze_size', default='small', type=str, metavar='S',
                    help='size of test maze (default: small)')
parser.add_argument('-eval-single', '--eval-single', action='store_true')

##############################
#edit the path of checkpoints#
##############################
# 'model name_depth_width':'path' for rns and rrns / 'model name_depth_timesteps':'path" for locrnn, gru, and hgru
# for example: 'locrnn_32_15': 'check_default/locrnn_mazes_large_SGD_depth=32_width=1_lr=0.002_batchsize=50_epoch=49_log.pth'
checkpoints = {''
                }

def test(net, testloader, test_setup, device):
    problem = test_setup.problem
    mode = test_setup.mode
    if problem == "segment":
        if mode == "default":
            accuracy = test_mazes_default(net, testloader, device)
        else:
            print(f"Mode {mode} not yet implemented. Exiting.")
            sys.exit()
    else:
        print(f"Problem {problem} not yet implemented. Exiting.")
        sys.exit()

    return accuracy

def test_mazes_default(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:

            inputs, targets = inputs.to(device), targets.to(device)[:, 0, :, :].long()
            outputs = net(inputs)

            predicted = outputs.argmax(1) * inputs.max(1)[0]
            correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def validate(test_loader, model):
    with torch.no_grad():
        test_setup=TestingSetup('segment', 'default')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_acc = test(model, test_loader, test_setup, device)
    return test_acc

if __name__ == "__main__":
    args = parser.parse_args()

    #edit the absolute path for maze datasets when necessary
    maze_data = {'mazes_small': 'maze_data/test_small',
                'mazes_medium': 'maze_data/test_medium',
                'mazes_large': 'maze_data/test_large'
                }
    if args.eval_single:
            sizes = [args.test_maze_size]
    else:
        sizes = ['mazes_small', 'mazes_medium', 'mazes_large']
    
    for size in sizes:
        testdir = maze_data[size]
        dataset = size.upper()
        test_path = testdir
        test_inputs_np = np.load(os.path.join(test_path, "inputs.npy"))
        test_targets_np = np.load(os.path.join(test_path, "solutions.npy"))
        test_inputs = torch.from_numpy(test_inputs_np).float().permute(0, 3, 1, 2)
        test_targets = torch.from_numpy(test_targets_np).permute(0, 3, 1, 2)
        maze_size = {"MAZES_SMALL": 9,
                     "MAZES_MEDIUM": 11,
                     "MAZES_LARGE": 13}[dataset]
        testset = MazeDataset(test_inputs, test_targets, maze_size)
        testloader = data.DataLoader(testset, num_workers=4, batch_size=1024,
                                 shuffle=False, drop_last=False)
        
        for i in checkpoints:
            args.depth = int(i[i.index('_', -3, -1)+1:])
            args.width = int(i[i.index('_', -7, -1)+1:i.index('_', -3, -1)])
            args.model_name = i[0:i.index('_', -7, -1)]
            with torch.no_grad():
                if 'locrnn' in args.model_name:
                    #edit timestep group for LocRNN
                    for timesteps in (11, 13, 15, 17, 19, 21, 23, 24):
                        model, _, _ = load_model_from_checkpoint('locrnn', checkpoints[i], dataset, width=timesteps, depth=32)
                        test_acc = validate(testloader, model)
                        print('model_name:%s_timesteps:%s, dataset:%s, test_acc:%s'%(args.model_name, timesteps, size, test_acc))
                elif args.model_name.startswith('hgru'):
                    model, _, _ = load_model_from_checkpoint('hgru', checkpoints[i], dataset, width=15, depth=32)
                    test_acc = validate(testloader, model)
                    print('model_name:%s_timesteps:%s, dataset:%s, test_acc:%s'%(args.model_name, 15, size, test_acc))
                elif args.model_name.startswith('gru'):
                    #edit timestep group for GRU
                    for timesteps in (5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25):
                        model, _, _ = load_model_from_checkpoint('gru', checkpoints[i], dataset, width=timesteps, depth=32)
                        test_acc = validate(testloader, model)
                        print('model_name:%s_timesteps:%s, dataset:%s, test_acc:%s'%(args.model_name, timesteps, size, test_acc))
                elif 'recur' in args.model_name:
                    #edit timestep group for RRNS
                    for depths in (24, 28, 32, 36, 40, 44, 48):
                        model, _, _ = load_model_from_checkpoint('recur_residual_network_segment', checkpoints[i], dataset, width=args.width, depth=depths)
                        test_acc = validate(testloader, model)
                        print('model_name:%s_width%s_depth%s, dataset:%s, test_acc:%s'%(args.model_name, args.width, depths, size, test_acc))
                else:
                    model, _, optimizer_state_dict = load_model_from_checkpoint('residual_network_segment', checkpoints[i], dataset, width=args.width, depth=args.depth)  
                    test_acc = validate(testloader, model)
                    print('model_name:%s_width%s_depth%s, dataset:%s, test_acc:%s'%(args.model_name, args.width, args.depth, size, test_acc))   



    
        

        

        
            
            