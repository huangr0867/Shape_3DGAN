import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import *
import os
from model import Generator, Discriminator

import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import constants
import visdom

def generate_from_pretrained(args):

    image_saved_path = constants.OUTPUT_PATH + '/pretrained_generated'
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)
    
    pretrained_file_path_G = constants.OUTPUT_PATH + '/models/G.pth'
    pretrained_file_path_D = constants.OUTPUT_PATH + '/models/D.pth'

    D = Discriminator(args)
    G = Generator(args)

    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))

    G.to(constants.DEVICE)
    D.to(constants.DEVICE)
    G.eval()
    D.eval()

    N = 200

    for i in range(N):

        z = generateZ(args, 1)

        fake = G(z)
        samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
        y_prob = D(fake)
        y_real = torch.ones_like(y_prob)
        saveGeneratedShapeVoxel(samples, image_saved_path, 'pretrained_generated' + str(i))
        
        # total_dist = 0
        # for i in range(len(xi)):
        #     distance_from_center = distance(xi[i], yi[i], zi[i], (15,15,15))
        #     total_dist += distance_from_center
        
        # avg_radius = total_dist/len(xi)
        # print('average radius:' + str(avg_radius))

        # radius_path = os.path.join(constants.OUTPUT_PATH, 'avg_radius')

        # # Ensure the directory exists
        # os.makedirs(radius_path, exist_ok=True)

        # rmse_file_path = os.path.join(radius_path, 'avg_radius.txt')

        # # Write or append the RMSE value to the file
        # with open(rmse_file_path, 'a') as f:
        #     if os.path.getsize(rmse_file_path) == 0:
        #         f.write(str(avg_radius))
        #     else:
        #         f.write('\n' + str(avg_radius))