
'''
utils.py

Some utility functions

'''

import scipy.ndimage as nd
import scipy.io as io
import matplotlib
import constants

if constants.DEVICE.type != 'cpu':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle



def getVoxelFromArray(path, cube_len=32):
    voxels = np.load(path, allow_pickle=True)['a'] # 30 x 30 x 30
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0)) # 32 x 32 x 32
    return voxels


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def saveGeneratedShape(voxels, path, iteration):
    voxels = voxels[:1].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.05, hspace=0.05)
    x, y, z = voxels[0].nonzero()
    ax = plt.subplot(gs[0], projection='3d')
    ax.set_aspect('equal')
    ax.scatter(x, y, z, zdir='z', c='blue')

    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()


class ShapeDataset(data.Dataset):
    def __init__(self, root, args, train_or_val="train"):
        self.root = root
        self.listdir = os.listdir(self.root)
        if '.DS_Store' in self.listdir:
            self.listdir.remove('.DS_Store')

        data_size = len(self.listdir)
        self.listdir = self.listdir[0:int(data_size)]
        
        print ('data_size =', len(self.listdir))
        self.args = args

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            volume = np.asarray(getVoxelFromArray(f, constants.CUBE_LEN), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)


def generateZ(args, batch):
    if constants.Z_DIS == "norm":
        Z = torch.Tensor(batch, constants.Z_DIM).normal_(0, 0.33).to(constants.DEVICE)
    elif constants.Z_DIS == "uni":
        Z = torch.randn(batch, constants.Z_DIM).to(constants.DEVICE).to(constants.DEVICE)
    else:
        print("z_dist is not normal or uniform")

    return Z
