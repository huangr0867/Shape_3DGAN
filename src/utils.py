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
    """
    Load voxel data from a file and pad it to the specified cube size.

    Args:
        path (str): Path to the file containing voxel data.
        cube_len (int): Desired cube size.

    Returns:
        np.ndarray: Padded voxel data.
    """

    voxels = np.load(path, allow_pickle=True)['a'] # 30 x 30 x 30
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0)) # 32 x 32 x 32
    return voxels

def saveGeneratedShape(voxels, path, iteration):
    """
    Save a 3D scatter plot of generated voxel shapes.

    Args:
        voxels (np.ndarray): Generated voxel data.
        path (str): Path to save the visualization.
        iteration (int): Iteration number for file naming.
    """
    
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
        """
        Dataset class for loading voxel data.

        Args:
            root (str): Root directory containing voxel data files.
            args (argparse.Namespace): Command-line arguments.
            train_or_val (str): Specifies whether the dataset is for training or validation.
        """

        self.root = root
        self.listdir = os.listdir(self.root)
        if '.DS_Store' in self.listdir:
            self.listdir.remove('.DS_Store')

        data_size = len(self.listdir)
        self.listdir = self.listdir[0:int(data_size)]
        
        self.args = args

    def __getitem__(self, index):
        """
        Load voxel data for a given index.

        Args:
            index (int): Index of the voxel data file.

        Returns:
            torch.FloatTensor: Loaded voxel data as a PyTorch FloatTensor.
        """

        with open(self.root + self.listdir[index], "rb") as f:
            volume = np.asarray(getVoxelFromArray(f, constants.CUBE_LEN), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """

        return len(self.listdir)


def generateZ(args, batch):
    """
    Generate random noise vectors for the generator.

    Args:
        args (argparse.Namespace): Command-line arguments.
        batch (int): Number of noise vectors to generate.

    Returns:
        torch.Tensor: Generated noise vectors.
    """

    if constants.Z_DIS == "norm":
        Z = torch.Tensor(batch, constants.Z_DIM).normal_(0, 0.33).to(constants.DEVICE)
    elif constants.Z_DIS == "uni":
        Z = torch.randn(batch, constants.Z_DIM).to(constants.DEVICE).to(constants.DEVICE)
    else:
        print("z_dist is not normal or uniform")

    return Z
