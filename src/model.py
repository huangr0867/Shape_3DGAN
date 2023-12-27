import torch
import torch.nn as nn
import constants

class Generator(nn.Module):
    def __init__(self, args):
        """
        Generator class for the GAN model.

        Args:
            args (argparse.Namespace): Command-line arguments.
        """
        
        super(Generator, self).__init__()
        self.args = args
        self.cube_len = constants.CUBE_LEN
        self.bias = False
        self.z_dim = constants.Z_DIM
        self.f_dim = 32

        self.main = nn.Sequential(
            nn.ConvTranspose3d(self.z_dim, self.f_dim*8, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias),
            nn.BatchNorm3d(self.f_dim*8),
            nn.ReLU(True),

            nn.ConvTranspose3d(self.f_dim*8, self.f_dim*4, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias),
            nn.BatchNorm3d(self.f_dim*4),
            nn.ReLU(True),

            nn.ConvTranspose3d(self.f_dim*4, self.f_dim*2, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias),
            nn.BatchNorm3d(self.f_dim*2),
            nn.ReLU(True),

            nn.ConvTranspose3d(self.f_dim*2, self.f_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias),
            nn.BatchNorm3d(self.f_dim),
            nn.ReLU(True),

            nn.ConvTranspose3d(self.f_dim, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1,1,1)),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        out = x.view(-1, self.z_dim, 1, 1, 1)
        out = self.main(out)
        out = torch.squeeze(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        """
        Discriminator class for the GAN model.

        Args:
            args (argparse.Namespace): Command-line arguments.
        """

        super(Discriminator, self).__init__()
        self.args = args
        self.cube_len = constants.CUBE_LEN
        self.leak_value = constants.LEAK_VALUE
        self.bias = False

        self.f_dim = 32

        self.main = nn.Sequential(
            nn.Conv3d(1, self.f_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias),
            nn.BatchNorm3d(self.f_dim),
            nn.LeakyReLU(self.leak_value, inplace=True),

            nn.Conv3d(self.f_dim, self.f_dim*2, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias),
            nn.BatchNorm3d(self.f_dim*2),
            nn.LeakyReLU(self.leak_value, inplace=True),

            nn.Conv3d(self.f_dim*2, self.f_dim*4, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias),
            nn.BatchNorm3d(self.f_dim*4),
            nn.LeakyReLU(self.leak_value, inplace=True),

            nn.Conv3d(self.f_dim*4, self.f_dim*8, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias),
            nn.BatchNorm3d(self.f_dim*8),
            nn.LeakyReLU(self.leak_value, inplace=True),

            nn.Conv3d(self.f_dim*8, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1,1,1)),
            nn.Sigmoid()

        )

    def forward(self, x):
        out = x.view(-1, 1, self.cube_len, self.cube_len, self.cube_len)
        out = self.main(out)
        out = torch.squeeze(out)
        return out

