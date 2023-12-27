'''
main.py

Welcome, this is the entrance to 3dgan
'''

import argparse
from trainer import trainer
import torch


def str2bool(string):
    string = string.lower()
    if string in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif string in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # add arguments
    parser = argparse.ArgumentParser()

    # loggings parameters
    parser.add_argument('--log', type=str2bool, default=True, help='log training epochs')
    parser.add_argument('--loss', type=str2bool, default=True, help='record losses')
    args = parser.parse_args()
    print(args)

    # run program
    trainer(args)


if __name__ == '__main__':
    main()
