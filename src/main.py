'''
main.py

Welcome, this is the entrance to 3dgan
'''

import argparse
from trainer import trainer
import torch

from tester import tester


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
    parser.add_argument('--logs', type=str, default='log', help='logs by tensorboardX')
    parser.add_argument('--local_test', type=str2bool, default=False, help='local test verbose')
    parser.add_argument('--test', type=str2bool, default=False, help='call tester.py')
    parser.add_argument('--use_visdom', type=str2bool, default=False, help='visualization by visdom')
    args = parser.parse_args()
    print(args)

    # list params
    # params.print_params()

    # run program
    if not args.test:
        trainer(args)
    else:
        tester(args)


if __name__ == '__main__':
    main()
