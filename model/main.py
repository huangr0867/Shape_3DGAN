import argparse
from trainer import trainer
from generate import generate_from_pretrained
import torch


def str2bool(string):
    """
    Convert a string to a boolean value.

    Args:
        string (str): Input string to be converted to boolean.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid boolean representation.
    """
    string = string.lower()
    if string in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif string in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    """
    The main function to parse command-line arguments and execute the training program.
    """
    
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Logging parameters
    parser.add_argument('--log', type=str2bool, default=True, help='log training epochs')
    parser.add_argument('--loss', type=str2bool, default=True, help='record losses')
    parser.add_argument('--pretrained', type=str2bool, default=False, help='record losses')
    parser.add_argument('--batch_size', type=float, default=2, help='batch size during training')
    parser.add_argument('--epochs', type=float, default=1000, help='training epochs')

    # Parse command-line arguments
    args = parser.parse_args()
    print(args)

    # Run the training program with the parsed arguments
    if args.pretrained:
        generate_from_pretrained(args)
    else:
        trainer(args)


if __name__ == '__main__':
    main()
