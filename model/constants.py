import torch

EPOCHS = 500
G_LR = 0.0025 # generator learning rate
D_LR = 0.00001 # discrminator learning rate
BETA = (0.5, 0.999)
Z_DIS = "norm" # distribution
Z_DIM = 200
LEAK_VALUE = 0.2
CUBE_LEN = 32 # change based on size of radius/shape
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_PATH = '../data/generated'
D_THRESHOLD = 0.8
SAVE_MODEL_ITER = 1 # iteration to save trained model
DATASET_PATH = '/Users/ryanhuang/Desktop/Code/Shape_3DGAN/shape_data/sphere/1000pts'