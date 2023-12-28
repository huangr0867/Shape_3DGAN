import torch
from torch import optim
from torch import nn
from utils import *
import os

from model import Generator, Discriminator

import datetime
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import constants
from tqdm import tqdm

def trainer(args):
    """
    Main function for training the Generative Adversarial Network (GAN).

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    
    # Set output directory
    save_file_path = constants.OUTPUT_PATH
    print("output directory: " + save_file_path)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    # Set up logging directories
    if args.log:
        image_saved_path = constants.OUTPUT_PATH + '/images'
        model_saved_path = constants.OUTPUT_PATH + '/models'

        if not os.path.exists(image_saved_path):
            os.makedirs(image_saved_path)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)

    # Dataset
    dsets_path = constants.DATASET_PATH
    print("dataset directory: " + dsets_path)
    train_dsets = ShapeDataset(dsets_path, args, "train")
    train_dset_loaders = torch.utils.data.DataLoader(train_dsets, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=1)
    dset_len = len(train_dsets)

    # Model
    D = Discriminator(args)
    G = Generator(args)

    D_solver = optim.Adam(D.parameters(), lr=constants.D_LR, betas=constants.BETA)
    G_solver = optim.Adam(G.parameters(), lr=constants.G_LR, betas=constants.BETA)

    D.to(constants.DEVICE)
    G.to(constants.DEVICE)

    criterion_D = nn.MSELoss()

    criterion_G = nn.L1Loss()

    G_losses = []
    D_losses = []

    for epoch in range(args.epochs):

        start = time.time()

        D.train()
        G.train()

        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_adv_G = 0.0

        for train_iter, X in enumerate(tqdm(train_dset_loaders)):

            X = X.to(constants.DEVICE)

            batch = X.size()[0]

            Z = generateZ(args, batch)

            # Train the discriminator
            d_real = D(X)

            fake = G(Z)
            d_fake = D(fake)

            real_labels = torch.ones_like(d_real).to(constants.DEVICE)
            fake_labels = torch.zeros_like(d_fake).to(constants.DEVICE)

            d_real_loss = criterion_D(d_real, real_labels)

            d_fake_loss = criterion_D(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss

            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            if d_total_acu < constants.D_THRESHOLD:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            # Train the generator
            Z = generateZ(args, batch)

            fake = G(Z)
            d_fake = D(fake)

            adv_g_loss = criterion_D(d_fake, real_labels)

            recon_g_loss = criterion_G(fake, X)

            g_loss = adv_g_loss

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()

            # Logging losses
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())


        # Save model or save image for each epoch
        epoch_loss_G = running_loss_G / dset_len
        epoch_loss_D = running_loss_D / dset_len
        epoch_loss_adv_G = running_loss_adv_G / dset_len

        end = time.time()
        epoch_time = end - start
        phase = 'train'

        print('Epochs-{} ({}) , D(x) : {:.4}, D(G(x)) : {:.4}'.format(epoch, phase, epoch_loss_D, epoch_loss_adv_G))
        print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

        if (epoch + 1) % constants.SAVE_MODEL_ITER == 0:
            print('model_saved, images_saved...')
            torch.save(G.state_dict(), model_saved_path + '/G.pth')
            torch.save(D.state_dict(), model_saved_path + '/D.pth')

            samples = fake.cpu().data[:8].squeeze().numpy()

            saveGeneratedShape(samples, image_saved_path, epoch)

    # Save the losses plot
    loss_path = constants.OUTPUT_PATH + '/loss'
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_path + '/loss_during_training.png', bbox_inches='tight')
    plt.close()

            