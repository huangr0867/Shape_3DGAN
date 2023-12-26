'''
trainer.py

Train 3dgan models
'''

import torch
from torch import optim
from torch import nn
from utils import *
import os

from model import Generator, Discriminator

# added
import datetime
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import constants
from tqdm import tqdm


def save_train_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['train_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['train_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def save_val_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['val_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['val_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def trainer(args):
    # added for output dir
    save_file_path = constants.OUTPUT_PATH
    print(save_file_path)  # ../outputs/dcgan
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    # for using tensorboard
    if args.logs:
        writer = SummaryWriter(constants.OUTPUT_PATH + '/' + args.logs + '/logs')

        image_saved_path = constants.OUTPUT_PATH + '/' + args.logs + '/images'
        model_saved_path = constants.OUTPUT_PATH + '/' + args.logs + '/models'

        if not os.path.exists(image_saved_path):
            os.makedirs(image_saved_path)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)

    # datset define
    dsets_path = constants.DATASET_PATH

    print(dsets_path)

    train_dsets = ShapeDataset(dsets_path, args, "train")
    # val_dsets = ShapeNetDataset(dsets_path, args, "val")

    train_dset_loaders = torch.utils.data.DataLoader(train_dsets, batch_size=32, shuffle=True,
                                                     num_workers=1)
    # val_dset_loaders = torch.utils.data.DataLoader(val_dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    dset_len = len(train_dsets)
    # print (dset_len["train"])

    # model define
    D = Discriminator(args)
    G = Generator(args)

    D_solver = optim.Adam(D.parameters(), lr=constants.D_LR, betas=constants.BETA)
    # D_solver = optim.SGD(D.parameters(), lr=args.d_lr, momentum=0.9)
    G_solver = optim.Adam(G.parameters(), lr=constants.G_LR, betas=constants.BETA)

    D.to(constants.DEVICE)
    G.to(constants.DEVICE)

    # criterion_D = nn.BCELoss()
    criterion_D = nn.MSELoss()

    criterion_G = nn.L1Loss()

    itr_val = -1
    itr_train = -1

    for epoch in range(constants.EPOCHS):

        start = time.time()

        D.train()
        G.train()

        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_adv_G = 0.0

        phase = 'train'

        for i, X in enumerate(tqdm(train_dset_loaders)):

            if phase == 'train':
                itr_train += 1

            X = X.to(constants.DEVICE)


            batch = X.size()[0]

            Z = generateZ(args, batch)
                # print (Z.size())

            # ============= Train the discriminator =============#
            d_real = D(X)

            fake = G(Z)
            d_fake = D(fake)

            real_labels = torch.ones_like(d_real).to(constants.DEVICE)
            fake_labels = torch.zeros_like(d_fake).to(constants.DEVICE)

            d_real_loss = criterion_D(d_real, real_labels)

            d_fake_loss = criterion_D(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss

            # no deleted
            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            if d_total_acu < constants.D_THRESHOLD:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            # =============== Train the generator ===============#

            Z = generateZ(args, batch)

            fake = G(Z)  # generated fake: 0-1, X: 0/1
            d_fake = D(fake)

            adv_g_loss = criterion_D(d_fake, real_labels)


            recon_g_loss = criterion_G(fake, X)

            g_loss = adv_g_loss

            if args.local_test:
                print('Iteration-{} , D(x) : {:.4}, D(G(x)) : {:.4}'.format(itr_train, d_loss.item(),
                                                                                adv_g_loss.item()))

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()

            # =============== logging each 10 iterations ===============#

            running_loss_G += recon_g_loss.item() * X.size(0)
            running_loss_D += d_loss.item() * X.size(0)
            running_loss_adv_G += adv_g_loss.item() * X.size(0)

            if args.logs:
                loss_G = {
                    'adv_loss_G': adv_g_loss,
                    'recon_loss_G': recon_g_loss,
                }

                loss_D = {
                    'adv_real_loss_D': d_real_loss,
                    'adv_fake_loss_D': d_fake_loss,
                }


                if itr_train % 10 == 0 and phase == 'train':
                    save_train_log(writer, loss_D, loss_G, itr_train)

        # =============== each epoch save model or save image ===============#
        epoch_loss_G = running_loss_G / dset_len
        epoch_loss_D = running_loss_D / dset_len
        epoch_loss_adv_G = running_loss_adv_G / dset_len

        end = time.time()
        epoch_time = end - start

        print('Epochs-{} ({}) , D(x) : {:.4}, D(G(x)) : {:.4}'.format(epoch, phase, epoch_loss_D, epoch_loss_adv_G))
        print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

        if (epoch + 1) % constants.SAVE_MODEL_ITER == 0:
            print('model_saved, images_saved...')
            torch.save(G.state_dict(), model_saved_path + '/G.pth')
            torch.save(D.state_dict(), model_saved_path + '/D.pth')

            samples = fake.cpu().data[:8].squeeze().numpy()

            saveGeneratedShape(samples, image_saved_path, epoch)

            