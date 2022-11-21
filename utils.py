import sys
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset


class recorder():
    def __init__(self, total_epochs, iters_per_epoch):
        self.total_epochs = total_epochs
        self.iters_per_epoch = iters_per_epoch
        self.train_time = time.time()
        self.batch_time = 0
        self.loss = {'loss_G_identity':[0],'loss_G_GAN':[0],'loss_G_cycle':[0],'loss_G': [0],'loss_D_A':[0],'loss_D_B':[0],'loss_D':[0]}
        self.epoch_loss = {'loss_G_identity':[0],'loss_G_GAN':[0],'loss_G_cycle':[0],'loss_G': [0],'loss_D_A':[0],'loss_D_B':[0],'loss_D':[0]}

    def record(self, recordloss=None, epochs=1, iter=0, force_print=0):  # record every iteration
        if force_print == 0:
            sys.stdout.write(
                '\rEpoch: %d/%d Iterations: %03d/%03d' % (epochs, self.total_epochs, (iter + 1), self.iters_per_epoch))

        if iter == 0:  # begin of an epoch
            for name in self.loss.keys():
                self.loss[name].append(0)
        if force_print == 0:
            for name, lossvalue in self.loss.items():  # sum the loss
                self.loss[name][epochs] += recordloss[name].data.cpu().numpy()

        if iter == self.iters_per_epoch-1 or force_print == 1: # end of an epoch
            for name, lossvalue in self.loss.items():
                self.loss[name][epochs] = self.loss[name][epochs] / self.iters_per_epoch

            plt.plot(range(1, len(self.loss['loss_G_identity'])), self.loss['loss_G_identity'][1:], 'r')
            plt.plot(range(1, len(self.loss['loss_G_GAN'])), self.loss['loss_G_GAN'][1:], 'y')
            plt.plot(range(1, len(self.loss['loss_G_cycle'])), self.loss['loss_G_cycle'][1:], 'b')
            plt.plot(range(1, len(self.loss['loss_G'])), self.loss['loss_G'][1:], 'k')
            plt.title("Loss vs Number of epochs (Generators)")
            plt.xlabel("Number of epochs")
            plt.ylabel("Loss")
            plt.legend(['loss_G_identity','loss_G_GAN', 'loss_G_cycle', 'loss_G'])
            plt.savefig('Result/train/saveloss/Loss_G.png', bbox_inches='tight')
            plt.clf()

            plt.plot(range(1, len(self.loss['loss_D_A'])), self.loss['loss_D_A'][1:], 'r')
            plt.plot(range(1, len(self.loss['loss_D_B'])), self.loss['loss_D_B'][1:], 'b')
            plt.plot(range(1, len(self.loss['loss_D'])), self.loss['loss_D'][1:], 'k')
            plt.title("Loss vs Number of epochs (Discriminators)")
            plt.xlabel("Number of epochs")
            plt.ylabel("Loss")
            plt.legend(['loss_D_A','loss_D_B','loss_D'])
            plt.savefig('Result/train/saveloss/Loss_D.png', bbox_inches='tight')
            plt.clf()

            self.batch_time = time.time() - self.train_time

            print('\rEpoch: %d/%d Iterations: %d Time:%.3f loss_G:%.3f loss_D:%.3f ' % (
            epochs, self.total_epochs, self.iters_per_epoch, self.batch_time, self.loss['loss_G'][epochs],
            self.loss['loss_D'][epochs]))

            self.train_time = time.time()


def tensor2image(image_tensor):
    image_np = image_tensor[0].cpu().float().numpy()  # Mapping from(-1,1) to (0,255)
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = (image_np + 1.0)/ 2.0 * 255.0
    return image_np.astype(np.uint8)


class allocate():
    def __init__(self, max_size=50):
        self.datalist = []
        self.max_size = max_size

    def stack(self, input_data):
        return_data = []
        for unit in input_data.data:
            if len(self.datalist) < self.max_size:
                unit = torch.unsqueeze(unit, 0)
                self.datalist.append(unit)
                return_data.append(unit)
            else:
                unit = torch.unsqueeze(unit, 0)
                if random.uniform(0,2) > 1:
                    t = random.randint(0, self.max_size-1)
                    return_data.append(self.datalist[t].clone())
                    self.datalist[t] = unit
                else:
                    return_data.append(unit)
        return Variable(torch.cat(return_data))


class CustomImgDataset(Dataset):
    def __init__(self, root_dir, transform=None, data_mode='Unaligned', train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.data_mode = data_mode
        self.paths_A, self.paths_B = self._load_data()

    def _load_data(self):
        imageA_file = f"{'train' if self.train else 'test'}A"
        imageB_file = f"{'train' if self.train else 'test'}B"
        file_A = os.path.join(self.root_dir, imageA_file)
        file_B = os.path.join(self.root_dir, imageB_file)
        assert(os.path.isdir(file_A))
        assert(os.path.isdir(file_B))
        image_setA = []
        image_setB= []
        for (root, dir, files) in sorted(os.walk(file_A)):
            for file in files:
                path = os.path.join(root, file)
                image_setA.append(path)
        for (root, dir, files) in sorted(os.walk(file_B)):
            for file in files:
                path = os.path.join(root, file)
                image_setB.append(path)
        return sorted(image_setA), sorted(image_setB)

    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))

    def __getitem__(self, idx):
        if self.data_mode == 'aligned':  # if choose aligned mode, modify image file names for better sort
            assert(len(self.paths_A) == len(self.paths_B))
        imA = Image.open(self.paths_A[idx % len(self.paths_A)]).convert('RGB')
        idx_B = idx
        if self.data_mode == 'Unaligned':
            idx_B = random.randint(0, len(self.paths_B) - 1)
        imB = Image.open(self.paths_B[idx_B % len(self.paths_B)]).convert('RGB')
        imageA, imageB = self.transform(imA), self.transform(imB)
        return {'A': imageA, 'B': imageB}


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
















