
import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet


import pickle
import h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

from train_func import train
from test_func import test



import sys

class Config:
    image_size = (32,32)        # Image size
    channels = 1           # Image channels
    cond_dim = 16850        # Condition dimension
    ts_feature = [337,20]
    batch_size = 16        # Batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps=1e-8
    train_num = [0,1000]    # Training data index range
    test_num = [1000,1100]  # Test data index range
    beta = 'linear'




def get_params():
    parser = argparse.ArgumentParser(description='test for flow matching model')
    parser.add_argument('--batchsize', type=int, default=Config.batch_size,
                        help='batch size per device for training Unet model')
    parser.add_argument('--numworkers', type=int, default=1, help='num workers for training Unet model')
    parser.add_argument('--inch', type=int, default=1, help='input channels for Unet model')
    parser.add_argument('--modch', type=int, default=64, help='model channels for Unet model')


    parser.add_argument('--outch', type=int, default=1, help='output channels for Unet model')
    parser.add_argument('--chmul', type=list, default=[1,2,4], help='architecture parameters training Unet model')
    parser.add_argument('--numres', type=int, default=2, help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim', type=int, default=Config.cond_dim, help='dimension of conditional embedding')

    parser.add_argument('--tdim', type=int, default=256, help='dimension of latent embedding')

    parser.add_argument('--useconv', type=bool, default=False, help='whether use convlution in downsample')
    parser.add_argument('--droprate', type=float, default=0.3, help='dropout rate for model')
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--w', type=float, default=1, help='hyperparameters for classifier-free guidance strength')

    parser.add_argument('--epoch', type=int, default=500, help='epochs for training')

    parser.add_argument('--multiplier', type=float, default=2.5, help='multiplier for warmup')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for classifier-free guidance')
    parser.add_argument('--interval', type=int, default=20, help='epoch interval between two evaluations')

    parser.add_argument('--moddir', type=str, default='model', help='model addresses')
    parser.add_argument('--samdir', type=str, default='sample', help='sample addresses')
    parser.add_argument('--genbatch', type=int, default=5, help='batch size for sampling process')
    parser.add_argument('--clsnum', type=int, default=100, help='num of label classes')

    # FMIM uses ODE integration steps
    parser.add_argument('--num_steps', type=int, default=30, help='ODE integration steps for FMIM')

    # FMIM noise level parameter
    parser.add_argument('--sigma', type=float, default=1.0, help='noise level for FMIM')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()
    return args


# 1. Data module
class SyntheticConditionalDataset(Dataset):

    def __init__(self, data_norm, cond_norm, ind=[0, 100]):
        self.images = data_norm[ind[0]:ind[1]]  # Use images in the index range
        self.conditions = cond_norm[ind[0]:ind[1]]  # Use conditions in the index range

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.conditions[idx]


def create_dataloader(data_norm, cond_norm):
    dataset = SyntheticConditionalDataset(data_norm, cond_norm, ind=Config.train_num)
    return DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )


def create_dataloader2(data_norm, cond_norm):
    dataset = SyntheticConditionalDataset(data_norm, cond_norm, ind=Config.test_num)
    return DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )


def initial():

    rootpath = os.path.dirname(os.path.abspath(__file__))

    path = '...'

    with h5py.File(path + '/train_data.h5', 'r') as f:
        data_ = f['x'][:]
        cond = f['y'][:]

    if 0:
        redu = dim_reduction.pca(data_, 1024)
        with open('redu.pkl', 'wb') as f:
            pickle.dump(redu, f)
    else:
        with open(path+'/redu.pkl', 'rb') as f:
            redu = pickle.load(f)
    data = redu.encode(data_)

    print(f'data shape after pca: {data.shape}')

    print(f'data shape: {data.shape}, cond shape: {cond.shape}')


    with h5py.File(rootpath + '/data/ini_data.h5', 'r') as f:
        dobstrue = f['dobstrue'][:]  # Assume 'dobstrue' is the key of the dataset
        true_para = f['true_para'][:]  # Assume 'true_para' is the key of the label
        ACT = f['ACT'][:]  # Assume 'ACT' is the key of the label
    true_para = redu.encode(true_para.reshape(1, -1))  # PCA encoding for real parameters

    dobstrue = torch.tensor(dobstrue.reshape(1, -1), dtype=torch.float32)  # Convert to PyTorch tensor
    true_para = torch.tensor(true_para.reshape(1, Config.channels, Config.image_size[1], Config.image_size[0]), \
                             dtype=torch.float32)  # Convert to PyTorch tensor

    # Normalization
    data = data.reshape(data.shape[0], Config.channels, Config.image_size[1], Config.image_size[0])

    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    data_norm = (data - data_min) / (data_max - data_min)  # Normalize to [0,1]
    data_norm = np.nan_to_num(data_norm, nan=0.0)  # Replace NaN with 0
    data_norm = np.clip(data_norm, 0, 1)  # Ensure data is in [0,1] range

    data_norm = data_norm * 2 - 1  # Normalize to [-1,1]
    data_norm = torch.tensor(data_norm, dtype=torch.float32)  # Convert to PyTorch tensor

    cond_max = np.max(cond, axis=0)
    cond_min = np.min(cond, axis=0)

    cond_norm = (cond - np.min(cond, axis=0)) / (np.max(cond, axis=0) - np.min(cond, axis=0)+ Config.eps)  # Normalize to [0,1]
    cond_norm = np.nan_to_num(cond_norm, nan=0.0)  # Replace NaN with 0
    cond_norm = np.clip(cond_norm, 0, 1)  # Ensure data is in [0,1] range

    cond_norm = cond_norm * 2 - 1  # Normalize to [-1,1]
    cond_norm = torch.tensor(cond_norm, dtype=torch.float32)  # Convert to PyTorch tensor

    print(f'data_norm max: {data_norm.max()}, data min: {data_norm.min()}')
    print(f'cond_norm max: {cond_norm.max()}, cond min: {cond_norm.min()}')

    fig, ax = plt.subplots()
    ax.plot(dobstrue[0], 'r')
    ax.plot(cond_min, 'b')
    ax.plot(cond_max, 'b')
    fig.savefig('dobstrue.png')

    fig, ax = plt.subplots(1, 2)
    im = ax[0].imshow(data_norm[0, 0, :, :], cmap='jet')
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    im = ax[1].imshow(data_norm[1, 0, :, :], cmap='jet')
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    fig.savefig(rootpath + '/data/data_norm.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots()
    for i in range(9):
        ax.plot(cond_norm[i, :])
    fig.savefig(rootpath + '/data/cond_norm.png', dpi=300, bbox_inches='tight')
    # ax.legend()

    # Normalize real data again
    dobstrue_ = np.clip(dobstrue, cond_min, cond_max)  # Ensure data is in [0,1] range
    dobstrue_ = (dobstrue_ - cond_min) / (cond_max - cond_min+ Config.eps)
    dobstrue_ = np.nan_to_num(dobstrue_, nan=0.0)  # Replace NaN with 0
    dobstrue_ = np.clip(dobstrue_, 0, 1)  # Ensure data is in [0,1] range

    dobstrue_norm = dobstrue_ * 2 - 1  # Normalize to [-1,1]

    dataloader = create_dataloader(data_norm, cond_norm)
    testloader = create_dataloader2(data_norm, cond_norm)

    true_para_norm = (true_para - data_min) / (data_max - data_min + Config.eps) * 2 - 1
    true_para_norm[true_para_norm > 1] = 1
    true_para_norm[true_para_norm < -1] = -1
    # true_para_norm = true_para

    initial_res = {}
    initial_res['dobstrue'] = dobstrue
    initial_res['dobstrue_norm'] = dobstrue_norm
    initial_res['true_para'] = true_para
    initial_res['true_para_norm'] = true_para_norm

    initial_res['data_norm'] = data_norm
    initial_res['cond_norm'] = cond_norm

    initial_res['data_max'] = data_max
    initial_res['data_min'] = data_min

    initial_res['cond_max'] = cond_max
    initial_res['cond_min'] = cond_min

    initial_res['rootpath'] = rootpath
    initial_res['path'] = path

    initial_res['dataloader'] = dataloader
    initial_res['testloader'] = testloader

    initial_res['ACT'] = ACT

    return initial_res

def main():
    initial_res = initial()
    rootpath = initial_res['rootpath']
    path = initial_res['path']
    dobstrue = initial_res['dobstrue']
    true_para = initial_res['true_para']

    dobstrue_norm = initial_res['dobstrue_norm']
    true_para_norm = initial_res['true_para_norm']

    data_norm = initial_res['data_norm']
    cond_norm = initial_res['cond_norm']

    data_max = initial_res['data_max']
    data_min = initial_res['data_min']

    cond_max = initial_res['cond_max']
    cond_min = initial_res['cond_min']

    dataloader = initial_res['dataloader']
    testloader = initial_res['testloader']
    config = Config()
    data_path = rootpath + '/data'

    mode = 1
    if mode == 0:
        # Training
        os.chdir(rootpath)
        args = get_params()
        t1 = time.time()
        history = train(args, dataloader, testloader,
                           dobstrue_norm, true_para_norm, data_max, data_min, config)
        t2 = time.time()
        print(f'training time: {t2 - t1:.2f} seconds')
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        val_ema_loss = history["val_ema_loss"]

        print(f"train loss min: {np.nanmin(train_loss):.6f}")
        print(f"val ema loss min: {np.nanmin(val_ema_loss):.6f}")

        plt.subplots(figsize=(5, 3))
        plt.plot(train_loss, label='Train Loss')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Training Loss: {min(train_loss):.4f}')
        plt.savefig(data_path + '/' + 'train_loss.png', dpi=300, bbox_inches='tight')

        print(f'#### training time: {t2 - t1:.2f} seconds')
###
    elif mode==1:
        os.chdir(rootpath)
        args = get_params()
        print(f'current path: {os.getcwd()}')
        generated_data = test(dataloader, dobstrue_norm, true_para_norm, data_max, data_min,
                              args, config)
        print(f'generated_data shape: {generated_data.shape}')
        final_inv_para = generated_data.cpu().numpy()
        final_inv_para = final_inv_para.reshape(final_inv_para.shape[0], -1)
        with h5py.File('post_data.h5', 'w') as f:
            f.create_dataset('data', data=final_inv_para)



if __name__ == '__main__':
    main()