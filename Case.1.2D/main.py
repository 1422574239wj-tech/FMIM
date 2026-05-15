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


# Configuration parameters
class Config:
    image_size = (60, 60)
    channels = 1
    cond_dim = 900
    ts_feature = [50, 18]
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = 1e-8
    train_num = [0, 900]
    test_num = [900, 1000]


# Custom Dataset Class (Missing in original code, added to fix runtime error)
class SyntheticConditionalDataset(Dataset):
    def __init__(self, data_norm, cond_norm, ind):
        self.data = data_norm[ind[0]:ind[1]]
        self.cond = cond_norm[ind[0]:ind[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.cond[idx]


def get_params():
    parser = argparse.ArgumentParser(description='test for flow matching model')
    parser.add_argument('--batchsize', type=int, default=Config.batch_size,
                        help='batch size per device for training Unet model')
    parser.add_argument('--numworkers', type=int, default=1, help='num workers for training Unet model')
    parser.add_argument('--inch', type=int, default=1, help='input channels for Unet model')
    parser.add_argument('--modch', type=int, default=64, help='model channels for Unet model')
    parser.add_argument('--outch', type=int, default=1, help='output channels for Unet model')
    parser.add_argument('--chmul', type=list, default=[1, 2, 4], help='architecture parameters training Unet model')
    parser.add_argument('--numres', type=int, default=2, help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim', type=int, default=Config.cond_dim, help='dimension of conditional embedding')
    parser.add_argument('--tdim', type=int, default=128, help='dimension of latent embedding')
    parser.add_argument('--useconv', type=bool, default=False, help='whether use convolution in downsample')
    parser.add_argument('--droprate', type=float, default=0.1, help='dropout rate for model')
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
    parser.add_argument('--clsnum', type=int, default=10, help='num of label classes')
    parser.add_argument('--num_steps', type=int, default=30, help='ODE integration steps for Flow Matching')
    parser.add_argument('--sigma', type=float, default=1, help='noise level for Flow Matching')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()
    return args


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
    path = '...'
    rootpath = os.path.dirname(os.path.abspath(__file__))

    # Load data
    data = np.load(path + '/grfv_perm1000.npz')['arr_0']
    cond = np.load(path + '/grfv_prior_data1000.npz')['arr_0'].reshape(-1, Config.cond_dim)

    print(f'data shape: {data.shape}, cond shape: {cond.shape}')
    data = np.log(data + 1)

    # Load true data
    print(f"Loading true data from: {path + '/data/ini_data.h5'}")
    with h5py.File(path + '/data/ini_data.h5', 'r') as f:
        dobstrue = f['dobstrue'][:]
        true_para = f['true_para'][:]

    true_para = np.log(true_para + 1)

    # Convert to tensor
    dobstrue = torch.tensor(dobstrue.reshape(1, -1), dtype=torch.float32)
    true_para = torch.tensor(true_para.reshape(1, Config.channels, Config.image_size[1], Config.image_size[0]),
                             dtype=torch.float32)

    # Reshape data
    data = data.reshape(data.shape[0], Config.channels, Config.image_size[1], Config.image_size[0])

    # Normalize data
    data_max = np.max(data)
    data_min = np.min(data)
    data_norm = (data - data_min) / (data_max - data_min)
    data_norm = np.nan_to_num(data_norm, nan=0.0)
    data_norm = np.clip(data_norm, 0, 1)
    data_norm = data_norm * 2 - 1
    data_norm = torch.tensor(data_norm, dtype=torch.float32)

    # Normalize condition
    cond_max = np.max(cond, axis=0)
    cond_min = np.min(cond, axis=0)
    cond_norm = (cond - np.min(cond, axis=0)) / (np.max(cond, axis=0) - np.min(cond, axis=0))
    cond_norm = np.nan_to_num(cond_norm, nan=0.0)
    cond_norm = np.clip(cond_norm, 0, 1)
    cond_norm = cond_norm * 2 - 1
    cond_norm = torch.tensor(cond_norm, dtype=torch.float32)

    print(f'data_norm max: {data_norm.max()}, data min: {data_norm.min()}')
    print(f'cond_norm max: {cond_norm.max()}, cond min: {cond_norm.min()}')

    # Plot true data comparison
    fig, ax = plt.subplots()
    ax.plot(dobstrue[0], 'r')
    ax.plot(cond_min, 'b')
    ax.plot(cond_max, 'b')
    fig.savefig('dobstrue.png')

    # Normalize true data
    dobstrue_ = np.clip(dobstrue, cond_min, cond_max)
    dobstrue_ = (dobstrue_ - cond_min) / (cond_max - cond_min)
    dobstrue_ = np.nan_to_num(dobstrue_, nan=0.0)
    dobstrue_ = np.clip(dobstrue_, 0, 1)
    dobstrue_norm = dobstrue_ * 2 - 1

    # Plot comparison figures
    fig, ax = plt.subplots()
    ax.plot(dobstrue_norm[0], 'r', zorder=2)
    for i in range(900):
        ax.plot(cond_norm[i], 'b', alpha=0.1, zorder=1)
    fig.savefig('dobstrue_comp1.png')

    fig, ax = plt.subplots()
    ax.plot(dobstrue_norm[0], 'r', zorder=2)
    for i in range(900, 1000):
        ax.plot(cond_norm[i], 'b', alpha=0.1, zorder=1)
    fig.savefig('dobstrue_comp2.png')

    # Create dataloaders
    dataloader = create_dataloader(data_norm, cond_norm)
    testloader = create_dataloader2(data_norm, cond_norm)

    # Normalize true parameters
    true_para_norm = (true_para - data_min) / (data_max - data_min + Config.eps) * 2 - 1
    true_para_norm[true_para_norm > 1] = 1
    true_para_norm[true_para_norm < -1] = -1

    # Pack initial results
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
    data_path = os.path.join(rootpath, 'data')  # Fixed path join

    mode = 1  # 0 for training, 1 for testing
    if mode == 0:
        # Training mode
        os.chdir(rootpath)
        args = get_params()
        t1 = time.time()
        train_loss = train(args, dataloader, testloader,
                           dobstrue_norm, true_para_norm, data_max, data_min, config)
        t2 = time.time()
        print(f'Training time: {t2 - t1:.2f} seconds')

        # Save loss
        os.makedirs(data_path, exist_ok=True)
        with open(os.path.join(data_path, 'train_loss.pkl'), 'wb') as f:
            pickle.dump(train_loss, f)

        print(f'Min loss: {min(train_loss)}')

        # Plot loss curve
        plt.subplots(figsize=(5, 3))
        plt.plot(train_loss, label='Train Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss: {min(train_loss):.4f}')
        plt.savefig(os.path.join(data_path, 'train_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f'#### Training time: {t2 - t1:.2f} seconds')

    elif mode == 1:
        # Testing mode
        os.chdir(rootpath)
        args = get_params()
        print(f'Current working directory: {os.getcwd()}')
        generated_data = test(dataloader, dobstrue_norm, true_para_norm, data_max, data_min,
                              args, config)
        print(f'Generated data shape: {generated_data.shape}')

        # Save generated data
        final_inv_para = generated_data.cpu().numpy()
        final_inv_para = final_inv_para.reshape(final_inv_para.shape[0], -1)
        with h5py.File('post_data.h5', 'w') as f:
            f.create_dataset('data', data=final_inv_para)


if __name__ == '__main__':
    main()