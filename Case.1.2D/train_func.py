import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet
import torch.optim as optim
from FMIM import FMIM  # Renamed from RectifiedFlow
from utils import get_named_beta_schedule
from Scheduler import GradualWarmupScheduler
import pickle
import copy

path = os.path.dirname(os.path.abspath(__file__))


class EMA():
    """Exponential Moving Average class for maintaining sliding average of model parameters"""

    def __init__(self, beta):
        super().__init__()
        self.beta = beta  # Smoothing coefficient

    def update_model_average(self, ma_model, current_model):
        """Update moving average model parameters"""
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """Calculate moving average for single parameter"""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def generate(fmim_model, lab_test, params, Config):
    """Generate samples using FMIM model with classifier-free guidance"""
    fmim_model.model.eval()
    with torch.no_grad():
        genshape = (lab_test.shape[0], Config.channels, Config.image_size[1], Config.image_size[0])
        # Pass guidance strength params.w (from command line args)
        generated = fmim_model.sample(
            genshape,
            num_steps=params.num_steps,
            cemb=lab_test,
            w=params.w  # Enable guidance
        )
    return generated


def train(params, dataloader, testloader,
          dobstrue_norm, true_para_norm, data_max, data_min,
          Config
          ):
    """Main training function for FMIM model (removed all plotting logic)"""
    path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(path, 'data')
    os.makedirs(data_path, exist_ok=True)  # Ensure data directory exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize U-Net model
    net = Unet(
        in_ch=params.inch,
        mod_ch=params.modch,
        out_ch=params.outch,
        ch_mul=params.chmul,
        num_res_blocks=params.numres,
        cdim=params.cdim,
        use_conv=params.useconv,
        droprate=params.droprate,
        dtype=params.dtype,
        image_size=Config.image_size,
        tdim=params.tdim,
    )

    # Create FMIM model instance
    fmim_model = FMIM(
        dtype=params.dtype,
        model=net,
        sigma=params.sigma,  # Noise parameter
        device=device
    )

    # Move model to device
    fmim_model.model = fmim_model.model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        fmim_model.model.parameters(),
        lr=params.lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler: warmup first, then cosine annealing
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=params.epoch,
        eta_min=params.lr * 0.01,
        last_epoch=-1
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=params.multiplier,
        warm_epoch=params.epoch // 20,
        after_scheduler=cosineScheduler,
    )

    # Initialize EMA
    ema = EMA(0.995)
    ema_model = copy.deepcopy(fmim_model.model)
    update_ema_every = 10  # Update EMA every 10 steps
    step_start_ema = int(params.epoch * 0.7)  # Epoch to start EMA update

    # Training process
    train_loss = []
    for epc in range(params.epoch):
        fmim_model.model.train()  # Training mode
        loss_ep = 0  # Record total loss of current epoch

        for batch_idx, (img, lab) in enumerate(dataloader):
            b = img.shape[0]  # Batch size
            optimizer.zero_grad()  # Zero gradients

            x_1 = img.to(device)  # Real samples
            lab = lab.to(device)  # Conditional labels

            # Classifier-free guidance: randomly drop part of conditions (set to 0)
            mask = torch.rand(b, device=device) < params.threshold
            lab[mask] = 0

            # Calculate FMIM loss
            loss = fmim_model.train_loss(x_1, cemb=lab)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            loss_ep += loss.item()

            # Update EMA model
            if epc < step_start_ema:
                # Before EMA starts, directly copy parameters
                ema_model.load_state_dict(fmim_model.model.state_dict())
            else:
                # After EMA starts, perform exponential moving average update
                if batch_idx % update_ema_every == 0:
                    ema.update_model_average(ema_model, fmim_model.model)

        # Record and print average loss of current epoch
        avg_loss = loss_ep / len(dataloader)
        print(f"Epoch [{epc + 1}/{params.epoch}] Average Loss: {avg_loss:.4f}")
        train_loss.append(avg_loss)
        warmUpScheduler.step()  # Update learning rate

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Save final model
    torch.save(fmim_model.model.state_dict(), "fmim_model.pth")
    torch.save(ema_model.state_dict(), "ema_model.pth")

    # Save training loss
    with open(os.path.join(data_path, 'train_loss.pkl'), 'wb') as f:
        pickle.dump(train_loss, f)

    return train_loss