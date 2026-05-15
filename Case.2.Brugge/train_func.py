import os
import torch
import torch.optim as optim
from unet import Unet
from FMIM import FMIM
from Scheduler import GradualWarmupScheduler
import pickle
import copy
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))


class EMA:
    """Exponential Moving Average"""
    def __init__(self, beta: float):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data = self.update_average(ma_params.data, current_params.data)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1.0 - self.beta) * new


@torch.no_grad()
def evaluate_loss(fmim, dataloader, device, use_ema=False, val_repeats=2):
    """
    Evaluate loss.
    Note: The loss of FMIM samples x0 and t randomly each time, so the validation loss has noise.
    Using val_repeats>1 to calculate the mean will be more stable.
    """
    prev_ema_flag = fmim.ema
    fmim.ema = use_ema

    fmim.model.eval()
    fmim.ema_model.eval()

    total_loss = 0.0
    total_batches = 0

    for _ in range(val_repeats):
        for img, lab in dataloader:
            x1 = img.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            # Do not use classifier-free dropout during validation, directly evaluate performance under full conditions
            loss = fmim.train_loss(x1, cemb=lab)

            total_loss += loss.item()
            total_batches += 1

    fmim.ema = prev_ema_flag
    fmim.model.train()

    return total_loss / max(total_batches, 1)


def train(params, dataloader, valloader,
          dobstrue_norm, true_para_norm, data_max, data_min,
          Config):
    """
    Train FMIM and save the optimal model based on the validation set.
    Default saved files:
        - fmim.pth / ema_model.pth: Optimal on validation set
        - last_fmim.pth / last_ema_model.pth: Last epoch
    """
    path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(path, 'data')
    os.makedirs(data_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== Adjustable training strategy parameters (default values are used even if not in parser)=====
    ema_decay = getattr(params, "ema_decay", 0.995)
    update_ema_every = getattr(params, "update_ema_every", 1)
    ema_start_epoch = getattr(params, "ema_start_epoch", max(5, params.epoch // 10))
    grad_clip = getattr(params, "grad_clip", 1.0)
    val_interval = getattr(params, "interval", 1)
    val_repeats = getattr(params, "val_repeats", 2)
    patience = getattr(params, "patience", 50)  # <=0 means disable early stopping

    # ===== Initialize model =====
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
        ts_feature=Config.ts_feature
    )

    fmim = FMIM(
        dtype=params.dtype,
        model=net,
        sigma=params.sigma,
        device=device
    )
    fmim.model = fmim.model.to(device)

    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(
        fmim.model.parameters(),
        lr=params.lr,
        weight_decay=1e-4
    )

    # ===== Learning rate scheduler =====
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=params.epoch,
        eta_min=params.lr * 0.01,
        last_epoch=-1
    )
    warmup_scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=params.multiplier,
        warm_epoch=max(1, params.epoch // 20),
        after_scheduler=cosine_scheduler,
    )

    # ===== EMA =====
    ema = EMA(ema_decay)
    ema_model = copy.deepcopy(fmim.model).to(device)
    fmim.ema_model = ema_model
    fmim.ema = False

    # ===== Training history =====
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ema_loss": [],
        "lr": []
    }

    best_val = float("inf")
    best_epoch = -1
    no_improve_count = 0
    global_step = 0

    print("Start training...")
    print(f"ema_start_epoch = {ema_start_epoch}")
    print(f"val_interval = {val_interval}")
    print(f"val_repeats = {val_repeats}")

    for epc in range(params.epoch):
        fmim.model.train()
        loss_ep = 0.0

        for batch_idx, (img, lab) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            x1 = img.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True).clone()

            # Condition dropout for classifier-free guidance
            if params.threshold > 0:
                mask = torch.rand(x1.shape[0], device=device) < params.threshold
                lab[mask] = 0.0

            loss = fmim.train_loss(x1, cemb=lab)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(fmim.model.parameters(), grad_clip)

            optimizer.step()
            loss_ep += loss.item()
            global_step += 1

            # EMA update: Start as early as possible, do not delay to the last 70%
            if epc >= ema_start_epoch and global_step % update_ema_every == 0:
                ema.update_model_average(ema_model, fmim.model)

        # Before EMA starts, keep ema_model consistent with the current model
        if epc < ema_start_epoch:
            ema_model.load_state_dict(fmim.model.state_dict())

        avg_train_loss = loss_ep / max(len(dataloader), 1)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["lr"].append(current_lr)

        # Learning rate scheduling
        warmup_scheduler.step()

        # ===== Validation =====
        do_val = (valloader is not None) and (((epc + 1) % val_interval == 0) or (epc == 0))

        if do_val:
            val_loss = evaluate_loss(
                fmim=fmim,
                dataloader=valloader,
                device=device,
                use_ema=False,
                val_repeats=val_repeats
            )

            val_ema_loss = evaluate_loss(
                fmim=fmim,
                dataloader=valloader,
                device=device,
                use_ema=True,
                val_repeats=val_repeats
            )

            history["val_loss"].append(val_loss)
            history["val_ema_loss"].append(val_ema_loss)

            # Use EMA validation loss as the criterion for the best model
            metric = val_ema_loss

            improved = metric < best_val
            if improved:
                best_val = metric
                best_epoch = epc + 1
                no_improve_count = 0

                # Save "best model" to default filename, test_func.py can read it directly
                torch.save(fmim.model.state_dict(), "fmim.pth")
                torch.save(ema_model.state_dict(), "ema_model.pth")

                # Save an additional checkpoint with information
                torch.save({
                    "epoch": epc + 1,
                    "model_state_dict": fmim.model.state_dict(),
                    "ema_model_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_ema_loss": best_val,
                    "params": vars(params) if hasattr(params, "__dict__") else {}
                }, os.path.join(data_path, "best_checkpoint.pth"))

            else:
                no_improve_count += 1

            print(
                f"Epoch [{epc+1:03d}/{params.epoch}] | "
                f"train={avg_train_loss:.6f} | "
                f"val={val_loss:.6f} | "
                f"val_ema={val_ema_loss:.6f} | "
                f"lr={current_lr:.6e} | "
                f"best_val_ema={best_val:.6f} (epoch {best_epoch})"
            )

        else:
            history["val_loss"].append(np.nan)
            history["val_ema_loss"].append(np.nan)

            print(
                f"Epoch [{epc+1:03d}/{params.epoch}] | "
                f"train={avg_train_loss:.6f} | "
                f"lr={current_lr:.6e}"
            )

        # Save last epoch
        torch.save(fmim.model.state_dict(), "last_fmim.pth")
        torch.save(ema_model.state_dict(), "last_ema_model.pth")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Early stopping (optional)
        if patience is not None and patience > 0 and no_improve_count >= patience:
            print(f"Early stopping at epoch {epc+1}, best epoch = {best_epoch}, best val_ema = {best_val:.6f}")
            break

    # If no validation was performed during training, save the last model as the default model
    if best_epoch == -1:
        torch.save(fmim.model.state_dict(), "fmim.pth")
        torch.save(ema_model.state_dict(), "ema_model.pth")
        best_epoch = len(history["train_loss"])
        print("No validation performed, the last epoch model has been saved as the default model.")
    else:
        print(f"Training completed, the best model is from epoch {best_epoch}, best val_ema_loss = {best_val:.6f}")

    # Save loss history
    with open(os.path.join(data_path, "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    with open(os.path.join(data_path, "train_loss.pkl"), "wb") as f:
        pickle.dump(history["train_loss"], f)

    with open(os.path.join(data_path, "val_loss.pkl"), "wb") as f:
        pickle.dump(history["val_loss"], f)

    with open(os.path.join(data_path, "val_ema_loss.pkl"), "wb") as f:
        pickle.dump(history["val_ema_loss"], f)

    return history