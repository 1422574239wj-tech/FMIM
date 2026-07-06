import os
import time
import torch
import numpy as np
from unet import Unet
from FMIM import FMIM
import matplotlib.pyplot as plt

# Configure paths
path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')
os.makedirs(data_path, exist_ok=True)


def generate(fmim_model, lab_test, params, Config):
    fmim_model.model.eval()
    with torch.no_grad():
        genshape = (lab_test.shape[0], Config.channels, Config.image_size[1], Config.image_size[0])

        generated = fmim_model.sample(
            genshape,
            num_steps=params.num_steps,
            cemb=lab_test,
            w=params.w
        )
    return generated


def plot_images(images, true_ima, titles=None, name=None):

    plt.figure(figsize=(15, 5))

    for i in range(3):
        ax = plt.subplot(1, 5, i + 1)
        img = images[i].permute(1, 2, 0)
        im = plt.imshow(img[:, :, 0], cmap='jet', vmin=-1, vmax=1)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.04)

    ax = plt.subplot(1, 5, 4)
    img = torch.mean(images[:3], axis=0).permute(1, 2, 0)
    im = plt.imshow(img[:, :, 0], cmap='jet', vmin=-1, vmax=1)
    plt.title('Mean Image')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = plt.subplot(1, 5, 5)
    img = true_ima.permute(1, 2, 0)
    im = plt.imshow(img[:, :, 0], cmap='jet', vmin=-1, vmax=1)
    plt.title('True Image')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f'generated_images_{name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_images1(images, true_ima, titles=None, name=None):

    plt.figure(figsize=(15, 5))

    for i in range(3):
        ax = plt.subplot(1, 5, i + 1)
        img = images[i].permute(1, 2, 0)
        im = plt.imshow(img[:, :, 0], cmap='jet')
        if titles:
            plt.title(titles[i])
        plt.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.04)

    ax = plt.subplot(1, 5, 4)
    img = torch.mean(images[:3], axis=0).permute(1, 2, 0)
    im = plt.imshow(img[:, :, 0], cmap='jet')
    plt.title('Mean Image')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = plt.subplot(1, 5, 5)
    img = true_ima.permute(1, 2, 0)
    im = plt.imshow(img[:, :, 0], cmap='jet')
    plt.title('True Image')
    plt.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f'generated_images_{name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def test(dataloader, dobstrue_norm, true_para_norm, data_max, data_min, params, Config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    print(f"Random seed: {seed}")

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

    fmim_model = FMIM(
        dtype=params.dtype,
        model=net,
        sigma=params.sigma,
        device=device
    )

    # Load main model weights
    try:
        fmim_model.model.load_state_dict(
            torch.load("fmim.pth", map_location=device)
        )
        fmim_model.model.to(device)
        print("Successfully loaded main model weights and moved to device")
    except Exception as e:
        print(f"Failed to load main model: {e}")
        return None

    ema_model = Unet(
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
    ).to(device)

    try:
        ema_model.load_state_dict(
            torch.load("ema_model.pth", map_location=device)
        )
        fmim_model.ema_model = ema_model
        fmim_model.ema = True
        fmim_model.ema_model.eval()
        print("Successfully loaded EMA model weights")
    except Exception as e:
        print(f"Failed to load EMA model: {e}")
        print("Using main model for testing")
        fmim_model.ema = False

    # Test training set samples
    print("\n=== Testing Training Set Samples ===")
    for batch_idx, (x, lab) in enumerate(dataloader):
        x_test = x[:5].to(device)
        lab_test = lab[:5].to(device)
        print(f'x_test shape: {x_test.shape}, sample values: {x_test[0, 0, :2, :2].cpu().numpy()}')
        print(f'lab_test shape: {lab_test.shape}, sample values: {lab_test[0, :5].cpu().numpy()}')
        break

    print(f'lab_test first 50 values: {lab_test[0, :50].cpu().numpy()}')

    # Generate samples for training set
    generated_data = []
    for i in range(5):
        print(f'Generating {i}-th training set sample...')
        generated_ = generate(fmim_model, lab_test, params, Config)
        print(f"{i}-th training set sample shape: {generated_.shape}")
        generated_data.append(generated_)

    # Process generated results
    generated_data = torch.stack(generated_data, dim=0)
    generated_data = generated_data.squeeze(0)
    generated_data = generated_data.detach().cpu()
    print(f'Processed training set generated shape: {generated_data.shape}')
    print(f'Training set test sample shape: {x_test.shape}')

    for i in range(5):
        plot_images(generated_data[:, i], x_test[i].cpu(), name=f'train_{i}')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lab_test[0, :].cpu().numpy())
    ax.set_title('Training Label')
    fig.savefig(os.path.join(data_path, 'train_lab.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Test real samples
    print("\n=== Testing Real Samples ===")
    dobstrue_norm = torch.tensor(dobstrue_norm.reshape(1, -1), dtype=torch.float32)
    lab_test = dobstrue_norm.to(device).float()
    print(f'Real sample label shape: {lab_test.shape}, first 50 values: {lab_test[0, :50].cpu().numpy()}')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lab_test[0, :].cpu().numpy())
    ax.set_title('Test Label')
    fig.savefig(os.path.join(data_path, 'test_lab.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Generate 50 real samples with timing
    generated_data = []
    start_time = time.time()
    print(f"Start generating 50 real samples, time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    for i in range(50):
        if i % 10 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = elapsed / (i + 1) * (50 - i - 1) if i > 0 else 0
            print(f'Progress: {i}/50, Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s')

        generated_ = generate(fmim_model, lab_test, params, Config)
        generated_data.append(generated_)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / 50
    print(f"\n50 real samples generated successfully!")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total time: {total_time:.2f}s, Average per sample: {avg_time:.2f}s")

    # Process generation results
    generated_data = torch.stack(generated_data, dim=0)
    generated_data = generated_data.squeeze(1)
    generated_data = generated_data.detach().cpu()
    print(f'Real sample generation result shape: {generated_data.shape}')
    print(f'True parameter shape: {true_para_norm.shape}')

    plot_images(generated_data, true_para_norm[0].cpu(), name='test')

    gen_data_ = generated_data / 2 + 0.5
    gen_data = gen_data_ * (data_max - data_min) + data_min

    x_test_ = true_para_norm.cpu() / 2 + 0.5
    x_test_1 = x_test_ * (data_max - data_min) + data_min

    plot_images1(gen_data, x_test_1[0].cpu(), name='test_ori')

    return gen_data
