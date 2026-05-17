from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=10000) -> torch.Tensor:
    """Enhanced time embedding for better FMIM continuous time characteristics"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([
        torch.cos(args), torch.sin(args),
        torch.cos(2*args), torch.sin(2*args)
    ], dim=-1)
    if embedding.shape[-1] > dim:
        embedding = embedding[:, :dim]
    elif embedding.shape[-1] < dim:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :dim - embedding.shape[-1]])], dim=-1)
    return embedding


class Upsample(nn.Module):
    """Upsample layer remains unchanged"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f'Channel mismatch: {x.shape[1]} vs {self.in_ch}'
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.layer(x)


class Downsample(nn.Module):
    """Downsample layer remains unchanged"""
    def __init__(self, in_ch: int, out_ch: int, use_conv: bool):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if use_conv:
            self.layer = nn.Sequential(
                nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, out_ch),
                nn.SiLU()
            )
        else:
            self.layer = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f'Channel mismatch: {x.shape[1]} vs {self.in_ch}'
        return self.layer(x)


class EmbedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, temb, cemb):
        pass


class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x: torch.Tensor, temb: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb)
            else:
                x = layer(x)
        return x


class ResBlock(EmbedBlock):
    """Enhanced residual block to improve velocity field prediction accuracy"""
    def __init__(self, in_ch: int, out_ch: int, tdim: int, cdim: int, droprate: float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.cdim = cdim
        self.droprate = droprate

        self.block_1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )

        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch)
        )
        self.cemb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cdim, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch)
        )

        self.block_2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(p=self.droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
        latent += self.temb_proj(temb)[:, :, None, None]
        latent += self.cemb_proj(cemb)[:, :, None, None]
        latent = self.block_2(latent)
        latent += self.residual(x)
        return latent


def grid_reshape(x, ld=5):
    if x < np.power(2, ld):
        x1 = np.power(2, ld)
    else:
        x1 = np.round(x / np.power(2, ld)) * np.power(2, ld)
    return int(x1)


class Unet(nn.Module):
    """U-Net model adapted for FMIM"""
    def __init__(self, in_ch=1, mod_ch=64, out_ch=1, ch_mul=[1, 2, 4], num_res_blocks=2, cdim=16850,
                 use_conv=True, droprate=0, dtype=torch.float32, image_size=(32, 32), tdim=256,
                 ts_feature=None):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        self.cdim = cdim
        self.use_conv = use_conv
        self.droprate = droprate
        self.dtype = dtype
        self.ts_feature = ts_feature

        self.temb_layer = nn.Sequential(
            nn.Linear(1, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim)
        )

        self.cemb_layer = nn.Sequential(
            nn.Linear(self.cdim, int(4 * tdim)),
            nn.SiLU(),
            nn.Linear(int(4 * tdim), 512),
            nn.SiLU(),
            nn.Linear(512, tdim)
        )

        self.downblocks = nn.ModuleList([
            EmbedSequential(nn.Conv2d(in_ch, self.mod_ch, 3, padding=1))
        ])
        now_ch = self.ch_mul[0] * self.mod_ch
        chs = [now_ch]
        for i, mul in enumerate(self.ch_mul):
            nxt_ch = mul * self.mod_ch
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(now_ch, nxt_ch, tdim, tdim, self.droprate),
                ]
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mul) - 1:
                self.downblocks.append(EmbedSequential(Downsample(now_ch, now_ch, self.use_conv)))
                chs.append(now_ch)

        self.middleblocks = EmbedSequential(
            ResBlock(now_ch, now_ch, tdim, tdim, self.droprate),
            ResBlock(now_ch, now_ch, tdim, tdim, self.droprate),
        )

        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul))[::-1]:
            nxt_ch = mul * self.mod_ch
            for j in range(num_res_blocks + 1):
                layers = [
                    ResBlock(now_ch + chs.pop(), nxt_ch, tdim, tdim, self.droprate),
                ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, self.out_ch, 3, stride=1, padding=1),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=1, padding=0)
        )
        self.x1 = grid_reshape(image_size[0])
        self.y1 = grid_reshape(image_size[1])
        self.image_size = image_size

    def forward(self, x: torch.Tensor, t: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        x = F.interpolate(x, size=(self.y1, self.x1), mode='bilinear', align_corners=False)

        temb = self.temb_layer(t)

        cemb = self.cemb_layer(cemb)

        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            h = block(h, temb, cemb)
            hs.append(h)
        h = self.middleblocks(h, temb, cemb)
        for block in self.upblocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, temb, cemb)
        h = h.type(self.dtype)
        out = self.out(h)
        out = F.interpolate(out, size=(self.image_size[1], self.image_size[0]), mode='bilinear', align_corners=False)
        return out
