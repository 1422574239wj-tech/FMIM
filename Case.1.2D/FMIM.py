import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy


class FMIM(nn.Module):  # Renamed from RectifiedFlow
    def __init__(self,
                 dtype: torch.dtype,
                 model: nn.Module,
                 sigma: float = 1.0,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device)
        self.model.dtype = self.dtype
        self.sigma = sigma
        self.device = device

        self.ema = False
        self.ema_model = copy.deepcopy(self.model)

    def get_noise(self, shape: tuple) -> torch.Tensor:
        return torch.randn(shape, dtype=self.dtype, device=self.device) * self.sigma

    def sample_pt(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size, dtype=self.dtype, device=self.device)

    def rectified_trajectory(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        return (1 - t) * x0 + t * x1

    def compute_rectified_flow(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - x0

    def train_loss(self, x1: torch.Tensor, **model_kwargs) -> torch.Tensor:
        batch_size = x1.shape[0]

        x0 = self.get_noise(x1.shape)
        t = self.sample_pt(batch_size)

        x_t = self.rectified_trajectory(x0, x1, t)

        u_true = self.compute_rectified_flow(x0, x1)

        if self.ema:
            u_pred = self.ema_model(x_t, t, **model_kwargs)
        else:
            u_pred = self.model(x_t, t, **model_kwargs)

        # L2 loss
        return F.mse_loss(u_pred, u_true, reduction='mean')

    def ode_solver(self,
                   x0: torch.Tensor,
                   t0: float = 0.0,
                   t1: float = 1.0,
                   num_steps: int = 300,
                   method: str = 'rk4',
                   w: float = 0.0,
                   **model_kwargs) -> torch.Tensor:

        dt = (t1 - t0) / num_steps
        x = x0

        model_func = self.ema_model if self.ema else self.model

        cemb = model_kwargs.get('cemb', None)
        uncond_cemb = torch.zeros_like(cemb) if cemb is not None else None

        def get_model_kwargs(cond_emb):
            kwargs = model_kwargs.copy()
            if cond_emb is not None:
                kwargs['cemb'] = cond_emb
            return kwargs

        if method == 'euler':
            for i in range(num_steps):
                t = torch.tensor(t0 + i * dt, device=self.device, dtype=self.dtype).repeat(x.shape[0])

                if w > 0 and cemb is not None:
                    v_cond = model_func(x, t, **get_model_kwargs(cemb))
                    v_uncond = model_func(x, t, **get_model_kwargs(uncond_cemb))
                    v = v_uncond + w * (v_cond - v_uncond)
                else:
                    v = model_func(x, t, **model_kwargs)

                x = x + dt * v

        elif method == 'heun':
            for i in range(num_steps):
                t_current = t0 + i * dt
                t = torch.tensor(t_current, device=self.device, dtype=self.dtype).repeat(x.shape[0])

                if w > 0 and cemb is not None:
                    k1_cond = model_func(x, t, **get_model_kwargs(cemb))
                    k1_uncond = model_func(x, t, **get_model_kwargs(uncond_cemb))
                    k1 = k1_uncond + w * (k1_cond - k1_uncond)

                    t_next = torch.tensor(t_current + dt, device=self.device, dtype=self.dtype).repeat(x.shape[0])
                    k2_cond = model_func(x + dt * k1, t_next, **get_model_kwargs(cemb))
                    k2_uncond = model_func(x + dt * k1, t_next, **get_model_kwargs(uncond_cemb))
                    k2 = k2_uncond + w * (k2_cond - k2_uncond)
                else:
                    k1 = model_func(x, t, **model_kwargs)
                    t_next = torch.tensor(t_current + dt, device=self.device, dtype=self.dtype).repeat(x.shape[0])
                    k2 = model_func(x + dt * k1, t_next, **model_kwargs)

                x = x + dt * 0.5 * (k1 + k2)

        elif method == 'rk4':
            for i in range(num_steps):
                t_current = t0 + i * dt
                t = torch.tensor(t_current, device=self.device, dtype=self.dtype).repeat(x.shape[0])

                if w > 0 and cemb is not None:
                    k1_cond = model_func(x, t, **get_model_kwargs(cemb))
                    k1_uncond = model_func(x, t, **get_model_kwargs(uncond_cemb))
                    k1 = k1_uncond + w * (k1_cond - k1_uncond)

                    t_half = torch.tensor(t_current + dt / 2, device=self.device, dtype=self.dtype).repeat(x.shape[0])
                    k2_cond = model_func(x + dt * k1 / 2, t_half, **get_model_kwargs(cemb))
                    k2_uncond = model_func(x + dt * k1 / 2, t_half, **get_model_kwargs(uncond_cemb))
                    k2 = k2_uncond + w * (k2_cond - k2_uncond)

                    k3_cond = model_func(x + dt * k2 / 2, t_half, **get_model_kwargs(cemb))
                    k3_uncond = model_func(x + dt * k2 / 2, t_half, **get_model_kwargs(uncond_cemb))
                    k3 = k3_uncond + w * (k3_cond - k3_uncond)

                    t_next = torch.tensor(t_current + dt, device=self.device, dtype=self.dtype).repeat(x.shape[0])
                    k4_cond = model_func(x + dt * k3, t_next, **get_model_kwargs(cemb))
                    k4_uncond = model_func(x + dt * k3, t_next, **get_model_kwargs(uncond_cemb))
                    k4 = k4_uncond + w * (k4_cond - k4_uncond)
                else:
                    k1 = model_func(x, t, **model_kwargs)
                    t_half = torch.tensor(t_current + dt / 2, device=self.device, dtype=self.dtype).repeat(x.shape[0])
                    k2 = model_func(x + dt * k1 / 2, t_half, **model_kwargs)
                    k3 = model_func(x + dt * k2 / 2, t_half, **model_kwargs)
                    t_next = torch.tensor(t_current + dt, device=self.device, dtype=self.dtype).repeat(x.shape[0])
                    k4 = model_func(x + dt * k3, t_next, **model_kwargs)

                x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return x

    def sample(self, shape: tuple, num_steps: int = 30, method: str = 'euler', w: float = 0.0,
               **model_kwargs) -> torch.Tensor:
        x0 = self.get_noise(shape)
        return self.ode_solver(
            x0,
            t0=0.0,
            t1=1.0,
            num_steps=num_steps,
            method=method,
            w=w,
            **model_kwargs
        )
