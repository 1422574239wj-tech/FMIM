import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy


class RectifiedFlow(nn.Module):
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


        return F.mse_loss(u_pred, u_true, reduction='mean')

    def ode_solver(self,
                   x0: torch.Tensor,
                   t0: float = 0.0,
                   t1: float = 1.0,
                   num_steps: int = 300,
                   method: str = 'rk4',
                   **model_kwargs) -> torch.Tensor:

        dt = (t1 - t0) / num_steps
        x = x0

        model_func = self.ema_model if self.ema else self.model

        if method == 'euler':
            for i in range(num_steps):
                t = torch.tensor(t0 + i * dt, device=self.device, dtype=self.dtype).repeat(x.shape[0])
                v = model_func(x, t, **model_kwargs)
                x = x + dt * v
        elif method == 'heun':
            for i in range(num_steps):
                t_current = t0 + i * dt
                t = torch.tensor(t_current, device=self.device, dtype=self.dtype).repeat(x.shape[0])

                k1 = model_func(x, t, **model_kwargs)
                k2 = model_func(x + dt * k1,
                                torch.tensor(t_current + dt, device=self.device, dtype=self.dtype).repeat(x.shape[0]),
                                **model_kwargs)
                x = x + dt * 0.5 * (k1 + k2)
        elif method == 'rk4':
            for i in range(num_steps):
                t_current = t0 + i * dt
                t = torch.tensor(t_current, device=self.device, dtype=self.dtype).repeat(x.shape[0])

                k1 = model_func(x, t, **model_kwargs)
                k2 = model_func(x + dt * k1 / 2,
                                torch.tensor(t_current + dt / 2, device=self.device, dtype=self.dtype).repeat(
                                    x.shape[0]), **model_kwargs)
                k3 = model_func(x + dt * k2 / 2,
                                torch.tensor(t_current + dt / 2, device=self.device, dtype=self.dtype).repeat(
                                    x.shape[0]), **model_kwargs)
                k4 = model_func(x + dt * k3,
                                torch.tensor(t_current + dt, device=self.device, dtype=self.dtype).repeat(x.shape[0]),
                                **model_kwargs)
                x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    def sample(self, shape: tuple, num_steps: int = 300, method: str = 'euler', **model_kwargs) -> torch.Tensor:

        x0 = self.get_noise(shape)
        return self.ode_solver(x0, t0=0.0, t1=1.0, num_steps=num_steps, method=method, **model_kwargs)
