import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy


class RectifiedFlow(nn.Module):
    def __init__(self,
                 dtype: torch.dtype,
                 model: nn.Module,
                 sigma: float = 1.0,  # 标准Rectified Flow使用标准高斯噪声
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device)
        self.model.dtype = self.dtype
        self.sigma = sigma
        self.device = device

        # EMA模型（保持原有逻辑）
        self.ema = False
        self.ema_model = copy.deepcopy(self.model)

    def get_noise(self, shape: tuple) -> torch.Tensor:
        """生成初始噪声（标准高斯分布）"""
        return torch.randn(shape, dtype=self.dtype, device=self.device) * self.sigma

    def sample_pt(self, batch_size: int) -> torch.Tensor:
        """采样时间点t ∈ [0,1]"""
        return torch.rand(batch_size, dtype=self.dtype, device=self.device)

    def rectified_trajectory(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Rectified Flow的标准线性轨迹
        公式：x_t = (1-t)*x0 + t*x1
        """
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        return (1 - t) * x0 + t * x1

    def compute_rectified_flow(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        计算整流后的真实速度场
        对于线性轨迹，速度场是常数：u_t = x1 - x0
        """
        return x1 - x0

    def train_loss(self, x1: torch.Tensor, **model_kwargs) -> torch.Tensor:
        """训练损失：拟合整流后的速度场"""
        batch_size = x1.shape[0]

        # 采样初始噪声x0和时间t
        x0 = self.get_noise(x1.shape)
        t = self.sample_pt(batch_size)

        # 生成整流轨迹上的点x_t
        x_t = self.rectified_trajectory(x0, x1, t)

        # 计算整流后的真实速度场
        u_true = self.compute_rectified_flow(x0, x1)

        # 模型预测速度场
        if self.ema:
            u_pred = self.ema_model(x_t, t, **model_kwargs)
        else:
            u_pred = self.model(x_t, t, **model_kwargs)

        # L2损失
        return F.mse_loss(u_pred, u_true, reduction='mean')

    def ode_solver(self,
                   x0: torch.Tensor,
                   t0: float = 0.0,
                   t1: float = 1.0,
                   num_steps: int = 300,
                   method: str = 'rk4',
                   **model_kwargs) -> torch.Tensor:
        """ODE求解器，支持多种方法"""
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
        """采样：从初始噪声通过整流流生成数据"""
        x0 = self.get_noise(shape)
        return self.ode_solver(x0, t0=0.0, t1=1.0, num_steps=num_steps, method=method, **model_kwargs)