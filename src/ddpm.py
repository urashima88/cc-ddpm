import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_beta_schedule, count_params

class DDPM(nn.Module):
    def __init__(self,
                 eps_model,
                 timesteps=1000,
                 given_beta=None,
                 beta_schedule_type="linear",
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                ):
        super().__init__()
        self.device = device
        self.eps_model = eps_model.to(device)
        self.n_timesteps = timesteps
        self.register_schedule(given_beta,
                               beta_schedule_type,
                               timesteps,
                               linear_start,
                               linear_end, cosine_s)

    def register_schedule(self, given_beta, beta_schedule_type, timesteps, linear_start, linear_end, cosine_s):
        beta = None
        if given_beta:
            beta = given_beta
        else:
            beta = make_beta_schedule(beta_schedule_type, timesteps, linear_start, linear_end, cosine_s)

        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0) # alpha-bar for t steps
        timesteps, = beta.shape

        assert alpha_bar.shape[0] == timesteps

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        self.register_buffer("beta", beta.to(self.device))
        self.register_buffer("alpha", alpha.to(self.device))
        self.register_buffer("alpha_bar", alpha_bar.to(self.device))
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar.to(self.device))
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar.to(self.device))

        count_params(self.eps_model, verbose=True)

    def forward(self, x, y):
        batch_size = x.shape[0]
        t = torch.randint(low=0, high=self.n_timesteps, size=(batch_size,), device=self.device)
        noise = torch.randn_like(x, device=self.device)
        xt = self.sqrt_alpha_bar[t].view((batch_size, 1, 1, 1)) * x + \
        self.sqrt_one_minus_alpha_bar[t].view((batch_size, 1, 1, 1)) * noise

        y = y.long()
        pred_noise = self.eps_model(xt, t, y)

        return F.mse_loss(pred_noise, noise)

    def sample(self, n_samples, shape, y=None):
        self.eval()

        with torch.no_grad():
            xt = torch.randn(n_samples, *shape, device=self.device)
            for t in reversed(range(self.n_timesteps)):
                tb = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                pred_noise = self.eps_model(xt, tb, y)
                z = torch.randn_like(xt, device=self.device) if t > 0 else 0
                xt = 1 / torch.sqrt(self.alpha[t]) *  \
                (xt - pred_noise * (1 - self.alpha[t]) / self.sqrt_one_minus_alpha_bar[t]) \
                 + torch.sqrt(self.beta[t]) * z

            return xt