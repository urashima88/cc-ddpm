import torch

def make_beta_schedule(schedule_type, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    match schedule_type:
        case "linear":
            betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float32)
        case "cosine":
            timesteps = (torch.arange(n_timestep + 1, dtype=torch.float32) / n_timestep + cosine_s)
            alphas = timesteps / (1 + cosine_s) * torch.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = torch.clip(betas, min=0, max=0.999)
        case _:
            raise ValueError(f"schedule type '{schedule_type}' is unknown")

    return betas

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params")
    return total_params