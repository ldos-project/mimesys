import math

import torch


def cosine_beta_schedule(n_timesteps, s=0.008, dtype=torch.float32):
    x = (
        (torch.arange(0, n_timesteps + 1, dtype=dtype) / n_timesteps + s)
        / (1 + s)
        * math.pi
        * 0.5
    )
    fts = torch.pow(x.cos(), 2)
    return torch.clip(1 - fts[1:] / fts[:-1], 0, 0.999)


def diffusion_sample_fn(model, x, t, context_cond=None):
    """
    need to not use noise when t = 0
    """
    noise = torch.randn_like(x)
    noise[t == 0] = 0
    mean, _, log_var = model.p_mean_variance(x, t, context_cond)
    std = torch.exp(0.5 * log_var)
    return mean + std * noise
