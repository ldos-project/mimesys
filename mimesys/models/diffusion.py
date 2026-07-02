import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

from mimesys.models.diffusion_components import (
    cosine_beta_schedule,
    diffusion_sample_fn,
)


def extract(val, t, x_shape):
    """
    val: T
    t: B -> require to be type long
    x_shape: provides the length of the input
    [helper] get the value from val indexed by each t value
    using reshape here because val.gather may return tensor not contiguous
    """
    b, *_ = x_shape
    out = val.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_timesteps(t, batch_size, device):
    return torch.full((batch_size,), t, device=device, dtype=torch.long)


class GaussianDiffusion(nn.Module):
    def __init__(
        self, model, n_timesteps, clipped_denoised=False, cfg_drop_prob=0, cfg_guide_w=0
    ):
        super().__init__()
        """
        this gaussian diffusion model will condition on observation cond and hard condition for trajectories
        """
        print(f"Gaussian Diffusion Model with {n_timesteps} timesteps")
        beta = cosine_beta_schedule(n_timesteps)
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        alpha_hat_prev = torch.cat([torch.ones(1), alpha_hat[:-1]])

        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)
        self.register_buffer("alpha_hat_prev", alpha_hat_prev)

        self.register_buffer("sqrt_alpha_hat", torch.sqrt(alpha_hat))
        self.register_buffer("sqrt_one_minus_alpha_hat", torch.sqrt(1 - alpha_hat))
        self.register_buffer("sqrt_recip_alpha_hat", torch.sqrt(1 / alpha_hat))
        self.register_buffer("sqrt_recip_alpha_hat_m1", torch.sqrt(1 / alpha_hat - 1))

        posterior_variance = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta
        self.register_buffer(
            "posterior_mean_coeff_0",
            torch.sqrt(alpha_hat_prev) * beta / (1 - alpha_hat),
        )
        self.register_buffer(
            "posterior_mean_coeff_t",
            torch.sqrt(alpha) * (1 - alpha_hat_prev) / (1 - alpha_hat),
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )

        self.clipped_denoised = clipped_denoised
        self.n_timesteps = n_timesteps
        # weighted loss
        # Use relative error: MSE of (predicted - target) / (target + epsilon)
        self.epsilon = 1e-6  # small value to avoid division by zero
        self.threshold = 3  # Initialize threshold
        # self.loss_fn = lambda pred, target: nn.functional.mse_loss(
        #     (pred - target) / (target + self.epsilon), torch.zeros_like(target)
        # ) if torch.all((target >= -self.threshold) & (target <= self.threshold)) else nn.functional.mse_loss(pred, target)

        def loss_fn(pred, target):
            print(f"{torch.max(torch.abs(target)).item()}")
            if torch.all(torch.abs(pred - target) <= self.threshold):
                print(f"Using relative MSE loss: {torch.max(torch.abs(pred - target)).item()}")
                return nn.functional.mse_loss(
                    (pred - target) / torch.clamp(target + self.epsilon, min=self.epsilon),
                    torch.zeros_like(target)
                )
            else:
                return nn.functional.mse_loss(pred, target)

        def loss_fn(pred, target):
            return nn.functional.mse_loss(pred, target)

        # def loss_fn(pred, target, diversity_score):
        #     return (nn.functional.mse_loss(pred, target, reduction='none') * diversity_score.view(-1, 1, 1)).mean()

        self.loss_fn = nn.MSELoss() # loss_fn

        self.model = model
        self.cfg_drop_prob = cfg_drop_prob
        self.cfg_guide_w = cfg_guide_w
        # Mode-collapse penalty: encourage per-feature variance of model output
        # across the batch to match the variance of the noise targets. Mitigates
        # the failure mode where the diffusion model collapses to the conditional
        # mean (averaging over multi-modal action distributions per cond).
        # Enable via env var MIMESYS_DIVERSITY_WEIGHT=<float>.
        self.diversity_weight = float(os.environ.get("MIMESYS_DIVERSITY_WEIGHT", "0"))

        if self.cfg_drop_prob > 0:
            print(f"Training with CFG with dropout probability of {self.cfg_drop_prob}")
        if self.cfg_guide_w > 0:
            print(f"Evaluating with CFG with guidance weight of {self.cfg_guide_w}")
        if self.diversity_weight > 0:
            print(f"Training with variance-matching aux loss (weight={self.diversity_weight})")

    def update_threshold(self, new_threshold):
        """
        Dynamically update the threshold value.
        """
        self.threshold = new_threshold

    def predict_start_from_noise(self, xt, t, noise):
        """
        [for infernece] derived from forward diffusion process
        xt = sqrt(a_hat) x0 + sqrt(1 - a_hat) e
        => x0 = sqrt(1 / a_hat) xt - sqrt(1 / a_hat - 1) e
        """
        return (
            extract(self.sqrt_recip_alpha_hat, t, xt.shape) * xt
            - extract(self.sqrt_recip_alpha_hat_m1, t, xt.shape) * noise
        )

    def q_posterior(self, xt, t, x0):
        """
        [for inference] estimate true posterior given xt and x0
        x_{t-1} ~ q(x_{t-1} | x_t, x0, t)
        => mu = (sqrt(a_hat_{t-1}) b_t / (1 - a_hat_t)) x0 + (sqrt(a_t) (1 - a_hat_{t-1}) / (1 - a_hat_t)) x_t
        => variance = (1 - a_hat_{t-1}) / (1 - a_hat) * beta
        => log_var_clipped = log(clamp(variance))
        """
        mean = (
            extract(self.posterior_mean_coeff_0, t, xt.shape) * x0
            + extract(self.posterior_mean_coeff_t, t, xt.shape) * xt
        )
        variance = extract(self.posterior_variance, t, xt.shape)
        log_var_clipped = extract(self.posterior_log_var_clipped, t, xt.shape)
        return mean, variance, log_var_clipped

    def p_mean_variance(self, xt, t, context_cond=None):
        """
        [for inference] estimate prior mean and variance given x_t; there is observation condition for predicting the noise and hard state condition to follow
        => e = model(xt, t, cond) [the noise is the noise from start to t]
        => x0_recon = predict_start_from_noise(xt, t, e)
        => need to clip denoised x0_recon
        => p = q_posterior(xt, t, x0_recon)
        """
        cfg_mask_cond = torch.ones(xt.shape[0], device=xt.device)
        noise_cond = self.model(xt, t, context_cond, cfg_mask_cond)

        if self.cfg_guide_w > 1:
            cfg_mask_uncond = torch.zeros(xt.shape[0], device=xt.device)
            noise_uncond = self.model(xt, t, context_cond, cfg_mask_uncond)
            noise = noise_uncond + self.cfg_guide_w * (noise_cond - noise_uncond)
        else:
            noise = noise_cond

        x0_recon = self.predict_start_from_noise(xt, t, noise)
        if self.clipped_denoised:  # clip the denoised x0_recon
            x0_recon = torch.clamp(x0_recon, -1, 1)

        mean, variance, log_var_clipped = self.q_posterior(xt, t, x0_recon)

        return mean, variance, log_var_clipped

    def p_sample_loop(self, shape, context_cond=None):
        """
        [for inference] sample x0 from random noise
        => initialize xt
        => for t=T to 0
        =>   convert t to tensor
        =>   xt = sample_fn(self, xt, t, context_cond) [can be either ddpm sample, ddim sample, or guided ddpm]
        =>   apply hard condition on xt [optional]
        """
        device = self.alpha.device
        batch_size = shape[0]

        xt = torch.randn(*shape, device=device)
        chain = [xt]

        for t in reversed(range(self.n_timesteps)):
            t = make_timesteps(t, batch_size, device)
            xt = diffusion_sample_fn(
                self, xt, t, context_cond=context_cond
            )  # simple ddpm sample function
            chain.append(xt)

        return chain

    def p_sample_loop_with_logprobs(self, shape, context_cond=None):
        """
        [for inference] sample x0 from random noise
        => initialize xt
        => for t=T to 0
        =>   convert t to tensor
        =>   xt = sample_fn(self, xt, t, context_cond) [can be either ddpm sample, ddim sample, or guided ddpm]
        =>   apply hard condition on xt [optional]
        """
        device = self.alpha.device
        batch_size = shape[0]

        xt = torch.randn(*shape, device=device)
        chain = [xt]
        logprobs = []
        logprobs_raw = []

        for t_idx in reversed(range(self.n_timesteps)):
            t = make_timesteps(t_idx, batch_size, device)
            mean, _, log_var = self.p_mean_variance(xt, t, context_cond)

            # Calculate log probability of xt given mean and variance
            std = torch.exp(0.5 * log_var)
            std_clipped = torch.clip(std, min=1e-6) # to avoid numerical instability
            normal_dist = Normal(mean, std_clipped)

            xt = normal_dist.sample()
            log_prob = normal_dist.log_prob(xt)
            log_prob = log_prob.mean(dim=list(range(1, log_prob.ndim))) # sum over the state dimension
            logprobs.append(log_prob)
            chain.append(xt)

        # final_xt = chain[-1]
        # for log_prob in logprobs_raw:
        #     # Mask specific action regions
        #     action_mask = torch.zeros_like(final_xt, dtype=torch.bool)
        #     # top3_threads = torch.topk(torch.sum(final_xt, dim=2), 3, dim=1).indices
        #     top3_threads = torch.ones(batch_size, dtype=torch.int)
        #     for i in range(action_mask.shape[0]):
        #         action_mask[i, top3_threads[i]] = True

        #     # Apply mask to log probabilities
        #     log_prob_new = log_prob * action_mask
        #     log_prob_new = log_prob_new.mean(dim=list(range(1, log_prob_new.ndim)))
        #     logprobs.append(log_prob_new)

        # Stack logprobs to shape (n_timesteps, batch_size)
        logprobs = torch.stack(logprobs, dim=0)
        chain = torch.stack(chain, dim=0)  # (n_timesteps+1, batch_size, C, H, W)

        return chain, logprobs

    def get_logprob_for_ddpo(self, context_cond, xt, prev_xt, t_idx, batch_size):
        device = self.alpha.device
        t = make_timesteps(t_idx, batch_size, device)
        mean, _, log_var = self.p_mean_variance(xt, t, context_cond)
        # Calculate log probability of xt given mean and variance

        std = torch.exp(0.5 * log_var)
        std_clipped = torch.clip(std, min=1e-6) # to avoid numerical instability
        normal_dist = Normal(mean, std_clipped)
        logprob = normal_dist.log_prob(prev_xt)
        logprob = logprob.view(batch_size, -1).sum(dim=1)  # sum over all dims except batch

        return logprob


    def q_sample(self, x0, t, noise=None):
        """
        [for training] get x_t from x0, t, and noise
        => x_t = sqrt(a_hat) x0 + sqrt(1 - a_hat) e
        """
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alpha_hat, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_hat, t, x0.shape) * noise
        )

    def p_losses(self, x0, t, context_cond, diversity_score):
        """
        [for training] compute the loss given particular x0, cond, and t
        => sample noise
        => xt = q_sample(x0, t, noise)
        => noise_pred = model(xt, t, cond)
        => loss = loss_fn(noise, noise_pred)
        """
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        cfg_mask = torch.ones(xt.shape[0], device=xt.device)
        cfg_mask[np.random.rand(xt.shape[0]) < self.cfg_drop_prob] = 0

        noise_pred = self.model(xt, t, context_cond, cfg_mask)
        assert noise.shape == noise_pred.shape

        loss = self.loss_fn(noise_pred, noise)

        # Optional variant (c): auxiliary row-sum supervision. The trainer sets
        # `self.row_sum_aux_weight > 0` to enable. Decodes the noise prediction
        # back to a predicted x0, sums each thread row's stressor weights, and
        # compares against the per-core CPU% target (alphabetical metric layout:
        # indices [0:20] = per-core CPU 0..19). Encourages the simple algebraic
        # relationship `row_sum ∝ CPU%` that the U-Net's spatial mixing makes
        # hard to learn directly.
        # Variance-matching auxiliary loss: penalize when per-feature variance
        # of model predictions across the batch differs from variance of the
        # noise targets. The diffusion model otherwise collapses to predicting
        # the conditional mean (averaging multi-modal actions per cond), so
        # forcing variance parity preserves output diversity.
        if self.diversity_weight > 0.0 and noise_pred.shape[0] > 1:
            pred_var = noise_pred.flatten(1).var(dim=0)
            target_var = noise.flatten(1).var(dim=0)
            diversity_aux = F.mse_loss(pred_var, target_var)
            loss = loss + self.diversity_weight * diversity_aux

        row_sum_w = getattr(self, "row_sum_aux_weight", 0.0)
        sparsity_w = getattr(self, "sparsity_aux_weight", 0.0)
        if row_sum_w > 0.0 or sparsity_w > 0.0:
            x0_pred = (
                xt - extract(self.sqrt_one_minus_alpha_hat, t, x0.shape) * noise_pred
            ) / extract(self.sqrt_alpha_hat, t, x0.shape)
            # x0_pred conventions: label is stored as (B, S=stressors, T=threads),
            # i.e. axis 1 = stressors (13 legacy / 19 v2 / 20 v3) and axis -1 = threads (20).
            if x0_pred.dim() == 3 and x0_pred.shape[1] in (13, 19, 20) and x0_pred.shape[-1] == 20:
                # Clamp to the [0,1] weight space the action_pred is *supposed* to
                # live in. Without an upper clamp, large noise-prediction errors at
                # high diffusion timesteps explode row_sums (≥ 19·max_pred) and
                # make this aux dominate the base MSE.
                action_pred = ((x0_pred + 1.0) * 0.5).clamp(0.0, 1.0)
                row_sums = action_pred.sum(dim=1)                        # sum over stressors -> (B, T=20)
                # Conditioning is normalized to [-1, 1] in our v2 dataloader; map
                # back to [0, 1] for scale match with row_sums (also in [0, 1] for
                # active rows since one or two dominant stressors carry the weight).
                metric = context_cond["metric"]
                if metric.dim() == 2 and metric.shape[1] >= 20:
                    cpu_target = ((metric[:, :20] + 1.0) * 0.5).clamp(0, 1)  # (B, 20)
                    sample_mask = cfg_mask.view(-1, 1)                       # (B, 1)

                    # row_sum aux: L1 match between row_sum and target CPU.
                    # Symmetric — pulls row_sum toward the cond-implied magnitude.
                    if row_sum_w > 0.0:
                        aux = F.l1_loss(row_sums * sample_mask,
                                        cpu_target * sample_mask)
                        loss = loss + row_sum_w * aux

                    # NEW: sparsity aux. Asymmetric — penalize ALL stressor weight
                    # (per-thread sum) where cond shows the core is idle.
                    # idle_weight = (1 - cpu_target) → high when cond says core idle.
                    # This explicitly drives the model to zero out positions the
                    # conditioning implies are inactive, addressing the K-overshoot
                    # observed when the model puts non-trivial intensity on every
                    # row regardless of cond.
                    if sparsity_w > 0.0:
                        idle_weight = (1.0 - cpu_target)                     # (B, 20)
                        sparsity_aux = ((row_sums * idle_weight) * sample_mask).mean()
                        loss = loss + sparsity_w * sparsity_aux

        return loss

    def loss(self, x0, context_cond, diversity_score = []):
        """
        [for training] compute the loss for some t
        => sample t
        => loss = p_losses(x0, cond, t)
        """
        t = torch.randint(
            0, self.n_timesteps, (x0.shape[0],), device=x0.device, dtype=torch.long
        )
        return self.p_losses(x0, t, context_cond, diversity_score)

    def forward(self, x, context_cond):
        return self.loss(x, context_cond)
