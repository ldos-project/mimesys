import h5py
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from mimesys.models.diffusion import GaussianDiffusion
from mimesys.models.temporal_unet_cond import (
    ActionAutoencoder,
    ActionDeltaPredictor,
    CLIPContextEncoder,
    MLPCond,
    TemporalUnetCond,
    TransformerTracePredictor,
)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, beta, step_start_ema):
        self.beta = beta
        self.step_start_ema = step_start_ema
        self.step = 0

    def update_params(self, ema_params, new_params):
        if ema_params is None:
            raise ValueError("ema_params is None")
        return ema_params * self.beta + new_params * (1 - self.beta)

    def update_model_average(self, ema_model, current_model):
        for ema_p, cur_p in zip(ema_model.parameters(), current_model.parameters()):
            ema_p.data = self.update_params(ema_p.data, cur_p.data)

    def step_ema(self, ema_model, model):
        self.step += 1
        if self.step <= self.step_start_ema:
            self.reset_parameters(ema_model, model)
            return
        self.update_model_average(ema_model, model)

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def _load_checkpoint(model: nn.Module, ckpt_path: str, key_prefix: str, device: str):
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = state_dict.get("state_dict", state_dict)
    state_dict = {k.replace(key_prefix, ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Model initializers
# ---------------------------------------------------------------------------

def initialize_clip_model(cfg: DictConfig) -> CLIPContextEncoder:
    return CLIPContextEncoder(
        mod1_input_dim=cfg.clip.modality1_input_dim,
        mod2_input_dim=cfg.clip.modality2_input_dim,
        modality1_encoder=cfg.clip.modality1,
        modality2_encoder=cfg.clip.modality2,
        embed_dim=cfg.clip.embed_dim,
    )


def load_clip_model(cfg: DictConfig, device: str = "cuda") -> CLIPContextEncoder:
    model = initialize_clip_model(cfg.model)
    return _load_checkpoint(model, cfg.data.clip_ckpt_path, "trainer_model.", device)


def initialize_action_autoencoder_model(cfg: DictConfig) -> ActionAutoencoder:
    return ActionAutoencoder(cfg.autoencoder)


def initialize_action_delta_predictor_model(cfg: DictConfig) -> ActionDeltaPredictor:
    return ActionDeltaPredictor(
        n_dim=cfg.action_delta_predictor.input_dim,
        n_channels=cfg.action_delta_predictor.channel_dim,
        b_horizon=cfg.action_delta_predictor.horizon_dim,
        hidden_dim=cfg.action_delta_predictor.hidden_dim,
    )


def initialize_transformer_model(cfg: DictConfig) -> TransformerTracePredictor:
    return TransformerTracePredictor(cfg.transformer)


def load_transformer_model(cfg: DictConfig, device: str = "cuda") -> TransformerTracePredictor:
    model = initialize_transformer_model(cfg.model)
    return _load_checkpoint(model, cfg.data.transformer_ckpt_path, "trainer_model.", device)


def initialize_diffusion_model(cfg: DictConfig, model_arch: str = "unet") -> GaussianDiffusion:
    if model_arch == "unet":
        base_model = TemporalUnetCond(
            input_dim=cfg.unet.input_dim,
            hidden_dim=cfg.unet.hidden_dim,
            dim_head=cfg.unet.dim_head,
            num_heads=cfg.unet.num_heads,
            dropout=cfg.unet.dropout,
            context_args=cfg.context,
        )
    else:
        base_model = MLPCond(
            input_dim=cfg.mlp.input_dim,
            hidden_dim=cfg.mlp.hidden_dim,
            dropout=cfg.mlp.dropout,
            context_args=cfg.context,
        )
    diffusion = GaussianDiffusion(
        model=base_model,
        n_timesteps=cfg.diffusion.n_timesteps,
        clipped_denoised=cfg.diffusion.clipped_denoised,
        **cfg.diffusion.cfg_args,
    )
    # Optional variant (c): auxiliary row-sum→CPU% supervision weight.
    diffusion.row_sum_aux_weight = float(
        getattr(cfg.diffusion, "row_sum_aux_weight", 0.0)
    )
    # NEW: asymmetric sparsity penalty that zeros out idle positions.
    diffusion.sparsity_aux_weight = float(
        getattr(cfg.diffusion, "sparsity_aux_weight", 0.0)
    )
    return diffusion


def load_model(cfg: DictConfig, model_type: str, model_path: str, device: str):
    model = initialize_diffusion_model(cfg, model_arch=model_type)
    state_dict = torch.load(model_path, map_location=device)
    if "ema_state_dict" in state_dict:
        print("[load_model] Loading EMA weights.")
        state_dict = state_dict["ema_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training infrastructure
# ---------------------------------------------------------------------------

def initialize_logger(cfg: DictConfig) -> WandbLogger:
    return WandbLogger(project=cfg.project_name, name=cfg.run_name)


class CustomModelCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer, pl_module):
        print(
            f"CustomModelCheckpoint: epoch={trainer.current_epoch}, "
            f"started={trainer.fit_loop.epoch_progress.current.started}"
        )
        super().on_validation_end(trainer, pl_module)


def initialize_callbacks(cfg: DictConfig) -> list:
    checkpoint_callback = CustomModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        dirpath=cfg.checkpoint.dirpath,
        filename="diffusion-{epoch:02d}",
        save_top_k=cfg.checkpoint.save_top_k,
        every_n_epochs=cfg.checkpoint.every_n_epochs,
        save_last=cfg.checkpoint.get("save_last", False),
        save_on_train_epoch_end=cfg.checkpoint.get("save_on_train_epoch_end", None),
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_callback, lr_monitor]


def write_fleetbench_actions_to_h5_file(normalized_actions, filename, normalized=True, transpose=True):
    action_tensor = []
    for action_list in normalized_actions:
        processed = []
        for sub_action_list in action_list:
            if normalized:
                line = [max(0, (item.item() + 1) / 2) for item in sub_action_list]
            else:
                line = [max(0, item.item()) for item in sub_action_list]
            processed.append(line)
        if transpose:
            processed = list(map(list, zip(*processed)))
        action_tensor.append(processed)

    with h5py.File(filename, "w") as f:
        f.create_dataset("execution_plan", data=np.array(action_tensor))


def write_fleetbench_actions_to_file(normalized_actions, f, normalized=True):
    for action_list in normalized_actions:
        if normalized:
            line = ",".join(str((item.item() + 1) / 2) for item in action_list)
        else:
            line = ",".join(str(item.item()) for item in action_list)
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Action error metrics
# ---------------------------------------------------------------------------

def calculate_fleetbench_action_error(
    label_actions: list[list[float]],
    predicted_actions: list[list[float]],
) -> float:
    error = 0.0
    for label, pred in zip(label_actions, predicted_actions):
        error += np.linalg.norm(np.array(label) - np.array(pred), ord=1)
    return error
