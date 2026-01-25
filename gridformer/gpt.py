# gpt architecture for world model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import numpy as np
from collections import deque
from gridformer.Utils.logger import logger
from gridformer.Utils.utils import _to_jsonable
from safetensors.torch import save_file, load_file



class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], config['head_size'], bias=False)
        self.query = nn.Linear(config['n_embd'], config['head_size'], bias=False)
        self.value = nn.Linear(config['n_embd'], config['head_size'], bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = nn.Dropout(config['dropout'])


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out




class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config['n_head'])])
        self.proj = nn.Linear(config['head_size'] * config['n_head'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.ReLU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x):
        return self.net(x)
    


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.selfAttention = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])

    def forward(self, x):
        y = self.selfAttention(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    


class WMGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prev_state_embedding = nn.Linear(self.config['state_dim'], self.config['n_embd'])
        self.action_embedding     = nn.Embedding(self.config['action_size'], self.config['n_embd'])
        self.traj_embedding       = nn.Linear(self.config['n_embd']*2, self.config['n_embd']) # prev_state + action embeddings

        self.time_embedding = nn.Embedding(self.config['max_timestep'], self.config['n_embd'])
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config['n_layers'])])

        self.ln_f = nn.LayerNorm(self.config['n_embd'])
        self.obs_head = nn.Linear(self.config['n_embd'], config['input_dim'])
        self.reward_head = nn.Linear(self.config['n_embd'], self.config['num_reward_bins'])
        self.done_head = nn.Linear(self.config['n_embd'], 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=self.config['learning_rate'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, obs, action, step, targets=None):
        #B,T = obs.shape

        obs_emb = self.prev_state_embedding(obs)
        act_emb = self.action_embedding(action)

        fused = torch.cat((obs_emb, act_emb), dim=-1)
        token = self.traj_embedding(fused)
        step = step.to(obs.device).long()                   # (B,T) long on same device
        step = step % self.config['max_timestep']           # safety (even if you already cycle)
        pos = self.time_embedding(step)
        x = token + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        obs_logits = self.obs_head(x)
        reward_logits = self.reward_head(x)
        done_logits = self.done_head(x)


        return obs_logits, reward_logits, done_logits
    


    def save(self, path: str):
        """
        Save model + (optional) optimizer + config.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        torch.save(ckpt, path)

    def load(self, path: str, device=None):
        if device is None:
            device = next(self.parameters()).device

        ckpt = torch.load(path, map_location=device)

        # Optional safety check: config match
        if "config" in ckpt and ckpt["config"] != self.config:
            raise ValueError("Checkpoint config != current model config (create model with same config before load).")

        self.load_state_dict(ckpt["model_state_dict"])

        if hasattr(self, "optimizer") and self.optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # move optimizer state tensors to device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        logger.info(f"Loaded model from {path}.")

    


    def save_safetensors(self, path: str, save_optimizer: bool = True):
        """
        Save model weights (always) + optimizer state tensors (optional) using safetensors.
        Writes:
          - {path}.safetensors
          - {path}.json
          - {path}.optim.safetensors   (optional)
        """
        base, ext = os.path.splitext(path)
        if ext:  # user passed something like "ckpt.pt" -> use "ckpt"
            base = base
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

        model_path = base + ".safetensors"
        meta_path = base + ".json"
        optim_path = base + ".optim.safetensors"

        # 1) model weights
        model_sd = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        save_file(model_sd, model_path)

        # 2) metadata (config + optimizer key structure)
        meta = {
            "format": "safetensors_ckpt_v1",
            "config": _to_jsonable(getattr(self, "config", None)),
            "has_optimizer": False,
            "optimizer_state_keys": None,
        }

        # 3) optimizer (optional, best-effort)
        if save_optimizer and hasattr(self, "optimizer") and self.optimizer is not None:
            try:
                opt_sd = self.optimizer.state_dict()

                # Flatten ONLY tensor parts of optimizer state into a safetensors dict
                # Key format: state/{param_idx}/{state_key}
                # (We keep "param_groups" + non-tensor scalars in JSON)
                opt_tensors = {}
                opt_state_keys = {}  # helps debug / validate
                for param_idx, st in opt_sd.get("state", {}).items():
                    param_idx_str = str(param_idx)
                    opt_state_keys[param_idx_str] = []
                    for sk, sv in st.items():
                        if torch.is_tensor(sv):
                            key = f"state/{param_idx_str}/{sk}"
                            opt_tensors[key] = sv.detach().cpu()
                            opt_state_keys[param_idx_str].append(sk)

                # Save optimizer tensor state
                save_file(opt_tensors, optim_path)

                # Save remaining optimizer info in metadata
                meta["has_optimizer"] = True
                meta["optimizer_state_keys"] = opt_state_keys
                meta["optimizer_param_groups"] = _to_jsonable(opt_sd.get("param_groups", []))

                # Also save non-tensor optimizer scalars per param (e.g., step counts if ints)
                # We store them to JSON so load can restore if desired.
                opt_state_nontensor = {}
                for param_idx, st in opt_sd.get("state", {}).items():
                    p = str(param_idx)
                    opt_state_nontensor[p] = {}
                    for sk, sv in st.items():
                        if not torch.is_tensor(sv):
                            opt_state_nontensor[p][sk] = _to_jsonable(sv)
                meta["optimizer_state_nontensor"] = opt_state_nontensor

            except Exception as e:
                # If optimizer save fails, still save model + config
                meta["has_optimizer"] = False
                meta["optimizer_save_error"] = str(e)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_safetensors(self, path: str, device=None, load_optimizer: bool = True, strict: bool = True):
        """
        Load model weights (+ optional optimizer state) from safetensors-based checkpoint.
        Expects:
          - {path}.safetensors
          - {path}.json
          - {path}.optim.safetensors   (optional)
        """
        if device is None:
            device = next(self.parameters()).device

        base, ext = os.path.splitext(path)
        if ext:
            base = base

        model_path = base + ".safetensors"
        meta_path = base + ".json"
        optim_path = base + ".optim.safetensors"

        # metadata (optional but recommended)
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Optional safety check: config match
            if meta.get("config") is not None and hasattr(self, "config"):
                if meta["config"] != _to_jsonable(self.config):
                    raise ValueError("Checkpoint config != current model config (create model with same config before load).")

        # load model weights
        model_sd = load_file(model_path)  # tensors on CPU
        model_sd = {k: v.to(device) for k, v in model_sd.items()}
        self.load_state_dict(model_sd, strict=strict)

        # optimizer (best-effort)
        if (
            load_optimizer
            and hasattr(self, "optimizer")
            and self.optimizer is not None
            and meta is not None
            and meta.get("has_optimizer", False)
            and os.path.exists(optim_path)
        ):
            try:
                opt_sd = self.optimizer.state_dict()

                # Restore param_groups from meta (structure)
                if "optimizer_param_groups" in meta:
                    opt_sd["param_groups"] = meta["optimizer_param_groups"]

                # Restore non-tensor fields (if present)
                if "optimizer_state_nontensor" in meta:
                    for pidx, fields in meta["optimizer_state_nontensor"].items():
                        pidx_int = int(pidx)
                        if pidx_int not in opt_sd["state"]:
                            opt_sd["state"][pidx_int] = {}
                        opt_sd["state"][pidx_int].update(fields)

                # Restore tensor fields from optim safetensors
                opt_tensors = load_file(optim_path)  # CPU tensors
                # Keys are state/{param_idx}/{state_key}
                for key, tensor in opt_tensors.items():
                    _, pidx, sk = key.split("/", 2)
                    pidx_int = int(pidx)
                    if pidx_int not in opt_sd["state"]:
                        opt_sd["state"][pidx_int] = {}
                    opt_sd["state"][pidx_int][sk] = tensor.to(device)

                self.optimizer.load_state_dict(opt_sd)

            except Exception as e:
                # Don’t fail the whole load if optimizer restore breaks
                # (torch version changes often break optimizer states)
                if "logger" in globals():
                    logger.warning(f"Loaded model but skipped optimizer restore due to: {e}")
        if "logger" in globals():
            logger.info(f"Loaded safetensors checkpoint from {base}.")




class WMTrainer:
    def __init__(self, config):
        self.model = WMGPT(config)
        self.device = self.model.device
        self.model.to(self.device)

        self.num_reward_bins = 255
        self.reward_bins = torch.linspace(-6.0, 6.0, self.num_reward_bins, device=self.device)

    
    def symlog(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def two_hot_targets(self, x_symlog: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
        """
        x_symlog: (B,T,1) in symlog space
        bin_centers: (K,) sorted ascending (symlog space)
        returns: (B,T,K) two-hot soft target distribution
        """
        x = x_symlog.squeeze(-1)  # (B,T)
        K = bin_centers.shape[0]

        # find where x would be inserted
        idx = torch.searchsorted(bin_centers, x.clamp(bin_centers[0], bin_centers[-1]))
        idx = idx.clamp(1, K - 1)

        lo = idx - 1
        hi = idx

        lo_c = bin_centers[lo]  # (B,T)
        hi_c = bin_centers[hi]  # (B,T)

        hi_w = (x - lo_c) / (hi_c - lo_c + 1e-8)
        lo_w = 1.0 - hi_w

        target = torch.zeros(x.shape[0], x.shape[1], K, device=x.device, dtype=torch.float32)
        target.scatter_(2, lo.unsqueeze(-1), lo_w.unsqueeze(-1))
        target.scatter_(2, hi.unsqueeze(-1), hi_w.unsqueeze(-1))
        return target

    def soft_ce_loss(self, logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,T,K)
        target_dist: (B,T,K), sums to 1
        """
        logp = F.log_softmax(logits, dim=-1)
        return -(target_dist * logp).sum(dim=-1).mean()
    


    def train(self, train_loader, epochs=10, val_loader=None, log_every=1):
        """
        Train for multiple epochs.

        Args:
            train_loader: DataLoader for training
            epochs (int): number of epochs
            val_loader: optional DataLoader for validation
            log_every (int): print every N epochs

        Returns:
            dict: {"train_loss": [...], "val_loss": [...] (if val_loader provided)}
        """
        history = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        max_t = self.model.config["max_timestep"]
        act_n = self.model.config["action_size"]
        best_metric = math.inf          # lower is better
        best_epoch = 0
        patience = 10
        bad_epochs = 0
        best_path = getattr(self, "best_ckpt_path", None) or "checkpoints/best_model"

        for ep in range(1, epochs + 1):
            # -------- TRAIN --------
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                obs, action, reward, done, next_obs, step = batch

                # checks (CPU)
                if step.max().item() >= max_t or step.min().item() < 0:
                    raise ValueError(
                        f"[epoch {ep}] step out of range: [{step.min().item()}, {step.max().item()}], max_timestep={max_t}"
                    )
                if action.max().item() >= act_n or action.min().item() < 0:
                    raise ValueError(
                        f"[epoch {ep}] action out of range: [{action.min().item()}, {action.max().item()}], action_size={act_n}"
                    )

                # move to device
                obs = obs.to(self.device)
                action = action.to(self.device)
                reward = reward.to(self.device)
                done = done.to(self.device)
                next_obs = next_obs.to(self.device)
                step = step.to(self.device)

                # forward
                obs_pred, reward_pred, done_pred = self.model(obs, action, step)

                # losses
                loss_obs = F.mse_loss(obs_pred, next_obs)

                reward_symlog = self.symlog(reward)  # (B,T,1)
                reward_target_dist = self.two_hot_targets(reward_symlog, self.reward_bins)  # (B,T,K)
                loss_reward = self.soft_ce_loss(reward_pred, reward_target_dist)

                loss_done = F.binary_cross_entropy_with_logits(done_pred, done)

                loss = loss_obs + loss_reward + loss_done

                # backward
                self.model.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.model.optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)
            history["train_loss"].append(train_loss)

            # -------- VAL  --------
            if val_loader is not None:
                self.model.eval()
                val_total = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch in val_loader:
                        obs, action, reward, done, next_obs, step = batch

                        # (checks optional in val, but keep them for safety)
                        if step.max().item() >= max_t or step.min().item() < 0:
                            raise ValueError(
                                f"[val epoch {ep}] step out of range: [{step.min().item()}, {step.max().item()}], max_timestep={max_t}"
                            )
                        if action.max().item() >= act_n or action.min().item() < 0:
                            raise ValueError(
                                f"[val epoch {ep}] action out of range: [{action.min().item()}, {action.max().item()}], action_size={act_n}"
                            )

                        obs = obs.to(self.device)
                        action = action.to(self.device)
                        reward = reward.to(self.device)
                        done = done.to(self.device)
                        next_obs = next_obs.to(self.device)
                        step = step.to(self.device)

                        obs_pred, reward_pred, done_pred = self.model(obs, action, step)

                        loss_obs = F.mse_loss(obs_pred, next_obs)
                        reward_symlog = self.symlog(reward)
                        reward_target_dist = self.two_hot_targets(reward_symlog, self.reward_bins)
                        loss_reward = self.soft_ce_loss(reward_pred, reward_target_dist)
                        loss_done = F.binary_cross_entropy_with_logits(done_pred, done)

                        loss = loss_obs + loss_reward + loss_done

                        val_total += float(loss.item())
                        val_batches += 1

                val_loss = val_total / max(val_batches, 1)
                history["val_loss"].append(val_loss)
            
            current_metric = val_loss if (val_loader is not None) else train_loss
            if current_metric < (best_metric - 1e-8):
                best_metric = current_metric
                best_epoch = ep
                bad_epochs = 0
                self.model.save_safetensors(best_path)
                logger.info(f"[ckpt] saved best at epoch {ep}: metric={best_metric:.6f} -> {best_path}")

            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info(f"[early stop] no improvement for {patience} epochs. best_epoch={best_epoch}, best_metric={best_metric:.6f}")
                    break

            # -------- LOG --------
            if (ep % log_every) == 0:
                if val_loader is None:
                    logger.info(f"Epoch {ep}/{epochs} | train_loss: {train_loss:.6f}")
                else:
                    logger.info(f"Epoch {ep}/{epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        return history

