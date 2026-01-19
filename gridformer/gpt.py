import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from collections import deque
from gridformer.Utils.logger import logger



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
        pos = self.time_embedding(torch.tensor(step, dtype=torch.long, device=self.device))
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
    

    def train(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            obs = batch['obs'].to(self.device)           # (B, T, state_dim)
            action = batch['action'].to(self.device)     # (B, T)
            reward = batch['reward'].to(self.device)     # (B, T, 1)
            done = batch['done'].to(self.device)         # (B, T, 1)
            step = batch['step'].to(self.device)         # (B, T)

            obs_pred, reward_pred, done_pred = self.model(obs, action, step)

            loss_obs = F.mse_loss(obs_pred, obs)
            reward_symlog = self.symlog(reward)  # (B,T,1)
            reward_target_dist = self.two_hot_targets(reward_symlog, self.reward_bins)  # (B,T,K)
            loss_reward = self.soft_ce_loss(reward_pred, reward_target_dist)

            loss_done = F.binary_cross_entropy_with_logits(done_pred, done)

            loss = loss_obs + loss_reward + loss_done

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
        