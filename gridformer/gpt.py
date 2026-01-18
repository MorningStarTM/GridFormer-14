import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from collections import deque



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
        self.reward_head = nn.Linear(self.config['n_embd'], 1)
        self.done_head = nn.Linear(self.config['n_embd'], 1)

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
        print(f"fused shape: {fused.shape}")
        token = self.traj_embedding(fused)
        print(f"token shape: {token.shape}")
        pos = self.time_embedding(torch.tensor(step, dtype=torch.long, device=self.device))
        print(f"pos shape: {pos.shape}")
        x = token + pos
        print(f"x shape before blocks: {x.shape}")
        x = self.blocks(x)
        print(f"x shape after blocks: {x.shape}")
        x = self.ln_f(x)
        print(f"x shape after ln_f: {x.shape}")

        obs_logits = self.obs_head(x)
        reward_logits = self.reward_head(x)
        done_logits = self.done_head(x)


        return obs_logits, reward_logits, done_logits