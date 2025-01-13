import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(self.config['n_emb'], self.config['head_size'], bias=False)
        self.query = nn.Linear(self.config['n_emb'], self.config['head_size'], bias=False)
        self.value = nn.Linear(self.config['n_emb'], self.config['head_size'], bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.config['block_size'], self.config['block_size'])))
        self.dropout = nn.Dropout(self.config['dropout'])


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
        self.config = config
        self.heads = nn.ModuleList([Head(self.config['head_size']) for _ in range(self.config['num_heads'])])
        self.proj = nn.Linear(self.config['head_size'] * self.config['num_heads'], self.config['n_emb'])
        self.dropout = nn.Dropout(self.config['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(self.config['n_emb'], 4 * self.config['n_embd']),
            nn.ReLU(),
            nn.Linear(4 * self.config['n_embd'], self.config['n_embd']),
            nn.Dropout(self.config['dropout'])
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.selfAttention = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.selfAttention(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x



class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embeding_table = nn.Embedding(self.config['vocab_size'], self.config['n_emb'])
        self.position_embedding_table = nn.Embedding(self.config['block_size'], self.config['n_emb'])
        self.blocks = nn.Sequential(*[Block(self.config['n_emb'], n_head=self.config['n_head']) for _ in range(self.config['n_layers'])])

        self.ln_f = nn.LayerNorm(self.config['n_emb'])
        self.lm_head = nn.Linear(self.config['n_emb'], self.config['vocab_size'])

        self.apply(self._init_weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B,T = index.shape

        tok_emb = self.token_embeding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    

    def generate(self, index, max_new_token):
        for _ in range(max_new_token):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next))
        return index
    

    def generate_latent(self, current_latent, latent_dim):
        """
        Generate the next latent space given the current latent space.

        Args:
            current_latent (Tensor): Current discrete latent space, shape (batch_size, latent_dim).
            latent_dim (int): Number of dimensions in the latent space.

        Returns:
            Tensor: Predicted next discrete latent space, shape (batch_size, latent_dim).
        """
        logits, _ = self.forward(current_latent)
        logits = logits[:, -1, :]  # Get logits for the last timestep
        probs = F.softmax(logits, dim=-1)

        # Sample for all dimensions of the latent space
        next_latent = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Shape: (batch_size, latent_dim)
        return next_latent
