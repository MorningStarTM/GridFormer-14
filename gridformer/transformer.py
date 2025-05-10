import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, input_dim:int):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.embedding = nn.Linear(self.input_dim, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)


        #create matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even positions and cos for odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    


class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropput: nn.Dropout):
        d_k = query.shape[-1]

        #(batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1) #(batch, h, seq_len, seq_len)

        if dropput is not None:
            attention_score = dropput(attention_score)
        
        return (attention_score @ value), attention_score
    


    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) ---> (batch, seq_len, d_model)

        #(Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    



class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            return self.norm(x)
        



class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

    

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    



class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, obs_dim):
        super().__init__()
        self.next_obs_head = nn.Linear(d_model, obs_dim)
        self.reward_head = nn.Linear(d_model, 1)
        self.done_head = nn.Linear(d_model, 1)  # sigmoid later


    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            next_obs: (batch_size, seq_len, obs_dim)
            reward: (batch_size, seq_len)
            done: (batch_size, seq_len)
        """
        next_obs = self.next_obs_head(x)
        reward = self.reward_head(x).squeeze(-1)  # (B, seq_len)
        done = torch.sigmoid(self.done_head(x).squeeze(-1))  # (B, seq_len), sigmoid for probability

        return next_obs, reward, done

    

class GridFormer(nn.Module):
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 seq_len: int,
                 d_model: int = 512,
                 n_layers: int = 3,
                 n_heads: int = 4,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        # Embedding layers for input (observation + action)
        self.src_embed = InputEmbedding(d_model, obs_dim + action_dim)
        self.tgt_embed = InputEmbedding(d_model, obs_dim + action_dim)

        # Positional encodings
        self.src_pos = PositionalEncoding(d_model, seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, seq_len, dropout)

        # Encoder
        encoder_layers = nn.ModuleList([
            EncoderBlock(
                MultiHeadAttentionBlock(d_model, n_heads, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            ) for _ in range(n_layers)
        ])
        self.encoder = Encoder(encoder_layers)

        # Decoder
        decoder_layers = nn.ModuleList([
            DecoderBlock(
                MultiHeadAttentionBlock(d_model, n_heads, dropout),
                MultiHeadAttentionBlock(d_model, n_heads, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            ) for _ in range(n_layers)
        ])
        self.decoder = Decoder(decoder_layers)

        # Output projection layer
        self.projection = ProjectionLayer(d_model, obs_dim)

        # Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, action, src_mask=None, tgt_mask=None):
        src = self.src_pos(self.src_embed(torch.cat([obs, action], dim=-1)))
        tgt = self.tgt_pos(self.tgt_embed(torch.cat([obs, action], dim=-1)))  

        memory = self.encoder(src, src_mask)
        decoded = self.decoder(tgt, memory, src_mask, tgt_mask)

        next_obs, reward, done = self.projection(decoded)
        return next_obs, reward, done
