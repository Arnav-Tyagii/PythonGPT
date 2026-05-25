import torch.nn as nn
from model.attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):  # x: (B, T, C)
        x = x + self.attn(self.ln1(x))    # pre-norm attention + residual
        x = x + self.ffn(self.ln2(x))     # pre-norm FFN + residual
        return x                          # (B, T, C)
