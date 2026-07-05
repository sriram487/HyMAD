import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.dropout_p = dropout
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(query_dim, self.inner_dim)
        self.k_proj = nn.Linear(context_dim, self.inner_dim)
        self.v_proj = nn.Linear(context_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.inner_dim)

    def forward(self, query, context):
        B, Tq, _ = query.shape
        _, Tc, _ = context.shape

        Q = self.q_proj(query).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, Tc, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, Tc, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).reshape(B, Tq, self.inner_dim)
        return self.out_proj(out)
