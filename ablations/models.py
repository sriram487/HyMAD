import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.sincnet import SincConv1D
from model.transformer import SelfAttention, CrossAttention
from model.positional_encoding import PositionalEncoding
from model.model import SincNetRNN as FullHyMAD   # (f) re-exported for convenience

# ── Architecture constants (must match model/model.py) ────────────────────────
SINC_CH = 40    # SincNet out_channels
SINC_K  = 251   # SincNet kernel_size
POOL_T  = 64    # AdaptiveAvgPool1d output length
H       = 64    # RNN hidden size
HEADS   = 4     # attention heads
HD      = 32    # CrossAttention head_dim  →  4 × 32 = 128 per branch → 256 concat


# ── Shared utilities ──────────────────────────────────────────────────────────

class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _normalize(waveform: torch.Tensor) -> torch.Tensor:
    x = waveform.unsqueeze(1).to(torch.float32)
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-9)


class _SincFront(nn.Module):
    """SincNet → BN → ReLU → Pool → BN → ReLU → (B, POOL_T, SINC_CH)."""
    def __init__(self):
        super().__init__()
        self.sinc = SincConv1D(out_channels=SINC_CH, kernel_size=SINC_K, sample_rate=8000)
        self.bn1  = nn.BatchNorm1d(SINC_CH)
        self.pool = nn.AdaptiveAvgPool1d(POOL_T)
        self.bn2  = nn.BatchNorm1d(SINC_CH)

    def forward(self, x):
        x = F.relu(self.bn1(self.sinc(x)))
        x = F.relu(self.bn2(self.pool(x)))
        return x.transpose(1, 2)


class _Conv1dFront(nn.Module):
    """Standard Conv1d front-end — same shape as _SincFront but no bandpass constraint."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, SINC_CH, kernel_size=SINC_K, padding=SINC_K // 2)
        self.bn1  = nn.BatchNorm1d(SINC_CH)
        self.pool = nn.AdaptiveAvgPool1d(POOL_T)
        self.bn2  = nn.BatchNorm1d(SINC_CH)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv(x)))
        x = F.relu(self.bn2(self.pool(x)))
        return x.transpose(1, 2)


class _RNNBlock(nn.Module):
    """3-layer LSTM + LayerNorm."""
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(SINC_CH, H, num_layers=3, dropout=0.3, batch_first=True)
        self.ln  = nn.LayerNorm(H)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.ln(out)


# ── (a) No RNN ────────────────────────────────────────────────────────────────
class HyMAD_NoRNN(nn.Module):
    """Replace 3-layer RNN with a per-timestep Linear projection.
    Removes sequential recurrence; all other components unchanged."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.front   = _SincFront()
        self.proj    = nn.Linear(SINC_CH, H)   # no recurrence, just linear projection
        self.ln      = nn.LayerNorm(H)
        self.pe_s    = PositionalEncoding(d_model=SINC_CH)
        self.pe_r    = PositionalEncoding(d_model=H)
        self.sa_s    = SelfAttention(dim=SINC_CH, heads=HEADS)
        self.sa_r    = SelfAttention(dim=H,       heads=HEADS)
        self.ca_s2r  = CrossAttention(SINC_CH, H,       HEADS, HD)
        self.ca_r2s  = CrossAttention(H,       SINC_CH, HEADS, HD)
        self.dropout = nn.Dropout(0.3)
        self.head    = _MLPHead(256, num_classes)

    def forward(self, w):
        s = self.front(_normalize(w))                     # (B, 16, 20)
        r = F.relu(self.ln(self.proj(s)))                 # (B, 16, 64) — no recurrence
        s_sa = self.sa_s(self.pe_s(s))
        r_sa = self.sa_r(self.pe_r(r))
        a1 = self.ca_s2r(s_sa, r_sa).mean(1)
        a2 = self.ca_r2s(r_sa, s_sa).mean(1)
        return self.head(self.dropout(torch.cat([a1, a2], -1)))


# ── (b) Conv1d instead of SincNet ─────────────────────────────────────────────
class HyMAD_Conv1d(nn.Module):
    """Replace SincNet bandpass filterbank with unconstrained Conv1d.
    Removes the frequency-selective inductive bias; all other components unchanged."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.front   = _Conv1dFront()
        self.rnn_blk = _RNNBlock()
        self.pe_s    = PositionalEncoding(d_model=SINC_CH)
        self.pe_r    = PositionalEncoding(d_model=H)
        self.sa_s    = SelfAttention(dim=SINC_CH, heads=HEADS)
        self.sa_r    = SelfAttention(dim=H,       heads=HEADS)
        self.ca_s2r  = CrossAttention(SINC_CH, H,       HEADS, HD)
        self.ca_r2s  = CrossAttention(H,       SINC_CH, HEADS, HD)
        self.dropout = nn.Dropout(0.3)
        self.head    = _MLPHead(256, num_classes)

    def forward(self, w):
        s = self.front(_normalize(w))
        r = self.rnn_blk(s)
        s_sa = self.sa_s(self.pe_s(s))
        r_sa = self.sa_r(self.pe_r(r))
        a1 = self.ca_s2r(s_sa, r_sa).mean(1)
        a2 = self.ca_r2s(r_sa, s_sa).mean(1)
        return self.head(self.dropout(torch.cat([a1, a2], -1)))


# ── (c) No Self-Attention ──────────────────────────────────────────────────────
class HyMAD_NoSelfAttn(nn.Module):
    """Skip both self-attention blocks; feed positionally-encoded features
    directly to cross-attention. Tests intra-modal refinement contribution."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.front   = _SincFront()
        self.rnn_blk = _RNNBlock()
        self.pe_s    = PositionalEncoding(d_model=SINC_CH)
        self.pe_r    = PositionalEncoding(d_model=H)
        self.ca_s2r  = CrossAttention(SINC_CH, H,       HEADS, HD)
        self.ca_r2s  = CrossAttention(H,       SINC_CH, HEADS, HD)
        self.dropout = nn.Dropout(0.3)
        self.head    = _MLPHead(256, num_classes)

    def forward(self, w):
        s = self.front(_normalize(w))
        r = self.rnn_blk(s)
        s_pe = self.pe_s(s)
        r_pe = self.pe_r(r)
        a1 = self.ca_s2r(s_pe, r_pe).mean(1)
        a2 = self.ca_r2s(r_pe, s_pe).mean(1)
        return self.head(self.dropout(torch.cat([a1, a2], -1)))


# ── (d) Unidirectional Cross-Attention ────────────────────────────────────────
class HyMAD_UniCrossAttn(nn.Module):
    """Use only sinc→rnn cross-attention; drop the rnn→sinc direction.
    Tests whether bidirectional inter-modal interaction is necessary."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.front   = _SincFront()
        self.rnn_blk = _RNNBlock()
        self.pe_s    = PositionalEncoding(d_model=SINC_CH)
        self.pe_r    = PositionalEncoding(d_model=H)
        self.sa_s    = SelfAttention(dim=SINC_CH, heads=HEADS)
        self.sa_r    = SelfAttention(dim=H,       heads=HEADS)
        self.ca_s2r  = CrossAttention(SINC_CH, H, HEADS, HD)  # one direction only
        self.dropout = nn.Dropout(0.3)
        self.head    = _MLPHead(HEADS * HD, num_classes)       # 128, not 256

    def forward(self, w):
        s = self.front(_normalize(w))
        r = self.rnn_blk(s)
        s_sa = self.sa_s(self.pe_s(s))
        r_sa = self.sa_r(self.pe_r(r))
        a1 = self.ca_s2r(s_sa, r_sa).mean(1)                  # (B, 128)
        return self.head(self.dropout(a1))


# ── (e) Naive Fusion (mean-pool + concat) ─────────────────────────────────────
class HyMAD_NaiveFusion(nn.Module):
    """Replace both cross-attention blocks with mean-pool + linear-project + concat.
    Tests the value of cross-attention fusion over simple feature aggregation."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.front   = _SincFront()
        self.rnn_blk = _RNNBlock()
        self.pe_s    = PositionalEncoding(d_model=SINC_CH)
        self.pe_r    = PositionalEncoding(d_model=H)
        self.sa_s    = SelfAttention(dim=SINC_CH, heads=HEADS)
        self.sa_r    = SelfAttention(dim=H,       heads=HEADS)
        self.proj_s  = nn.Linear(SINC_CH, 128)  # project to match cross-attn output dim
        self.proj_r  = nn.Linear(H,       128)
        self.dropout = nn.Dropout(0.3)
        self.head    = _MLPHead(256, num_classes)

    def forward(self, w):
        s = self.front(_normalize(w))
        r = self.rnn_blk(s)
        s_sa = self.sa_s(self.pe_s(s))
        r_sa = self.sa_r(self.pe_r(r))
        s_p = self.proj_s(s_sa.mean(1))   # (B, 128)
        r_p = self.proj_r(r_sa.mean(1))   # (B, 128)
        return self.head(self.dropout(torch.cat([s_p, r_p], -1)))
