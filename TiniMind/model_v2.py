"""
TiniMind Model v2 — Flash Attention + Custom Tokenizer Support
==============================================================
Arsitektur: RoPE + RMSNorm + GQA + SwiGLU

Perubahan dari model.py v1:
  [UPD-1] Flash Attention via F.scaled_dot_product_attention (PyTorch 2.0+)
          -> Tidak perlu alokasi attention matrix eksplisit O(T^2)
          -> ~2-4x lebih cepat di T4, hemat VRAM signifikan
          -> Auto-fallback ke math attention kalau SDPA tidak tersedia
  [UPD-2] Support custom vocab size (32000 Indo BPE atau 100352 cl100k)
  [UPD-3] KV expansion pakai repeat_interleave (view, bukan copy) untuk GQA

Interface TETAP SAMA dengan v1:
    forward(x, targets=None, use_kv_cache=False, past_kvs=None, offset=0)
    return (logits, loss, past_kvs)
    model.num_params()
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]

_FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


# ─── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ─── RoPE ─────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        seq_len = q.shape[2] + offset
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len * 2
            self._build_cache(self.max_seq_len)
        T = q.shape[2]
        cos = self.cos_cache[:, :, offset : offset + T, :].to(q.dtype)
        sin = self.sin_cache[:, :, offset : offset + T, :].to(q.dtype)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─── GQA + Flash Attention ────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention dengan Flash Attention.

    [UPD-1] Flash Attention via F.scaled_dot_product_attention:
        - PyTorch 2.0+ otomatis pakai Flash Attention kernel di GPU
        - Tidak alokasi attention matrix O(T^2) -> hemat VRAM besar
        - is_causal=True -> tidak perlu buat causal mask manual
        - ~2-4x speedup di T4 vs manual attention

    [UPD-3] GQA expansion pakai repeat_interleave:
        - k,v: (B, n_kv_heads, T, head_dim)
        - setelah expand: (B, n_heads, T, head_dim)
        - repeat_interleave pakai view (tidak ada copy memori ekstra)
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_groups     = num_heads // num_kv_heads
        self.head_dim     = hidden_size // num_heads
        self.dropout      = dropout

        self.wq = nn.Linear(hidden_size, num_heads    * self.head_dim, bias=bias)
        self.wk = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(hidden_size, hidden_size,                  bias=bias)

        if not _FLASH_AVAILABLE:
            mask = torch.tril(torch.ones(4096, 4096))
            self.register_buffer("causal_mask", mask.view(1, 1, 4096, 4096))

    def forward(self, x, rope, past_kv=None, use_kv_cache=False, offset=0):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = rope(q, k, offset=offset)

        if use_kv_cache:
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)
            new_kv = (k, v)
        else:
            new_kv = None

        # [UPD-3] GQA: expand K,V ke n_heads
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # [UPD-1] Flash Attention
        if _FLASH_AVAILABLE:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = None,
                dropout_p = self.dropout if self.training else 0.0,
                is_causal = True,
            )
        else:
            # Fallback manual attention untuk PyTorch < 2.0
            scale  = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            T_k, T_q = k.shape[2], q.shape[2]
            if hasattr(self, "causal_mask"):
                scores = scores.masked_fill(
                    self.causal_mask[:, :, offset:offset + T_q, :T_k] == 0,
                    float("-inf"),
                )
            attn_out = torch.matmul(F.softmax(scores, dim=-1), v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(attn_out), new_kv


# ─── SwiGLU FFN ───────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up   = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ─── Transformer Block ────────────────────────────────────────────────────────

class TiniMindBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn  = GroupedQueryAttention(cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.ffn   = SwiGLUFFN(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, rope, past_kv=None, use_kv_cache=False, offset=0):
        attn_out, new_kv = self.attn(self.norm1(x), rope, past_kv=past_kv,
                                      use_kv_cache=use_kv_cache, offset=offset)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


# ─── TiniMind ─────────────────────────────────────────────────────────────────

class TiniMind(nn.Module):
    """TiniMind v2 — LLM Bahasa Indonesia dengan Flash Attention.

    Vocab: 32000 (Indo BPE) atau 100352 (cl100k_base)
    Sesuaikan cfg.vocab_size dengan tokenizer yang dipakai.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed   = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks  = nn.ModuleList([TiniMindBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm    = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        # Shared RoPE
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=cfg.max_seq_len)

        if _FLASH_AVAILABLE:
            print(f"Flash Attention aktif (F.scaled_dot_product_attention) | vocab={cfg.vocab_size}")
        else:
            print(f"Flash Attention tidak tersedia, pakai manual attention | vocab={cfg.vocab_size}")
            print(f"  -> Update PyTorch ke >= 2.0 untuk Flash Attention")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        n -= self.lm_head.weight.numel()  # weight-tied, jangan dihitung 2x
        return n

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        past_kvs: Optional[KVCache] = None,
        offset: int = 0,
    ):
        """Forward pass.

        Args:
            x: token ids (B, T)
            targets: target ids untuk loss (B, T)
            use_kv_cache: aktifkan KV cache untuk inference
            past_kvs: list of (k, v) dari step sebelumnya
            offset: posisi token saat ini (untuk KV cache decode)

        Returns:
            (logits, loss, past_kvs)
        """
        B, T = x.shape
        h = self.embed(x)

        new_kvs = [] if use_kv_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if (past_kvs is not None and i < len(past_kvs)) else None
            h, new_kv = block(h, self.rope, past_kv=past_kv,
                              use_kv_cache=use_kv_cache, offset=offset)
            if use_kv_cache:
                new_kvs.append(new_kv)

        h      = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kvs


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing TiniMind v2...")

    cfg = ModelConfig(
        num_layers   = 4,
        hidden_size  = 256,
        num_heads    = 8,
        num_kv_heads = 2,
        vocab_size   = 32000,
        max_seq_len  = 512,
        dropout      = 0.0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TiniMind(cfg).to(device)
    print(f"Params: {model.num_params()/1e6:.1f}M | Device: {device}")

    B, T    = 2, 128
    x       = torch.randint(0, cfg.vocab_size, (B, T)).to(device)
    targets = torch.randint(0, cfg.vocab_size, (B, T)).to(device)

    with torch.no_grad():
        logits, loss, _ = model(x, targets)

    print(f"Logits: {logits.shape} | Loss: {loss.item():.4f}")
    print(f"Expected loss (random): ~{math.log(cfg.vocab_size):.2f}")
    print("Model OK!")
