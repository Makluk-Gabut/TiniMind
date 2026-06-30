"""
TiniMind Configuration — Upgraded
Tambahan vs versi lama:
  [NEW-1] num_kv_heads — untuk GQA (Grouped Query Attention)
  [NEW-2] Config prod_500m — target ~500M params untuk T4
  [NEW-3] intermediate_size otomatis pakai rumus SwiGLU (8/3 × hidden)
          bukan 4× seperti FFN biasa
  [NEW-4] TiniConfig alias tetap ada untuk kompatibilitas infer.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Konfigurasi arsitektur TiniMind."""

    # Dimensi model
    num_layers: int = 24
    hidden_size: int = 1024
    num_heads: int = 16

    # [NEW-1] GQA: num_kv_heads < num_heads → hemat KV cache
    # Contoh: num_heads=16, num_kv_heads=4 → 4x lebih hemat dari MHA
    # Syarat: num_heads % num_kv_heads == 0
    num_kv_heads: int = 4

    # FFN intermediate size
    # SwiGLU pakai 8/3 × hidden (bukan 4×) supaya jumlah params setara
    # None → otomatis dihitung di __post_init__
    intermediate_size: int = 0

    # Vocabulary & sequence
    # [FIX-1] vocab_size HARUS > eot_token (100257) dari tiktoken cl100k_base
    # cl100k_base token IDs: 0-99999 (BPE) + 100257 (EOT) + 100258-100276 (FIM/special)
    # n_vocab tiktoken = 100277 → dibulatkan ke kelipatan 64 = 100352 (optimal GPU)
    # Bug lama (100257): EOT token ID 100257 OOB → CUDA device-side assert
    vocab_size: int = 100352         # tiktoken cl100k_base (n_vocab=100277, rounded ×64)
    max_seq_len: int = 2048
    block_size: int = 2048

    # Regularization
    dropout: float = 0.0            # 0.0 lebih umum untuk LLM besar

    def __post_init__(self):
        # Validasi
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) harus habis dibagi "
                f"num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) harus habis dibagi "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        # [NEW-3] Auto-compute intermediate_size untuk SwiGLU
        # Formula: round_to_multiple(8/3 × hidden, 256)
        if self.intermediate_size == 0:
            raw = int(8 / 3 * self.hidden_size)
            self.intermediate_size = (raw + 255) // 256 * 256

        est = self._estimate_params()
        logger.info(
            f"TiniMind config: {self.num_layers}L × {self.hidden_size}H "
            f"| {self.num_heads}Q-heads / {self.num_kv_heads}KV-heads "
            f"| ~{est/1e6:.0f}M params"
        )

    def _estimate_params(self) -> int:
        """Estimasi jumlah parameter."""
        head_dim = self.hidden_size // self.num_heads

        # Embedding (weight-tied dengan lm_head, jadi hitung sekali)
        embed = self.vocab_size * self.hidden_size

        per_block = (
            # RMSNorm x2 (ringan)
            2 * self.hidden_size +
            # GQA: Q proj + K proj + V proj + out proj
            self.hidden_size * self.num_heads * head_dim +       # Q
            self.hidden_size * self.num_kv_heads * head_dim +    # K
            self.hidden_size * self.num_kv_heads * head_dim +    # V
            self.hidden_size * self.hidden_size +                 # out
            # SwiGLU: gate + up + down
            self.hidden_size * self.intermediate_size +           # gate
            self.hidden_size * self.intermediate_size +           # up
            self.intermediate_size * self.hidden_size             # down
        )

        # Final RMSNorm
        final_norm = self.hidden_size

        # lm_head weight-tied → tidak tambah params
        return embed + self.num_layers * per_block + final_norm


# ─────────────────────────────────────────────────────────────────
#  QUICK TEST CONFIG (~60M params)
#  Untuk validasi pipeline: forward pass, training loop, inference
#  Target: jalan di CPU sekalipun, selesai dalam menit
# ─────────────────────────────────────────────────────────────────

@dataclass
class TestConfig:
    """Config ringan untuk test pipeline — ~60M params."""
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        num_layers=8,
        hidden_size=512,
        num_heads=8,
        num_kv_heads=2,
        vocab_size=100352,   # [FIX-1] sama seperti ModelConfig
        max_seq_len=512,
        block_size=512,
        dropout=0.1,
    ))

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 4
    num_epochs: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Data
    use_streaming: bool = True
    num_workers: int = 0

    # Logging
    log_every: int = 20
    eval_every: int = 200
    save_every: int = 500


# ─────────────────────────────────────────────────────────────────
#  PRODUCTION CONFIG — ~500M params
#  Target hardware: T4 (16GB VRAM)
#  Estimasi training:
#    - 1B token × bf16 × grad_accum=8 → ~6-10 jam per 1B token
#    - Mulai dari 1B token untuk proof of concept
# ─────────────────────────────────────────────────────────────────

@dataclass
class ProductionConfig:
    """Config utama TiniMind ~500M params untuk T4.

    Konsep epoch:
    - 1 epoch = 2000 optimizer steps (bukan 1x jalan semua data)
    - Save otomatis setiap akhir epoch = setiap 2000 step
    - 1B token total / ~65M token per epoch = ~15 epoch
    - Ada 15 checkpoint tersimpan di Drive

    Kalkulasi token per epoch:
    - batch_size=2, accum=8, block=2048
    - effective_batch = 2 x 8 x 2048 = 32768 token/step
    - 2000 step x 32768 = ~65M token per epoch

    VRAM estimate T4 (16GB):
    - Model weights (bf16): ~1.2GB
    - Activations + gradients + optimizer: ~10-12GB
    - Total: ~13-14GB pas dengan gradient checkpointing
    """
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        num_layers=32,
        hidden_size=1280,
        num_heads=20,
        num_kv_heads=5,
        vocab_size=100352,   # [FIX-1]
        max_seq_len=2048,
        block_size=2048,
        dropout=0.0,
    ))

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 2
    steps_per_epoch: int = 3000        # 1 epoch = 3000 optimizer step
    num_epochs: int = 10               # 10 epoch x 3000 step = 30000 step total (~1B token)
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 200

    # Data
    use_streaming: bool = True
    num_workers: int = 2

    # Logging — save per epoch + berkala setiap 1500 step
    log_every: int = 50
    eval_every: int = 500
    # [FIX-2] save_every=0 menyebabkan ZeroDivisionError (global_step % 0)
    # pada step pertama yang tidak ketemu eval_every.
    # Set 1500 = 2x save per epoch (1 epoch = 3000 step)
    save_every: int = 1500


# ─────────────────────────────────────────────────────────────────
#  MEDIUM CONFIG — ~130M params
#  Untuk membuktikan arsitektur works sebelum scale ke 500M
#  Lebih cepat 4x dari prod_500m
# ─────────────────────────────────────────────────────────────────

@dataclass
class MediumConfig:
    """Config medium ~130M params.
    
    1 epoch = 2000 optimizer step (~65M token)
    5 epoch total = ~325M token
    """
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        num_layers=16,
        hidden_size=768,
        num_heads=12,
        num_kv_heads=4,
        vocab_size=100352,   # [FIX-1]
        max_seq_len=2048,
        block_size=2048,
        dropout=0.0,
    ))

    learning_rate: float = 3e-4
    batch_size: int = 4
    steps_per_epoch: int = 2000
    num_epochs: int = 5
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 300

    use_streaming: bool = True
    num_workers: int = 2

    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000   # [FIX-2] bukan 0


# [NEW-4] Alias untuk kompatibilitas infer.py dan Train.py lama
TiniConfig = ModelConfig


def get_config(name: str) -> Union[TestConfig, MediumConfig, ProductionConfig]:
    """Load config by name."""
    configs = {
        "test_60m":   TestConfig(),
        "medium_130m": MediumConfig(),
        "prod_500m":  ProductionConfig(),
    }

    if name not in configs:
        raise ValueError(
            f"Unknown config: '{name}'\n"
            f"Tersedia: {', '.join(configs.keys())}"
        )

    cfg = configs[name]
    logger.info(f"Loaded config: {name}")
    return cfg
