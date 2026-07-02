"""
TiniMind Configuration
========================
4 ukuran model tersedia:
  tiny_130m  — 16L x 768H,  12Q/4KV  | ~125M  | T4 ✓
  prod_300m  — 24L x 1024H, 16Q/4KV  | ~303M  | T4 ✓  ← default training saat ini
  prod_500m  — 32L x 1152H, 18Q/6KV  | ~490M  | A100 40GB+
  prod_1b    — 38L x 1536H, 24Q/8KV  | ~1.01B | A100 80GB+

Catatan:
- vocab_size=32000 untuk tokenizer/indo_bpe_32k.model (default semua config)
- prod_300m adalah config yang AKTUAL dipakai di TiniMind_Training.ipynb
- Preset ini hanya titik awal — override via CLI train.py (--lr, --batch-size, dll)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Konfigurasi arsitektur TiniMind."""

    num_layers:        int   = 24
    hidden_size:       int   = 1024
    num_heads:         int   = 16
    # GQA: num_kv_heads < num_heads → hemat KV cache
    # Syarat: num_heads % num_kv_heads == 0
    num_kv_heads:      int   = 4
    # SwiGLU intermediate: 8/3 × hidden (auto-hitung jika 0)
    intermediate_size: int   = 0
    vocab_size:        int   = 32000   # Indo BPE tokenizer
    max_seq_len:       int   = 2048
    block_size:        int   = 2048
    dropout:           float = 0.0

    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) harus habis dibagi num_heads ({self.num_heads})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) harus habis dibagi num_kv_heads ({self.num_kv_heads})")
        if self.intermediate_size == 0:
            raw = int(8 / 3 * self.hidden_size)
            self.intermediate_size = (raw + 255) // 256 * 256
        logger.info(f"TiniMind: {self.num_layers}L x {self.hidden_size}H | "
                    f"{self.num_heads}Q/{self.num_kv_heads}KV | ~{self._est()/1e6:.0f}M params")

    def _est(self) -> int:
        hd = self.hidden_size // self.num_heads
        pb = (2 * self.hidden_size
              + self.hidden_size * self.num_heads    * hd
              + self.hidden_size * self.num_kv_heads * hd * 2
              + self.hidden_size * self.hidden_size
              + self.hidden_size * self.intermediate_size * 2
              + self.intermediate_size * self.hidden_size)
        return self.vocab_size * self.hidden_size + self.num_layers * pb + self.hidden_size


# Alias untuk kompatibilitas file lama (infer.py, dll)
TiniConfig = ModelConfig


# ─── Training configs ───────────────────────────────────────────

@dataclass
class Tiny130MConfig:
    """~125M params — muat di T4, cocok untuk eksperimen cepat."""
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        num_layers=16, hidden_size=768, num_heads=12, num_kv_heads=4, vocab_size=32000))
    learning_rate: float = 3e-4
    batch_size: int = 8
    steps_per_epoch: int = 2000
    num_epochs: int = 10
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    use_streaming: bool = True
    num_workers: int = 2
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 500


@dataclass
class Prod300MConfig:
    """~303M params — T4 ✓ — config AKTUAL yang dipakai training saat ini.

    24L x 1024H, 16Q/4KV, vocab=32000
    batch=4, grad_accum=8 → ~32K token/step efektif
    VRAM T4: ~11-13GB dengan fp16
    """
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        num_layers=24, hidden_size=1024, num_heads=16, num_kv_heads=4, vocab_size=32000))
    learning_rate: float = 3e-4
    batch_size: int = 4
    steps_per_epoch: int = 2000
    num_epochs: int = 10
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    use_streaming: bool = True
    num_workers: int = 2
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 1000


@dataclass
class Prod500MConfig:
    """~490M params — butuh A100 40GB+, tidak muat di T4.

    32L x 1152H, 18Q/6KV, vocab=32000
    """
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        num_layers=32, hidden_size=1152, num_heads=18, num_kv_heads=6, vocab_size=32000))
    learning_rate: float = 2e-4
    batch_size: int = 4
    steps_per_epoch: int = 2000
    num_epochs: int = 15
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    use_streaming: bool = True
    num_workers: int = 4
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 1000


@dataclass
class Prod1BConfig:
    """~1.01B params — butuh A100 80GB+.

    38L x 1536H, 24Q/8KV, vocab=32000
    """
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        num_layers=38, hidden_size=1536, num_heads=24, num_kv_heads=8, vocab_size=32000))
    learning_rate: float = 1e-4
    batch_size: int = 2
    steps_per_epoch: int = 2000
    num_epochs: int = 15
    gradient_accumulation_steps: int = 32
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    use_streaming: bool = True
    num_workers: int = 4
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 1000


# ─── Lookup ─────────────────────────────────────────────────────

_CONFIGS = {
    "tiny_130m": Tiny130MConfig,
    "prod_300m": Prod300MConfig,
    "prod_500m": Prod500MConfig,
    "prod_1b":   Prod1BConfig,
}

def get_config(name: str):
    if name not in _CONFIGS:
        raise ValueError(f"Unknown config: '{name}'\nTersedia: {', '.join(_CONFIGS)}")
    cfg = _CONFIGS[name]()
    logger.info(f"Loaded config: {name}")
    return cfg
