from __future__ import annotations
import os
import torch
import torch.nn as nn
from typing import Optional


# ─── LoRA Linear ────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wrapper nn.Linear + low-rank adapter A dan B."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank   = rank
        self.scale  = alpha / rank

        in_f  = linear.in_features
        out_f = linear.out_features

        # A: init gaussian kecil, B: init nol
        # Saat awal, delta W = B@A = 0 → output sama dengan base model
        self.lora_A = nn.Parameter(torch.randn(rank, in_f)  * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

        # Freeze base weight
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base   = self.linear(x)
        delta  = (x @ self.lora_A.T) @ self.lora_B.T
        return base + delta * self.scale

    def merge(self) -> nn.Linear:
        """Gabungkan adapter ke weight base → kembalikan nn.Linear biasa."""
        merged = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
        )
        merged.weight.data = (
            self.linear.weight.data
            + (self.lora_B @ self.lora_A) * self.scale
        )
        if self.linear.bias is not None:
            merged.bias.data = self.linear.bias.data.clone()
        return merged


# ─── Utilities ──────────────────────────────────────────────────

# Layer mana saja yang di-LoRA (sesuai kebiasaan LLaMA fine-tuning)
_DEFAULT_TARGET = {"wq", "wv"}   # minimal: Q dan V
_ALL_ATTN       = {"wq", "wk", "wv", "wo"}
_ALL_LINEAR     = {"wq", "wk", "wv", "wo", "gate", "up", "down"}


def apply_lora(
    model: nn.Module,
    rank:    int   = 16,
    alpha:   float = 32.0,
    targets: set   = _DEFAULT_TARGET,
) -> nn.Module:
    """Wrap layer yang dipilih dengan LoRALinear, freeze sisanya.

    Args:
        model:   TiniMind instance
        rank:    Rank adapter (lebih besar = lebih ekspresif, lebih berat)
        alpha:   Skala (alpha/rank menentukan learning rate efektif adapter)
        targets: Set nama atribut yang di-LoRA.
                 {'wq','wv'}          → minimal (paling hemat, cukup untuk banyak kasus)
                 {'wq','wk','wv','wo'} → semua attention
                 {'wq','wk','wv','wo','gate','up','down'} → attention + FFN

    Returns:
        Model yang siap dilatih (hanya LoRA params punya requires_grad=True)
    """
    # Freeze semua dulu
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for name, module in model.named_modules():
        attr = name.split(".")[-1]
        if isinstance(module, nn.Linear) and attr in targets:
            parent_name = ".".join(name.split(".")[:-1])
            parent      = model.get_submodule(parent_name)
            setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha))
            replaced += 1

    total  = sum(p.numel() for p in model.parameters())
    lora_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA applied: {replaced} layers | "
          f"trainable={lora_n/1e6:.2f}M / total={total/1e6:.1f}M "
          f"({100*lora_n/total:.2f}%)")
    return model


def lora_params(model: nn.Module):
    """Generator parameter LoRA saja (untuk optimizer)."""
    return (p for p in model.parameters() if p.requires_grad)


def merge_and_remove_lora(model: nn.Module) -> nn.Module:
    """Merge semua LoRALinear ke base weight → model siap inference/export."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name = ".".join(name.split(".")[:-1])
            attr        = name.split(".")[-1]
            parent      = model.get_submodule(parent_name)
            setattr(parent, attr, module.merge())

    # Unfreeze semua setelah merge
    for p in model.parameters():
        p.requires_grad_(True)

    print("LoRA merged into base weights.")
    return model


# ─── Save / Load adapter ────────────────────────────────────────

def save_lora(model: nn.Module, path: str, meta: dict = None):
    """Simpan hanya LoRA adapter (A, B, scale) — jauh lebih kecil dari full checkpoint."""
    adapters = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapters[name] = {
                "lora_A": module.lora_A.data,
                "lora_B": module.lora_B.data,
                "rank":   module.rank,
                "scale":  module.scale,
            }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"adapters": adapters, "meta": meta or {}}, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"LoRA adapter saved: {path} ({size_mb:.1f} MB, {len(adapters)} layers)")


def load_lora(model: nn.Module, path: str):
    """Load LoRA adapter ke model yang sudah di-apply_lora() dulu."""
    data     = torch.load(path, map_location="cpu", weights_only=False)
    adapters = data["adapters"]
    loaded   = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in adapters:
            ad = adapters[name]
            module.lora_A.data = ad["lora_A"]
            module.lora_B.data = ad["lora_B"]
            loaded += 1
    print(f"LoRA adapter loaded: {loaded}/{len(adapters)} layers from {path}")
    return model
