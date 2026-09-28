from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig          # noqa: E402  (match nama kelas aktual di config.py)
from model_v2 import TiniMind           # noqa: E402  (match nama kelas aktual di model_v2.py)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quantize")


# ─── 1. Load checkpoint & infer config ─────────────────────────────────────────

def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint dari training loop (lihat TiniMind_Training.ipynb).

    Checkpoint training disimpan sebagai dict dengan minimal key 'model'
    (state_dict). Beberapa checkpoint lama mungkin menyimpan state_dict
    langsung tanpa wrapper dict — keduanya ditangani di sini.
    """
    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        # Sudah berupa state_dict langsung
        state_dict = ckpt
    else:
        raise ValueError(
            "Format checkpoint tidak dikenali. Diharapkan dict dengan key "
            "'model', atau state_dict langsung (dict[str, Tensor])."
        )
    return state_dict


def infer_config_from_state_dict(state_dict: dict, override_vocab_size: int = None) -> ModelConfig:
    """Rekonstruksi ModelConfig dari bentuk tensor di checkpoint.

    Ini menghindari hardcode vocab_size/hidden_size yang gampang basi
    kalau config.py di-update tapi checkpoint lama tidak cocok lagi.
    """
    embed_w = state_dict.get("embed.weight")
    if embed_w is None:
        raise ValueError("Key 'embed.weight' tidak ditemukan di checkpoint.")

    vocab_size, hidden_size = embed_w.shape
    if override_vocab_size is not None and override_vocab_size != vocab_size:
        log.warning(
            f"--vocab-size={override_vocab_size} berbeda dari checkpoint "
            f"({vocab_size}). Pakai nilai dari checkpoint untuk konsistensi."
        )

    num_layers = max(
        int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")
    ) + 1

    wq_w = state_dict["blocks.0.attn.wq.weight"]
    wk_w = state_dict["blocks.0.attn.wk.weight"]
    num_heads_dim, _ = wq_w.shape
    kv_dim, _ = wk_w.shape

    gate_w = state_dict["blocks.0.ffn.gate.weight"]
    intermediate_size, _ = gate_w.shape

    # head_dim diasumsikan konsisten; cari num_heads & num_kv_heads dari rasio
    # hidden_size = num_heads * head_dim, dan kita tahu num_heads_dim = num_heads * head_dim
    # sehingga num_heads_dim == hidden_size selalu (Q proj output == hidden_size)
    # num_kv_heads dihitung dari rasio kv_dim / head_dim, head_dim dicari via GCD heuristic.
    # Cara paling aman: cek beberapa head_dim umum (64, 128) yang habis membagi hidden_size.
    head_dim = None
    for candidate in (64, 128, 32, 96, 256):
        if hidden_size % candidate == 0 and kv_dim % candidate == 0:
            head_dim = candidate
            break
    if head_dim is None:
        raise ValueError(
            "Gagal menebak head_dim dari shape checkpoint. "
            "Pakai --num-heads dan --num-kv-heads manual."
        )

    num_heads = hidden_size // head_dim
    num_kv_heads = kv_dim // head_dim

    cfg = ModelConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
    )
    log.info(
        f"Config terdeteksi dari checkpoint: {num_layers}L x {hidden_size}H "
        f"| {num_heads}Q/{num_kv_heads}KV | vocab={vocab_size}"
    )
    return cfg


# ─── 2. Mode: Dynamic INT8 (CPU, torch native) ─────────────────────────────────

def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Dynamic quantization bawaan PyTorch — hanya men-target nn.Linear.

    Tidak menyentuh nn.Embedding (embed/lm_head weight-tied) atau RMSNorm.
    Berjalan di CPU saja (keterbatasan torch.ao.quantization saat ini).
    """
    model.eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    log.info("Dynamic INT8 quantization selesai (CPU, semua nn.Linear).")
    return quantized


# ─── 3. Mode: bitsandbytes 8-bit (GPU) ─────────────────────────────────────────

def quantize_bnb_8bit(model: nn.Module, device: str = "cuda") -> nn.Module:
    """Ganti nn.Linear di attention & FFN dengan bnb.nn.Linear8bitLt.

    lm_head dan embed (weight-tied) SENGAJA dilewati — lihat docstring
    modul di atas.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise SystemExit(
            "bitsandbytes belum terinstall. Jalankan: pip install bitsandbytes\n"
            "Atau pakai --mode dynamic untuk quantization CPU tanpa bitsandbytes."
        )

    skip_substrings = ("lm_head", "embed")

    def _replace(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(
                s in full_name for s in skip_substrings
            ):
                new_layer = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                )
                new_layer.weight = bnb.nn.Int8Params(
                    child.weight.data.clone(), requires_grad=False
                )
                if child.bias is not None:
                    new_layer.bias = nn.Parameter(child.bias.data.clone())
                setattr(module, name, new_layer)
            else:
                _replace(child, full_name)

    model.to(device)
    _replace(model)
    log.info(
        "bitsandbytes 8-bit quantization selesai (GPU). "
        "lm_head & embed tetap FP16/FP32 (weight-tied)."
    )
    return model


# ─── 4. Save & verify ──────────────────────────────────────────────────────────

def save_quantized(model: nn.Module, cfg: ModelConfig, output_path: str, mode: str):
    """Simpan model terkuantisasi beserta config-nya supaya bisa di-load ulang
    tanpa perlu menebak arsitektur lagi (lihat infer_config_from_state_dict)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "quantization_mode": mode,
        },
        output_path,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    log.info(f"Saved: {output_path} ({size_mb:.1f} MB)")


def sanity_check(model: nn.Module, cfg: ModelConfig, device: str = "cpu"):
    """Forward pass sekali untuk memastikan model hasil quantization tidak rusak."""
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16)).to(device)
    with torch.no_grad():
        logits, _, _ = model(x)
    expected_shape = (1, 16, cfg.vocab_size)
    assert logits.shape == expected_shape, (
        f"Shape output salah: {logits.shape}, diharapkan {expected_shape}"
    )
    log.info(f"Sanity check OK — output shape: {tuple(logits.shape)}")


# ─── 5. CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TiniMind post-training quantization")
    parser.add_argument("--checkpoint", required=True, help="Path ke checkpoint .pt")
    parser.add_argument("--output", required=True, help="Path output checkpoint terkuantisasi")
    parser.add_argument(
        "--mode", choices=["dynamic", "bnb8bit"], default="dynamic",
        help="dynamic = INT8 CPU (torch native) | bnb8bit = INT8 GPU (bitsandbytes)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=None,
        help="Override vocab_size (opsional — default ambil dari checkpoint)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    state_dict = load_checkpoint(args.checkpoint)
    cfg = infer_config_from_state_dict(state_dict, override_vocab_size=args.vocab_size)

    model = TiniMind(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"Missing keys saat load: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys saat load: {unexpected}")

    if args.mode == "dynamic":
        model = quantize_dynamic_int8(model)
        sanity_check(model, cfg, device="cpu")
    else:
        model = quantize_bnb_8bit(model, device=args.device)
        sanity_check(model, cfg, device=args.device)

    save_quantized(model, cfg, args.output, args.mode)


if __name__ == "__main__":
    main()
