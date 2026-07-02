"""
train.py — TiniMind Standalone Training Script
=================================================
Pengganti notebook untuk environment non-Colab (Lambda Cloud, RunPod,
server lokal, dll). Format data dan checkpoint TETAP SAMA dengan
TiniMind_Training.ipynb supaya checkpoint bisa saling dipakai.

Asumsi data: chunk_*.bin di --data-dir, masing-masing array uint16
hasil tokenisasi (sama seperti pipeline DataPreprocessor di
nanocore/data/preprocessing.py — format: np.uint16 token ids per file).

Cara pakai:
    python train.py --config prod_500m --data-dir ./data/mc4_indo \\
        --output-dir ./output/checkpoints

    # Resume dari checkpoint:
    python train.py --config prod_500m --data-dir ./data/mc4_indo \\
        --output-dir ./output/checkpoints --resume ./output/checkpoints/step_0003000_loss_3.86.pt

    # Override sebagian config tanpa edit config.py:
    python train.py --config medium_130m --max-steps 5000 --lr 2e-4
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, ModelConfig          # noqa: E402
from model_v2 import TiniMind                        # noqa: E402


# ─── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TiniMind from scratch")
    p.add_argument("--config", default="prod_300m",
                    choices=["tiny_130m", "prod_300m", "prod_500m", "prod_1b"],
                    help="Preset config dari config.py (lihat CATATAN di config.py: "
                         "ini titik awal, bukan otoritas mutlak)")
    p.add_argument("--data-dir", required=True, help="Folder berisi chunk_*.bin")
    p.add_argument("--output-dir", required=True, help="Folder simpan checkpoint")
    p.add_argument("--resume", default=None, help="Path checkpoint untuk resume")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="bf16 disarankan untuk A100/H100. fp16 untuk T4/V100 (tidak support bf16). "
                         "fp32 paling stabil tapi 3-4x lebih lambat.")
    p.add_argument("--val-chunks", type=int, default=2,
                    help="Jumlah chunk terakhir dipakai sebagai validation set")
    # Override opsional dari preset config
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    return p.parse_args()


# ─── Data loading (format identik dengan notebook Cell 3) ──────────────────

def load_chunks(data_dir: str, val_chunks: int):
    bin_files = sorted(glob.glob(f"{data_dir}/chunk_*.bin"))
    if not bin_files:
        raise FileNotFoundError(
            f"Tidak ada chunk_*.bin di {data_dir}. "
            f"Jalankan pipeline preprocessing dulu (lihat nanocore/data/preprocessing.py "
            f"atau Cell 5 di TiniMind_Training.ipynb)."
        )
    total_tok = sum(os.path.getsize(f) // 2 for f in bin_files)  # uint16 = 2 bytes
    print(f"Chunks: {len(bin_files)} | Total: {total_tok/1e9:.2f}B token")

    val_files = bin_files[-val_chunks:]
    train_files = bin_files[:-val_chunks]
    print(f"Train: {len(train_files)} chunks | Val: {len(val_files)} chunks")
    return train_files, val_files


def make_get_batch(seq_len: int, batch_size: int, device: str):
    """Closure get_batch dengan auto-retry kalau file korup/kosong.

    Catatan: di Colab versi sebelumnya ada masalah Drive disconnect
    (OSError 107) yang butuh remount logic. Di Lambda/RunPod/server
    biasa storage-nya lokal/persistent, jadi remount logic TIDAK
    relevan di sini — cukup retry biasa kalau ada file korup.
    """
    def get_batch(files: list[str]):
        for _ in range(5):
            try:
                f = random.choice(files)
                data = np.fromfile(f, dtype=np.uint16).astype(np.int64)
                if len(data) <= seq_len + 1:
                    continue
                ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
                x = torch.stack([torch.from_numpy(data[i:i + seq_len]) for i in ix]).to(device)
                y = torch.stack([torch.from_numpy(data[i + 1:i + seq_len + 1]) for i in ix]).to(device)
                return x, y
            except (OSError, ValueError) as e:
                print(f"Gagal baca chunk ({e}), coba file lain...")
                continue
        raise RuntimeError("Gagal load batch setelah 5x percobaan — cek integritas data.")
    return get_batch


# ─── LR schedule (cosine + warmup, identik dengan notebook) ────────────────

def make_lr_schedule(lr_max: float, lr_min: float, warmup_steps: int, max_steps: int):
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr_max * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
    return get_lr


# ─── Eval ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, get_batch, val_files, dtype, device, num_batches: int = 10):
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(val_files)
        with autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype, enabled=(device == "cuda")):
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler_step, cfg, output_dir, step, loss):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"step_{step:07d}_loss_{loss:.4f}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg,
    }, path)
    print(f"  >> Checkpoint saved: {path}")
    return path


def load_checkpoint_for_resume(path, model, optimizer, device):
    print(f"Resuming dari: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt.get("step", 0)
    print(f"Resume dari step {start_step}")
    return start_step


# ─── Main training loop ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    preset = get_config(args.config)
    model_cfg: ModelConfig = preset.model

    # Apply CLI overrides ke training hyperparams (bukan model architecture —
    # architecture tetap dari preset supaya checkpoint konsisten)
    lr_max = args.lr or preset.learning_rate
    batch_size = args.batch_size or preset.batch_size
    grad_accum = args.grad_accum or preset.gradient_accumulation_steps
    seq_len = args.seq_len or model_cfg.block_size
    log_every = args.log_every or preset.log_every
    save_every = args.save_every or preset.save_every
    eval_every = args.eval_every or preset.eval_every
    max_steps = args.max_steps or (preset.steps_per_epoch * preset.num_epochs
                                    if hasattr(preset, "steps_per_epoch")
                                    else preset.num_epochs * 1000)
    warmup_steps = preset.warmup_steps
    lr_min = 1e-5

    print(f"Config: {args.config} | dtype: {args.dtype} | device: {args.device}")
    print(f"max_steps={max_steps} | lr={lr_max} | batch={batch_size} | "
          f"grad_accum={grad_accum} | seq_len={seq_len}")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    train_dtype = dtype_map[args.dtype]

    if args.dtype == "bf16" and args.device == "cuda" and not torch.cuda.is_bf16_supported():
        print("WARNING: GPU ini tidak support bf16 (kemungkinan T4/V100). "
              "Pakai --dtype fp16 untuk hasil yang stabil.")

    # Data
    train_files, val_files = load_chunks(args.data_dir, args.val_chunks)
    get_batch = make_get_batch(seq_len, batch_size, args.device)

    # Model
    model = TiniMind(model_cfg).to(args.device)
    print(f"Params: {model.num_params()/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = GradScaler(enabled=(args.dtype == "fp16"))
    get_lr = make_lr_schedule(lr_max, lr_min, warmup_steps, max_steps)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint_for_resume(args.resume, model, optimizer, args.device)

    model.train()
    t0 = time.time()

    for step in range(start_step, max_steps):
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(grad_accum):
            x, y = get_batch(train_files)
            with autocast(device_type="cuda" if args.device == "cuda" else "cpu",
                          dtype=train_dtype, enabled=(args.device == "cuda")):
                _, loss, _ = model(x, y)
                loss = loss / grad_accum

            if args.dtype == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated_loss += loss.item()

        if args.dtype == "fp16":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:7d} | train: {accumulated_loss:.4f} | "
                  f"lr: {lr:.2e} | {elapsed:.1f}s")
            t0 = time.time()

        if step % eval_every == 0 and step > 0:
            val_loss = evaluate(model, get_batch, val_files, train_dtype, args.device)
            print(f"  eval @ step {step}: val_loss={val_loss:.4f}")

        if step % save_every == 0 and step > 0:
            val_loss = evaluate(model, get_batch, val_files, train_dtype, args.device)
            save_checkpoint(model, optimizer, step, model_cfg, args.output_dir, step, val_loss)

    # Final checkpoint
    final_val = evaluate(model, get_batch, val_files, train_dtype, args.device)
    save_checkpoint(model, optimizer, max_steps, model_cfg, args.output_dir, max_steps, final_val)
    print("Training selesai.")


if __name__ == "__main__":
    main()
