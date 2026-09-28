"""
train_deepspeed.py — TiniMind Training pakai DeepSpeed ZeRO Stage 2
=============================================================================
Kenapa DeepSpeed dibanding train_fsdp.py (FSDP):
  - DeepSpeed ZeRO lebih matang/battle-tested buat kasus "model gede,
    GPU kecil" — ini justru use-case utama DeepSpeed dirancang.
  - ZeRO Stage 2 (dipakai di sini) shard OPTIMIZER STATE + GRADIENT antar
    GPU, tapi WEIGHT tetap full-copy tiap GPU. Ini titik tengah: lebih
    simpel & stabil dibanding ZeRO Stage 3 (shard weight juga), tapi
    tetap motong memory optimizer state ~8GB -> ~4GB per GPU (2 GPU).

CATATAN JUJUR SEBELUM COBA:
  - DeepSpeed di Kaggle itu AREA YANG BELUM PERNAH DITES buat arsitektur
    TiniMind ini. `pip install deepspeed` KADANG lambat/gagal compile
    custom CUDA ops di environment yang gak selalu punya build tools
    lengkap. Kalau install gagal/lambat banget, ada trik di bagian
    instalasi di bawah buat skip compile ops (DS_BUILD_OPS=0).
  - Kombinasi TiniMindBlock (custom architecture) + DeepSpeed belum
    pernah divalidasi. Kemungkinan ada error yang butuh debug pas
    pertama kali dicoba — itu NORMAL untuk kombinasi baru kayak gini,
    bukan berarti scriptnya pasti salah total.

INSTALASI DI KAGGLE (jalankan di cell notebook, BUKAN di sini):
    # Kalau install biasa kelamaan/gagal compile ops, pakai ini:
    !DS_BUILD_OPS=0 DS_BUILD_FUSED_ADAM=0 pip install deepspeed --break-system-packages -q

CARA JALANIN (di cell notebook, pakai ! di depan):
    !deepspeed --num_gpus=2 train_deepspeed.py \\
        --config prod_1b \\
        --data-dir /kaggle/input/tinimind-data \\
        --output-dir /kaggle/working/checkpoints_1b \\
        --deepspeed_config ds_config_1b.json

    # Resume:
    !deepspeed --num_gpus=2 train_deepspeed.py \\
        --config prod_1b \\
        --data-dir /kaggle/input/tinimind-data \\
        --output-dir /kaggle/working/checkpoints_1b \\
        --deepspeed_config ds_config_1b.json \\
        --resume /kaggle/working/checkpoints_1b/step_XXXXXXX
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys
import time
import json

import numpy as np
import torch
import deepspeed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, ModelConfig    # noqa: E402
from model_v2 import TiniMind                 # noqa: E402


def is_main_process(local_rank: int) -> bool:
    return local_rank in (0, -1)


def parse_args():
    p = argparse.ArgumentParser(description="Train TiniMind pakai DeepSpeed ZeRO-2")
    p.add_argument("--config", default="prod_1b",
                    choices=["tiny_130m", "prod_300m", "prod_500m", "prod_1b"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", default=None,
                    help="Path folder checkpoint DeepSpeed (bukan file .pt tunggal — "
                         "DeepSpeed nyimpen checkpoint sebagai folder)")
    p.add_argument("--val-chunks", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000,
                    help="Naik dari default lama (1000) -- checkpoint gede (~14GB), gak perlu terlalu sering.")
    p.add_argument("--keep-checkpoints", type=int, default=2,
                    help="Berapa checkpoint TERBARU yang disimpan (selain 'final'). "
                         "Checkpoint lebih lama otomatis dihapus. 0 = simpan semua (tidak disarankan).")
    p.add_argument("--eval-every", type=int, default=500)
    # DeepSpeed nambahin argumen sendiri (--deepspeed, --deepspeed_config,
    # --local_rank) otomatis lewat launcher-nya
    parser = deepspeed.add_config_arguments(p)
    return parser.parse_args()


def load_chunks(data_dir: str, val_chunks: int, local_rank: int):
    bin_files = sorted(glob.glob(f"{data_dir}/chunk_*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"Tidak ada chunk_*.bin di {data_dir}.")
    if is_main_process(local_rank):
        total_tok = sum(os.path.getsize(f) // 2 for f in bin_files)
        print(f"Chunks: {len(bin_files)} | Total: {total_tok/1e9:.2f}B token")
    val_files = bin_files[-val_chunks:]
    train_files = bin_files[:-val_chunks]
    return train_files, val_files


def make_get_batch(seq_len: int, batch_size: int, device, local_rank: int):
    rng = random.Random(1234 + max(local_rank, 0))

    def get_batch(files):
        for attempt in range(15):
            try:
                f = rng.choice(files)
                data = np.fromfile(f, dtype=np.uint16).astype(np.int64)
                if len(data) <= seq_len + 1:
                    continue
                ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
                x = torch.stack([torch.from_numpy(data[i:i + seq_len]) for i in ix]).to(device)
                y = torch.stack([torch.from_numpy(data[i + 1:i + seq_len + 1]) for i in ix]).to(device)
                return x, y
            except Exception as e:
                print(f"[rank {local_rank}] Gagal baca chunk ({e}), coba file lain...")
        raise RuntimeError(f"[rank {local_rank}] Gagal load batch setelah 15x percobaan.")
    return get_batch


def log_loss(output_dir, step, train_loss, val_loss=None):
    log_path = os.path.join(output_dir, "loss_log.json")
    entry = {"step": step, "train": round(train_loss, 6)}
    if val_loss is not None:
        entry["val"] = round(val_loss, 6)
    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(log_path, "w") as f:
        json.dump(logs, f)


@torch.no_grad()
def evaluate(model_engine, get_batch, val_files, device, num_batches=10):
    model_engine.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(val_files)
        _, loss, _ = model_engine(x, y)
        losses.append(loss.item())
    model_engine.train()
    return sum(losses) / len(losses)


def rotate_checkpoints(output_dir: str, keep_last: int = 2):
    """Hapus checkpoint DeepSpeed lama, cuma sisain `keep_last` yang terbaru.
    Checkpoint DeepSpeed disimpan sebagai FOLDER (tag), misal step_0001000/,
    bukan file tunggal -- jadi hapusnya per-folder, bukan per-file.
    Checkpoint 'final' (ada _final di nama tag) TIDAK PERNAH dihapus otomatis,
    biar hasil akhir training selalu aman.
    """
    ckpt_dirs = sorted(
        [d for d in glob.glob(os.path.join(output_dir, "step_*")) if os.path.isdir(d)],
        key=os.path.getmtime,
    )
    # Checkpoint final gak pernah dihitung buat dihapus
    ckpt_dirs = [d for d in ckpt_dirs if "_final" not in os.path.basename(d)]

    to_delete = ckpt_dirs[:-keep_last] if keep_last > 0 else ckpt_dirs
    for d in to_delete:
        try:
            shutil.rmtree(d)
            print(f"  >> Checkpoint lama dihapus: {os.path.basename(d)}")
        except Exception as e:
            print(f"  Gagal hapus {d}: {e}")


def main():
    args = parse_args()
    preset = get_config(args.config)
    model_cfg: ModelConfig = preset.model
    seq_len = args.seq_len or model_cfg.block_size
    max_steps = args.max_steps or (preset.steps_per_epoch * preset.num_epochs
                                    if hasattr(preset, "steps_per_epoch")
                                    else preset.num_epochs * 1000)

    deepspeed.init_distributed()
    local_rank = args.local_rank if hasattr(args, "local_rank") else -1

    if is_main_process(local_rank):
        print("=" * 60)
        print(f"DeepSpeed ZeRO-2 training | config={args.config}")
        print(f"max_steps={max_steps} | seq_len={seq_len}")
        print("PERINGATAN: kombinasi TiniMind + DeepSpeed belum pernah dites")
        print("sebelumnya. Error di percobaan pertama itu wajar untuk kombinasi baru.")
        print("=" * 60)

    model = TiniMind(model_cfg)
    if is_main_process(local_rank):
        print(f"Params: {model.num_params()/1e6:.1f}M")

    # DeepSpeed initialize() yang urus: wrap model, bikin optimizer sesuai
    # ds_config (termasuk ZeRO sharding), dan setup gradient accumulation.
    # Parameter lr/optimizer TIDAK diambil dari CLI di sini -- semua
    # dikontrol lewat file ds_config_1b.json (--deepspeed_config).
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )
    device = model_engine.local_rank
    batch_size = model_engine.train_micro_batch_size_per_gpu()

    train_files, val_files = load_chunks(args.data_dir, args.val_chunks, local_rank)
    get_batch = make_get_batch(seq_len, batch_size, model_engine.device, local_rank)

    start_step = 0
    if args.resume:
        _, client_state = model_engine.load_checkpoint(args.resume)
        start_step = client_state.get("step", 0) if client_state else 0
        if is_main_process(local_rank):
            print(f"Resume dari step {start_step}")

    model_engine.train()
    t0 = time.time()

    for step in range(start_step, max_steps):
        x, y = get_batch(train_files)
        _, loss, _ = model_engine(x, y)

        # DeepSpeed urus backward + gradient accumulation + optimizer step
        # sendiri lewat 3 baris ini (beda dari train.py/train_ddp.py yang
        # manual). step() otomatis cuma beneran update weight begitu
        # gradient_accumulation_steps di ds_config udah kepenuhan.
        model_engine.backward(loss)
        model_engine.step()

        if is_main_process(local_rank) and step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:7d} | train: {loss.item():.4f} | {elapsed:.1f}s")
            log_loss(args.output_dir, step, loss.item())
            t0 = time.time()

        if step > start_step and step % args.eval_every == 0:
            val_loss = evaluate(model_engine, get_batch, val_files, model_engine.device)
            if is_main_process(local_rank):
                print(f"  eval @ step {step}: val_loss={val_loss:.4f}")
                log_loss(args.output_dir, step, loss.item(), val_loss)

        if step > start_step and step % args.save_every == 0:
            # DeepSpeed nyimpen checkpoint sebagai FOLDER (bukan 1 file .pt),
            # otomatis nangani sharded state dari tiap GPU.
            model_engine.save_checkpoint(args.output_dir, tag=f"step_{step:07d}",
                                          client_state={"step": step})
            if is_main_process(local_rank):
                print(f"  >> Checkpoint disimpan: step_{step:07d}")
                rotate_checkpoints(args.output_dir, keep_last=args.keep_checkpoints)

    model_engine.save_checkpoint(args.output_dir, tag=f"step_{max_steps:07d}_final",
                                  client_state={"step": max_steps})
    if is_main_process(local_rank):
        print("Training selesai.")


if __name__ == "__main__":
    main()
