"""
build_pretraining_data.py — Pipeline data pretraining multi-source (target 100B token)
=============================================================================
Streaming + filter dari 6 dataset sekaligus, filter yang SAMA (import dari
filter.py) diterapkan ke SEMUA sumber, bukan cuma CulturaX. DedupTracker
juga SATU instance dipakai lintas semua sumber -- jadi kalau ada artikel
yang sama nongol di Wikipedia DAN FineWeb-Edu DAN CulturaX (sering terjadi,
web-crawl suka overlap), itu ketangkep juga sebagai duplikat.

SUMBER & ALOKASI (total 100B token):
  1. Wikipedia (ID+EN)          — 5B   — wikimedia/wikipedia
  2. FineWeb-Edu                — 30B  — HuggingFaceFW/fineweb-edu
  3. Cosmopedia v2               — 10B  — HuggingFaceTB/smollm-corpus (config cosmopedia-v2)
  4. OpenWebMath+AlgebraicStack  — 10B  — EleutherAI/proof-pile-2
  5. StarCoder (multi-bahasa)    — 30B  — bigcode/starcoderdata
  6. CulturaX ID                 — 15B  — uonlp/CulturaX (config id)

CATATAN PENTING SOAL JUMLAH FILE (Kaggle Dataset limit 50 top-level):
  chunk_size diperbesar jadi 200M token per chunk (dari 10M), DAN chunk
  diorganisir ke subfolder (SHARD_SIZE chunk per folder) -- limit 50-file
  Kaggle cuma berlaku ke file/folder DI LEVEL ATAS, jadi struktur nested
  di subfolder tidak kena limit itu. Progress juga disave tiap beberapa
  chunk (bukan tiap 1 chunk) biar gak jadi overhead I/O berlebihan.

CATATAN PENTING SOAL URUTAN DATA:
  Chunk ditulis SUMBER PER SUMBER (bukan interleaved di level file), TAPI
  ini tidak masalah untuk training -- get_batch() di train.py/train_deepspeed.py
  sudah random-sample dari SEMUA file chunk yang ada, bukan berurutan. Jadi
  begitu semua sumber selesai diproses, training akan otomatis "mencampur"
  semua sumber secara random di tiap batch. Yang penting: JANGAN mulai
  training sebelum SEMUA sumber selesai diproses, kalau tidak model akan
  overexpose ke sumber yang diproses duluan.

CATATAN JUJUR SOAL WAKTU:
  100B token dari 6 sumber berbeda, dengan filter lengkap (HTML/CSS clean,
  language detect, dedup MinHash, dst) untuk SETIAP dokumen -- ini proses
  yang SANGAT LAMA. Kemungkinan berhari-hari sampai lebih dari seminggu
  tergantung kecepatan koneksi Kaggle dan seberapa banyak dokumen yang
  harus dilewati sebelum lolos filter. Kaggle session timeout tiap 12 jam
  (atau lebih pendek), jadi script ini WAJIB bisa di-resume -- sudah
  didesain begitu (baca progress dari file sebelum lanjut).

Cara pakai:
    python build_pretraining_data.py \\
        --tokenizer /path/to/tokenizer_1b/generalist_bpe_64k.model \\
        --output-dir /path/to/data_1b

    # Resume otomatis -- tinggal jalankan ulang command yang sama, script
    # akan skip sumber yang sudah selesai dan lanjut dari yang belum.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pakai ulang SEMUA fungsi filter dari filter.py -- bukan duplikat logic
from filter import (
    passes_all_filters,
    DedupTracker,
    educational_score,
)


# ─── Definisi sumber & alokasi token ─────────────────────────────────────

SOURCES = [
    {
        "name": "wikipedia_id",
        "hf_path": "wikimedia/wikipedia",
        "hf_config": "20231101.id",
        "text_field": "text",
        "target_tokens": 3_000_000_000,
    },
    {
        "name": "wikipedia_en",
        "hf_path": "wikimedia/wikipedia",
        "hf_config": "20231101.en",
        "text_field": "text",
        "target_tokens": 2_000_000_000,
    },
    {
        "name": "fineweb_edu",
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_config": "default",
        "text_field": "text",
        "target_tokens": 30_000_000_000,
    },
    {
        "name": "cosmopedia_v2",
        "hf_path": "HuggingFaceTB/smollm-corpus",
        "hf_config": "cosmopedia-v2",
        "text_field": "text",
        "target_tokens": 10_000_000_000,
    },
    {
        "name": "openwebmath",
        "hf_path": "EleutherAI/proof-pile-2",
        "hf_config": "open-web-math",
        "text_field": "text",
        "target_tokens": 6_000_000_000,
    },
    {
        "name": "algebraic_stack",
        "hf_path": "EleutherAI/proof-pile-2",
        "hf_config": "algebraic-stack",
        "text_field": "text",
        "target_tokens": 4_000_000_000,
    },
    {
        "name": "starcoder_python",
        "hf_path": "bigcode/starcoderdata",
        "hf_data_dir": "python",
        "text_field": "content",
        "target_tokens": 8_000_000_000,
    },
    {
        "name": "starcoder_javascript",
        "hf_path": "bigcode/starcoderdata",
        "hf_data_dir": "javascript",
        "text_field": "content",
        "target_tokens": 6_000_000_000,
    },
    {
        "name": "starcoder_java",
        "hf_path": "bigcode/starcoderdata",
        "hf_data_dir": "java",
        "text_field": "content",
        "target_tokens": 6_000_000_000,
    },
    {
        "name": "starcoder_go",
        "hf_path": "bigcode/starcoderdata",
        "hf_data_dir": "go",
        "text_field": "content",
        "target_tokens": 5_000_000_000,
    },
    {
        "name": "starcoder_cpp",
        "hf_path": "bigcode/starcoderdata",
        "hf_data_dir": "cpp",
        "text_field": "content",
        "target_tokens": 5_000_000_000,
    },
    {
        "name": "culturax_id",
        "hf_path": "uonlp/CulturaX",
        "hf_config": "id",
        "text_field": "text",
        "target_tokens": 15_000_000_000,
    },
]

TOTAL_TARGET = sum(s["target_tokens"] for s in SOURCES)
assert TOTAL_TARGET == 100_000_000_000, f"Total alokasi {TOTAL_TARGET/1e9}B, seharusnya 100B"

SHARD_SIZE = 40  # berapa chunk per subfolder -- 500 chunk / 40 = ~13 folder di level atas, aman dari limit 50


def chunk_path(output_dir: str, chunk_idx: int) -> str:
    """Chunk disimpan ke subfolder shard_XXX, BUKAN langsung di output_dir --
    ini yang bikin gak kena limit 50 top-level file Kaggle Dataset."""
    shard_num = chunk_idx // SHARD_SIZE
    shard_dir = os.path.join(output_dir, f"shard_{shard_num:03d}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, f"chunk_{chunk_idx:04d}.bin")


# ─── Progress tracking (biar bisa resume kalau sesi Kaggle terputus) ────

def load_progress(output_dir: str) -> dict:
    path = os.path.join(output_dir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            data.setdefault("tokens_per_source", {})  # kompatibel dengan progress.json versi lama
            return data
    return {"completed_sources": [], "chunk_idx": 0, "total_tokens": 0, "tokens_per_source": {}}


def save_progress(output_dir: str, progress: dict):
    path = os.path.join(output_dir, "progress.json")
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ─── Proses satu sumber ───────────────────────────────────────────────────

def process_source(source: dict, sp, dedup_tracker: DedupTracker,
                    output_dir: str, progress: dict, chunk_size: int,
                    hub_repo_id: str = None, buffer_docs: int = 5000):
    from datasets import load_dataset

    name = source["name"]
    print(f"\n{'='*60}")
    print(f"SUMBER: {name} (target: {source['target_tokens']/1e9:.1f}B token)")
    print(f"{'='*60}")

    load_kwargs = {"split": "train", "streaming": True, "trust_remote_code": True}
    if "hf_config" in source:
        load_kwargs["name"] = source["hf_config"]
    if "hf_data_dir" in source:
        load_kwargs["data_dir"] = source["hf_data_dir"]

    try:
        ds = load_dataset(source["hf_path"], **load_kwargs)
    except Exception as e:
        print(f"GAGAL load dataset {name}: {e}")
        print(f"Skip sumber ini, lanjut ke sumber berikutnya.")
        progress["completed_sources"].append(name)  # tandai selesai (gagal) biar gak dicoba ulang terus
        save_progress(output_dir, progress)
        return

    text_field = source["text_field"]

    # RESUME-AWARE: kalau sumber ini sudah punya progress sebagian (dari sesi
    # sebelumnya yang mati di tengah jalan), target dikurangi sisa yang belum
    # tercapai -- BUKAN restart dari 0. Ini mencegah sumber jadi 2x lipat dari
    # target seharusnya tiap kali resume.
    already_done_this_source = progress["tokens_per_source"].get(name, 0)
    remaining_target = max(0, source["target_tokens"] - already_done_this_source)

    if already_done_this_source > 0:
        print(f"  [{name}] Resume: sudah ada {already_done_this_source/1e9:.2f}B token dari sesi sebelumnya, "
              f"sisa target: {remaining_target/1e9:.2f}B")

    if remaining_target <= 0:
        print(f"  [{name}] Sudah tercapai dari sesi sebelumnya, skip.")
        progress["completed_sources"].append(name)
        save_progress(output_dir, progress)
        return

    source_target = remaining_target
    source_tokens = 0

    buf = []
    doc_buffer = []
    stats = {"seen": 0, "kept": 0}
    reject_reasons = {}

    def flush_buffer_sorted():
        nonlocal buf
        doc_buffer.sort(key=lambda t: educational_score(t), reverse=True)
        for text in doc_buffer:
            toks = sp.Encode(text)
            buf.extend(toks)
            progress["total_tokens"] += len(toks)
            nonlocal_source_tokens[0] += len(toks)
            # Catat progress PER-SUMBER (bukan cuma total) -- ini yang
            # dibaca ulang saat resume biar gak restart dari 0 untuk
            # sumber yang lagi diproses pas sesi mati.
            progress["tokens_per_source"][name] = already_done_this_source + nonlocal_source_tokens[0]
            while len(buf) >= chunk_size:
                path = chunk_path(output_dir, progress["chunk_idx"])
                np.array(buf[:chunk_size], dtype=np.uint16).tofile(path)
                print(f"  Saved {os.path.relpath(path, output_dir)} | "
                      f"[{name}] {nonlocal_source_tokens[0]/1e9:.2f}B/{source_target/1e9:.1f}B | "
                      f"total semua sumber: {progress['total_tokens']/1e9:.2f}B/100B")

                # Push chunk INI SAJA ke HF Hub (incremental) -- setiap chunk
                # baru langsung di-backup, gak nunggu 5 chunk kayak save
                # progress lokal, biar risiko kehilangan data lebih kecil
                # (chunk 200M token ~ beberapa menit kerja doang kalau ilang,
                # bukan berjam-jam).
                if hub_repo_id:
                    rel_path = os.path.relpath(path, output_dir)
                    push_file_to_hub(path, hub_repo_id, rel_path)

                buf = buf[chunk_size:]
                progress["chunk_idx"] += 1
                # Save progress + dedup_tracker tiap 5 chunk (bukan tiap 1
                # chunk) -- cukup untuk resume yang aman tanpa jadi overhead
                # I/O berlebihan. dedup_tracker WAJIB ikut disave, kalau
                # tidak, resume akan "lupa" dokumen yang sudah diproses dan
                # menganggap dokumen yang sama sebagai baru lagi (duplikat).
                if progress["chunk_idx"] % 5 == 0:
                    save_progress(output_dir, progress)
                    dedup_tracker.save(os.path.join(output_dir, "dedup_state.pkl"))
                    if hub_repo_id:
                        push_file_to_hub(os.path.join(output_dir, "progress.json"),
                                          hub_repo_id, "progress.json")
                        push_file_to_hub(os.path.join(output_dir, "dedup_state.pkl"),
                                          hub_repo_id, "dedup_state.pkl")
            if nonlocal_source_tokens[0] >= source_target:
                break
        doc_buffer.clear()

    nonlocal_source_tokens = [0]  # pakai list biar bisa dimutasi dalam nested function

    try:
        for doc in ds:
            stats["seen"] += 1
            text = doc.get(text_field, "")
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()

            ok, reason, cleaned_text = passes_all_filters(text, dedup_tracker)
            if not ok:
                key = reason.split("(")[0]
                reject_reasons[key] = reject_reasons.get(key, 0) + 1
                if stats["seen"] % 20000 == 0:
                    rejected = stats["seen"] - stats["kept"]
                    print(f"  [{name}] dilihat: {stats['seen']:,} | lolos: {stats['kept']:,} | "
                          f"dibuang: {rejected/stats['seen']*100:.1f}% | "
                          f"top alasan: {dict(sorted(reject_reasons.items(), key=lambda x: -x[1])[:3])}")
                continue

            stats["kept"] += 1
            doc_buffer.append(cleaned_text)

            if len(doc_buffer) >= buffer_docs:
                flush_buffer_sorted()

            if nonlocal_source_tokens[0] >= source_target:
                print(f"  [{name}] Target sumber ini tercapai!")
                break
    except Exception as e:
        print(f"  [{name}] Error saat streaming: {e}")
        print(f"  Lanjut dengan data yang sudah terkumpul dari sumber ini.")

    if doc_buffer:
        flush_buffer_sorted()
    if buf:
        path = chunk_path(output_dir, progress["chunk_idx"])
        np.array(buf, dtype=np.uint16).tofile(path)
        progress["chunk_idx"] += 1

    print(f"\n[{name}] SELESAI: {nonlocal_source_tokens[0]/1e9:.2f}B token "
          f"({stats['kept']:,}/{stats['seen']:,} dokumen lolos)")

    progress["completed_sources"].append(name)
    save_progress(output_dir, progress)
    dedup_tracker.save(os.path.join(output_dir, "dedup_state.pkl"))


# ─── Main ────────────────────────────────────────────────────────────────

def push_file_to_hub(local_path: str, repo_id: str, path_in_repo: str):
    """Push SATU file ke HF Hub dataset repo -- INCREMENTAL, cuma file ini
    doang yang diupload, bukan seluruh folder. Ini yang bikin backup gak
    makin lama-lama seiring data membesar (beda dari kaggle datasets
    version yang kemungkinan re-upload semua tiap kali)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        return True
    except Exception as e:
        print(f"  PERINGATAN: gagal push {path_in_repo} ke HF Hub: {e}")
        print(f"  (data tetap aman di lokal, cuma backup-nya yang gagal -- coba lagi nanti)")
        return False


def main():
    p = argparse.ArgumentParser(description="Bangun data pretraining 100B token dari 6 sumber")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunk-size", type=int, default=200_000_000,
                    help="Token per chunk. Diperbesar dari default lama (10M) "
                         "supaya total jumlah file tidak meledak untuk target 100B token.")
    p.add_argument("--push-to-hub", default=None,
                    help="Repo ID HuggingFace (misal 'username/tinimind-data-1b') -- "
                         "kalau diisi, tiap chunk baru + progress.json + dedup_state.pkl "
                         "OTOMATIS di-push ke HF Hub begitu selesai ditulis. INCREMENTAL "
                         "(cuma file baru), bukan re-upload semua tiap kali -- jauh lebih "
                         "cepat daripada backup manual yang re-upload seluruh folder.")
    args = p.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"Tokenizer vocab: {sp.GetPieceSize()}")

    os.makedirs(args.output_dir, exist_ok=True)
    progress = load_progress(args.output_dir)

    print(f"\nProgress sebelumnya: {progress['total_tokens']/1e9:.2f}B/100B token, "
          f"{len(progress['completed_sources'])}/{len(SOURCES)} sumber selesai")
    print(f"Sumber selesai: {progress['completed_sources']}\n")

    # DedupTracker SATU instance, dipakai lintas SEMUA sumber -- ini yang
    # bikin duplikat LINTAS sumber (misal artikel sama di Wikipedia & FineWeb)
    # juga ketangkep, bukan cuma duplikat dalam satu sumber.
    # DedupTracker di-LOAD dari disk kalau sudah pernah ada (resume), BUKAN
    # dibuat baru tiap kali script dijalankan -- ini yang mencegah dokumen
    # yang sudah diproses sesi sebelumnya dianggap "baru" lagi setelah resume.
    dedup_path = os.path.join(args.output_dir, "dedup_state.pkl")
    dedup_tracker = DedupTracker.load(dedup_path)
    if os.path.exists(dedup_path):
        print(f"Dedup state di-load: {len(dedup_tracker.exact_hashes):,} hash sudah tercatat")

    for source in SOURCES:
        if source["name"] in progress["completed_sources"]:
            print(f"SKIP {source['name']} (sudah selesai sebelumnya)")
            continue
        process_source(source, sp, dedup_tracker, args.output_dir,
                        progress, args.chunk_size, hub_repo_id=args.push_to_hub)

        if progress["total_tokens"] >= TOTAL_TARGET:
            print("\nTarget 100B token TERCAPAI, berhenti di sini walau masih ada sumber tersisa.")
            break

    print(f"\n{'='*60}")
    print(f"RINGKASAN AKHIR")
    print(f"{'='*60}")
    print(f"Total token: {progress['total_tokens']/1e9:.2f}B / 100B")
    print(f"Sumber selesai: {len(progress['completed_sources'])}/{len(SOURCES)}")
    print(f"Total chunk: {progress['chunk_idx']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
