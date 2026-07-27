"""
generate.py — TiniMind Inference with Unicode/Byte-Fallback Fix
======================================================================
PROBLEM: The output of generate() produces repeating U+FFFD () characters,
especially on immature models (where val loss is still high).

ROOT CAUSE: The indo_bpe_32k tokenizer was trained with byte_fallback=True
(256 tokens <0x00> through <0xFF>). This is an INTENDED feature — allowing
the tokenizer to represent any character (including those missing from the
training data) via a sequence of individual bytes, preventing information-losing
<unk> tokens.

However, the consequence is that multi-byte UTF-8 characters (accented letters,
symbols, non-ASCII characters) are represented as SEVERAL consecutive byte tokens
that must all be complete before they can be decoded into a single valid character.
An immature model frequently generates individual byte tokens WITHOUT their complete
pairs — resulting in U+FFFD during decoding.

This is NOT a bug in the tokenizer or training process — it is a natural consequence
of a model that isn't mature enough yet. A val loss of 3.3+ is still far too high for
clean generation. LONG-TERM SOLUTION: Continue pretraining until the val loss is lower
(ideally below 2.5) before relying on generate() outputs for anything practical.

SHORT-TERM SOLUTION (fixed in this file): Detect and strip incomplete byte-fallback
tokens BEFORE decoding. This keeps the output clean and free of U+FFFD characters —
even though the tradeoff is that some information/characters might be missing from the
output (missing is better than rendering as corrupt characters).

Usage (CLI):
    python generate.py --checkpoint /path/to/step_0010000_loss_3.3357.pt \
        --tokenizer /path/to/indo_bpe_32k.model \
        --prompt "Apa itu kecerdasan buatan?"

Usage (from notebook / direct import):
    from generate import load_model_and_tokenizer, generate
    model, sp, device = load_model_and_tokenizer(ckpt_path, tok_path)
    text = generate(model, sp, "<penggunna>...</penggunna><asisten>", device)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig       # noqa: E402
from model_v2 import TiniMind        # noqa: E402


# ─── Byte-fallback aware decoding ──────────────────────────────────────────

def _is_byte_fallback_piece(piece: str) -> bool:
    """Cek apakah satu piece dari tokenizer adalah token byte-fallback
    (format '<0xXX>', misal '<0x41>')."""
    return bool(re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece))


def safe_decode(sp, ids: list) -> str:
    """Decode token ids jadi teks, tapi buang RUN byte-fallback yang tidak
    membentuk sekuens UTF-8 valid — mencegah U+FFFD muncul di output.

    Strategi: kelompokkan ids jadi run of byte-fallback vs run token biasa.
    Untuk tiap run byte-fallback, coba decode byte-nya langsung. Kalau
    hasilnya mengandung karakter replacement (artinya sekuens byte tidak
    lengkap/tidak valid UTF-8), buang seluruh run itu dari output alih-alih
    menyisipkan U+FFFD.
    """
    pieces = [sp.id_to_piece(i) for i in ids]

    output_parts = []
    i = 0
    while i < len(pieces):
        if _is_byte_fallback_piece(pieces[i]):
            byte_run = []
            j = i
            while j < len(pieces) and _is_byte_fallback_piece(pieces[j]):
                hex_str = pieces[j][3:5]   # ambil "XX" dari "<0xXX>"
                byte_run.append(int(hex_str, 16))
                j += 1

            try:
                decoded_bytes = bytes(byte_run).decode("utf-8")
                if "\ufffd" not in decoded_bytes:
                    output_parts.append(decoded_bytes)
                # else: sekuens byte tidak valid UTF-8 lengkap -> dibuang
            except UnicodeDecodeError:
                pass  # dibuang, sama seperti di atas

            i = j
        else:
            normal_run = []
            j = i
            while j < len(pieces) and not _is_byte_fallback_piece(pieces[j]):
                normal_run.append(ids[j])
                j += 1
            output_parts.append(sp.decode(normal_run))
            i = j

    return "".join(output_parts)


def clean_repetition(text: str, max_repeat: int = 3) -> str:
    """Bonus fix: model yang belum matang cenderung mengulang kata yang
    sama berkali-kali ('udahudahudah...'). Ini gejala lain dari model
    belum matang (bukan masalah unicode) — dipangkas supaya output lebih
    terbaca saat testing manual. TIDAK menyelesaikan akar masalah — model
    tetap perlu pretrain lebih lanjut.
    """
    words = text.split()
    if not words:
        return text

    result = []
    repeat_count = 1
    for k in range(1, len(words)):
        if words[k] == words[k - 1]:
            repeat_count += 1
        else:
            repeat_count = 1
        if repeat_count <= max_repeat:
            result.append(words[k - 1])
    result.append(words[-1])
    return " ".join(result)


# ─── Model loading (dipakai CLI maupun import notebook) ────────────────────

def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str,
                              device: str = None):
    """Load model + tokenizer sekali, return siap dipakai berulang kali
    untuk generate() — hindari reload tiap panggilan kalau dipakai
    interaktif dari notebook.
    """
    import sentencepiece as spm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model") or ckpt.get("model_state")
    if state_dict is None:
        raise KeyError(
            f"Tidak bisa menemukan state dict di checkpoint. "
            f"Keys yang ada: {list(ckpt.keys())}"
        )

    cfg = ckpt.get("config") or ModelConfig(
        num_layers=24, hidden_size=1024, num_heads=16,
        num_kv_heads=4, vocab_size=32000, max_seq_len=2048
    )

    model = TiniMind(cfg).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    step = ckpt.get("step", "?")
    print(f"Params: {model.num_params()/1e6:.1f}M | Checkpoint step: {step}")

    return model, sp, device


# ─── Generation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, sp, prompt: str, device: str,
             max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 50) -> str:
    """Generate teks dari prompt. Prompt HARUS sudah diformat dengan
    special token kalau mau mensimulasikan percakapan, misal:
        "<penggunna>Apa itu AI?</penggunna><asisten>"
    """
    ids = torch.tensor([sp.Encode(prompt)], dtype=torch.long).to(device)
    past_kvs = None

    generated_ids = ids[0].tolist()

    for _ in range(max_new_tokens):
        inp = ids if past_kvs is None else ids[:, -1:]
        offset = 0 if past_kvs is None else ids.shape[1] - 1
        logits, _, past_kvs = model(inp, use_kv_cache=True, past_kvs=past_kvs, offset=offset)
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
        generated_ids.append(next_id.item())

    # FIX UTAMA: pakai safe_decode, bukan sp.Decode() langsung, supaya
    # byte-fallback yang tidak lengkap tidak menghasilkan U+FFFD.
    raw_text = safe_decode(sp, generated_ids)
    return clean_repetition(raw_text)


def generate_with_maturity_warning(model, sp, prompt_raw: str, device: str,
                                    step: int = None, **kwargs) -> str:
    """Wrapper generate() yang otomatis format prompt dengan special token
    dan kasih warning kalau checkpoint masih terlalu awal untuk hasil
    yang koheren.
    """
    prompt_formatted = f"<penggunna>{prompt_raw}</penggunna><asisten>"
    result = generate(model, sp, prompt_formatted, device, **kwargs)

    if step is not None and isinstance(step, int) and step < 15000:
        print()
        print("=" * 60)
        print(f"CATATAN: checkpoint ini baru step {step}. Output masih akan")
        print("terasa acak/tidak koheren karena model belum matang (belum")
        print("SFT juga). Fix di file ini menghilangkan karakter U+FFFD,")
        print("tapi TIDAK membuat model 'pintar' lebih cepat — itu perlu")
        print("lanjut pretrain sampai val loss lebih rendah dulu.")
        print("=" * 60)

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TiniMind inference dengan fix unicode")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model, sp, device = load_model_and_tokenizer(
        args.checkpoint, args.tokenizer, device=args.device
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = ckpt.get("step", None)

    result = generate_with_maturity_warning(
        model, sp, args.prompt, device, step=step,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k
    )

    print()
    print(result)


if __name__ == "__main__":
    main()
