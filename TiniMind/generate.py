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
             max_new_tokens: int = 200, temperature: float = 0.9, top_k: int = 50,
             repetition_penalty: float = 1.6, no_repeat_ngram_size: int = 2) -> str:
    """Generate teks dari prompt. Prompt HARUS sudah diformat dengan
    special token kalau mau mensimulasikan percakapan, misal:
        "<penggunna>Apa itu AI?</penggunna><asisten>"

    MVP sampling fix v2 (diperkuat dari versi sebelumnya):
      - repetition_penalty sekarang FREQUENCY-SCALED: token yang udah
        keluar 5x dihukum lebih keras daripada yang baru keluar 1x
        (bukan cuma "pernah muncul = dihukum flat" seperti versi awal).
        Ini penting buat model yang masih dini dan attractor-nya kuat
        (contoh kasus: prompt "nasi goreng" nyasar berulang-ulang ke
        vocab "download/bermain/Membuat").
      - no_repeat_ngram_size diturunin ke 2 (dari 3) — lebih agresif,
        blokir walau cuma 2-kata-berurutan yang berulang persis.
      - temperature default naik ke 0.9 (dari 0.8) — dikit lebih
        random, ngebantu keluar dari loop yang sama.
      - repetition_penalty=1.0 DAN no_repeat_ngram_size=0 kalau mau
        balikin ke behavior lama/vanilla buat perbandingan.

    Model masih pretrain-only (checkpoint dini) — fix ini bikin output
    jauh lebih variatif/nggak nge-loop, TAPI TIDAK membuat model lebih
    "pintar" atau lebih koheren secara makna kalimat panjang. Itu tetap
    butuh lanjut pretrain.
    """
    ids = torch.tensor([sp.Encode(prompt)], dtype=torch.long).to(device)
    past_kvs = None

    generated_ids = ids[0].tolist()
    prompt_len = len(generated_ids)

    for _ in range(max_new_tokens):
        inp = ids if past_kvs is None else ids[:, -1:]
        offset = 0 if past_kvs is None else ids.shape[1] - 1
        logits, _, past_kvs = model(inp, use_kv_cache=True, past_kvs=past_kvs, offset=offset)
        logits = logits[:, -1, :] / temperature

        # Repetition penalty FREQUENCY-SCALED: makin sering satu token
        # udah keluar, makin keras skornya dihukum (bukan cuma flat
        # "pernah muncul = dihukum sekali"). Ini jauh lebih efektif
        # ngelawan attractor kuat di model yang masih dini.
        if repetition_penalty != 1.0:
            from collections import Counter
            counts = Counter(generated_ids)
            for tok_id, cnt in counts.items():
                effective_penalty = repetition_penalty ** cnt  # makin sering, makin kuat hukumannya
                if logits[0, tok_id] > 0:
                    logits[0, tok_id] /= effective_penalty
                else:
                    logits[0, tok_id] *= effective_penalty

        # No-repeat n-gram: blokir token yang bakal bikin n-gram terakhir
        # persis sama dengan n-gram yang sudah pernah muncul.
        if no_repeat_ngram_size > 0 and len(generated_ids) - prompt_len + 1 >= no_repeat_ngram_size:
            n = no_repeat_ngram_size
            prefix = tuple(generated_ids[-(n - 1):]) if n > 1 else ()
            banned = set()
            for i in range(len(generated_ids) - n + 1):
                if tuple(generated_ids[i:i + n - 1]) == prefix:
                    banned.add(generated_ids[i + n - 1])
            for tok_id in banned:
                logits[0, tok_id] = -float("inf")

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
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.6,
                         help="1.0 = nonaktif. Frequency-scaled: makin sering token keluar, makin keras dihukum.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2,
                         help="0 = nonaktif. Blokir n-gram yang persis berulang.")
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
        temperature=args.temperature, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    print()
    print(result)


if __name__ == "__main__":
    main()
