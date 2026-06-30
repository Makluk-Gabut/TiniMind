"""
train_tokenizer_indo.py — TiniMind Custom Indonesian Tokenizer
==============================================================
Train BPE tokenizer 32k vocab dari Indonesian text (Wikipedia + mC4).

⚠️  PENTING: Ganti tokenizer = ganti vocab_size di ModelConfig!
    - cl100k_base: vocab 100352, embedding = 100352 × hidden_size
    - Indo BPE 32k: vocab 32000, embedding = 32000 × hidden_size
    - Hemat ~2/3 parameter untuk embedding table
    - Kata Indonesia: "pembelajaran" cl100k→4-5 token, IndoBPE→1-2 token
    - Checkpoint lama TIDAK kompatibel → mulai pretrain dari awal

Cara pakai di Colab:
    !pip install sentencepiece datasets -q
    !python train_tokenizer_indo.py

Output:
    /content/drive/MyDrive/TiniMind_Prototype/tokenizer/indo_bpe_32k.model
    /content/drive/MyDrive/TiniMind_Prototype/tokenizer/indo_bpe_32k.vocab
"""

import os
import io
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tokenizer")

# ─── Config ────────────────────────────────────────────────────────────────────

BASE       = "/content/drive/MyDrive/TiniMind_Prototype"
TOK_DIR    = f"{BASE}/tokenizer"
MODEL_PATH = f"{TOK_DIR}/indo_bpe_32k.model"
VOCAB_PATH = f"{TOK_DIR}/indo_bpe_32k.vocab"
TEXT_PATH  = "/tmp/indo_corpus.txt"

VOCAB_SIZE   = 32000
MAX_CHARS    = 200_000_000   # ~200MB text, cukup untuk 32k vocab
WIKI_ROWS    = 200_000
MC4_ROWS     = 300_000

SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<penggunna>", "</penggunna>",
    "<asisten>", "</asisten>",
]

# ─── 1. Kumpulkan text Indonesia ───────────────────────────────────────────────

def collect_text():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    os.makedirs(TOK_DIR, exist_ok=True)
    total_chars = 0

    log.info(f"Menulis corpus ke {TEXT_PATH}")
    with open(TEXT_PATH, "w", encoding="utf-8") as f:

        # Wikipedia Indonesia
        log.info("Streaming Wikipedia Indonesia...")
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.id",
                              split="train", streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                text = row.get("text", "").strip()
                if not text or len(text) < 50:
                    continue
                f.write(text + "\n")
                total_chars += len(text)
                count += 1
                if count % 10000 == 0:
                    log.info(f"  Wiki: {count:,} artikel ({total_chars/1e6:.1f}M chars)")
                if count >= WIKI_ROWS or total_chars >= MAX_CHARS * 0.4:
                    break
            log.info(f"OK Wikipedia: {count:,} artikel")
        except Exception as e:
            log.warning(f"FAIL Wikipedia: {e}")

        # mC4 Indonesian
        log.info("Streaming mC4 Indonesian...")
        try:
            ds = load_dataset("mc4", "id", split="train", streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                text = row.get("text", "").strip()
                if not text or len(text) < 100:
                    continue
                f.write(text[:1000] + "\n")
                total_chars += min(len(text), 1000)
                count += 1
                if count % 20000 == 0:
                    log.info(f"  mC4: {count:,} docs ({total_chars/1e6:.1f}M chars)")
                if count >= MC4_ROWS or total_chars >= MAX_CHARS:
                    break
            log.info(f"OK mC4: {count:,} docs")
        except Exception as e:
            log.warning(f"FAIL mC4: {e}")

    size_mb = os.path.getsize(TEXT_PATH) / 1e6
    log.info(f"Corpus total: {size_mb:.1f} MB")
    return TEXT_PATH


# ─── 2. Train tokenizer ────────────────────────────────────────────────────────

def train_tokenizer(text_path: str):
    try:
        import sentencepiece as spm
    except ImportError:
        raise SystemExit("Install sentencepiece: pip install sentencepiece")

    log.info(f"Training BPE tokenizer vocab={VOCAB_SIZE}...")
    log.info("Estimasi ~5-15 menit di Colab...")

    user_defined = ",".join(SPECIAL_TOKENS)

    spm.SentencePieceTrainer.train(
        input            = text_path,
        model_prefix     = MODEL_PATH.replace(".model", ""),
        vocab_size       = VOCAB_SIZE,
        model_type       = "bpe",
        character_coverage = 0.9995,
        pad_id           = 0,
        unk_id           = 1,
        bos_id           = 2,
        eos_id           = 3,
        pad_piece        = "<pad>",
        unk_piece        = "<unk>",
        bos_piece        = "<bos>",
        eos_piece        = "<eos>",
        user_defined_symbols = user_defined,
        shuffle_input_sentence = True,
        num_threads      = 4,
        input_sentence_size = 5_000_000,
        max_sentence_length  = 4192,
        byte_fallback    = True,
    )

    log.info(f"Model saved: {MODEL_PATH}")
    log.info(f"Vocab saved: {VOCAB_PATH}")


# ─── 3. Test tokenizer ─────────────────────────────────────────────────────────

def test_tokenizer(model_path: str):
    try:
        import sentencepiece as spm
    except ImportError:
        return

    log.info("\n─── Test Tokenizer ───")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    test_cases = [
        "Halo, siapa kamu?",
        "Kecerdasan buatan adalah teknologi masa depan.",
        "Proses pembelajaran mendalam membutuhkan banyak data.",
        "Indonesia adalah negara kepulauan yang indah.",
    ]

    for text in test_cases:
        tokens = sp.encode(text, out_type=str)
        ids    = sp.encode(text, out_type=int)
        print(f"\n  Input  : {text}")
        print(f"  Tokens : {tokens}")
        print(f"  Count  : {len(ids)} token")

    # Bandingkan dengan tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        log.info("\n─── Perbandingan vs cl100k_base ──")
        test_words = ["pembelajaran", "kecerdasan", "pemerintahan", "pengembangan"]
        print(f"\n{'Kata':<20} {'cl100k':>8} {'IndoBPE':>8} {'Hemat':>8}")
        print("-" * 50)
        for word in test_words:
            cl100k_n = len(enc.encode(word))
            indo_n   = len(sp.encode(word))
            hemat    = f"{(1 - indo_n/cl100k_n)*100:.0f}%"
            print(f"  {word:<18} {cl100k_n:>8} {indo_n:>8} {hemat:>8}")
    except ImportError:
        pass


# ─── 4. Wrapper class ─────────────────────────────────────────────────────────

class IndoTokenizer:
    """Drop-in replacement untuk tiktoken di training/inference.

    Cara pakai:
        from train_tokenizer_indo import IndoTokenizer
        tok = IndoTokenizer(model_path)
        ids = tok.encode("Halo dunia!")
        text = tok.decode(ids)
        tok.eot_token   # EOS token ID
        tok.n_vocab     # 32000
    """

    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError:
            raise SystemExit("Install sentencepiece: pip install sentencepiece")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.n_vocab   = self.sp.get_piece_size()
        self.eot_token = self.sp.piece_to_id("<eos>")
        self.pad_token = self.sp.piece_to_id("<pad>")
        self.bos_token = self.sp.piece_to_id("<bos>")

    def encode(self, text: str) -> list:
        return self.sp.encode(text, out_type=int)

    def encode_ordinary(self, text: str) -> list:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list) -> str:
        return self.sp.decode(ids)


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    if args.test_only:
        if os.path.exists(MODEL_PATH):
            test_tokenizer(MODEL_PATH)
        else:
            print(f"Model tidak ada: {MODEL_PATH}")
    else:
        if not args.skip_download or not os.path.exists(TEXT_PATH):
            text_path = collect_text()
        else:
            text_path = TEXT_PATH

        train_tokenizer(text_path)
        test_tokenizer(MODEL_PATH)

        print("\n" + "="*60)
        print("Tokenizer training selesai!")
        print(f"  Model : {MODEL_PATH}")
        print("\nLangkah selanjutnya:")
        print("  1. Update config.py: vocab_size = 32000")
        print("  2. Re-run prepare_mc4_indo.py (retokenize data)")
        print("  3. Training dari scratch dengan model_v2.py")
        print("="*60)
