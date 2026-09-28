import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tokenizer")

# Auto-detect environment: Kaggle pakai /kaggle/working, selain itu asumsikan
# Colab dengan Google Drive di-mount. Ini biar TIDAK PERLU patch manual tiap
# kali pindah environment -- kode ini sendiri yang mendeteksi.
if os.path.exists("/kaggle"):
    BASE = "/kaggle/working/TiniMind"
else:
    BASE = "/content/drive/MyDrive/TiniMind"
TOK_DIR    = f"{BASE}/tokenizer_1b"
MODEL_PATH = f"{TOK_DIR}/generalist_bpe_64k.model"
VOCAB_PATH = f"{TOK_DIR}/generalist_bpe_64k.vocab"
TEXT_PATH  = "/tmp/generalist_corpus.txt"

VOCAB_SIZE = 64_000
MAX_CHARS  = 400_000_000

WIKI_ID_ROWS  = 150_000
WIKI_EN_ROWS  = 100_000
CULTURAX_ROWS = 200_000
CODE_ROWS     = 150_000

SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<pengguna>", "</pengguna>",
    "<asisten>", "</asisten>",
]

# <pad>/<unk>/<bos>/<eos> SUDAH otomatis didefinisikan lewat pad_id/unk_id/
# bos_id/eos_id + pad_piece/unk_piece/dst di bawah -- kalau ikut dimasukkan
# lagi ke user_defined_symbols, SentencePiece error "must not be defined
# with --control_symbols and --user_defined_symbols". Cuma token CHAT yang
# perlu didaftarkan manual ke user_defined_symbols.
CHAT_TOKENS = ["<pengguna>", "</pengguna>", "<asisten>", "</asisten>"]


def collect_text():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    os.makedirs(TOK_DIR, exist_ok=True)
    total_chars = 0

    log.info(f"Menulis corpus ke {TEXT_PATH}")
    with open(TEXT_PATH, "w", encoding="utf-8") as f:

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
                    log.info(f"  Wiki-ID: {count:,} artikel ({total_chars/1e6:.1f}M chars)")
                if count >= WIKI_ID_ROWS or total_chars >= MAX_CHARS * 0.25:
                    break
            log.info(f"OK Wikipedia-ID: {count:,} artikel")
        except Exception as e:
            log.warning(f"FAIL Wikipedia-ID: {e}")

        log.info("Streaming Wikipedia English...")
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                               split="train", streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                text = row.get("text", "").strip()
                if not text or len(text) < 50:
                    continue
                f.write(text[:2000] + "\n")
                total_chars += min(len(text), 2000)
                count += 1
                if count % 10000 == 0:
                    log.info(f"  Wiki-EN: {count:,} artikel ({total_chars/1e6:.1f}M chars)")
                if count >= WIKI_EN_ROWS or total_chars >= MAX_CHARS * 0.45:
                    break
            log.info(f"OK Wikipedia-EN: {count:,} artikel")
        except Exception as e:
            log.warning(f"FAIL Wikipedia-EN: {e}")

        log.info("Streaming CulturaX Indonesia...")
        try:
            ds = load_dataset("uonlp/CulturaX", "id", split="train",
                               streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                text = row.get("text", "").strip()
                if not text or len(text) < 100:
                    continue
                f.write(text[:1000] + "\n")
                total_chars += min(len(text), 1000)
                count += 1
                if count % 20000 == 0:
                    log.info(f"  CulturaX-ID: {count:,} docs ({total_chars/1e6:.1f}M chars)")
                if count >= CULTURAX_ROWS or total_chars >= MAX_CHARS * 0.75:
                    break
            log.info(f"OK CulturaX-ID: {count:,} docs")
        except Exception as e:
            log.warning(f"FAIL CulturaX-ID: {e}")

        log.info("Streaming kode (Python/JS/Java)...")
        try:
            ds = load_dataset("bigcode/the-stack-smol", split="train",
                               streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                text = row.get("content", "").strip()
                if not text or len(text) < 50:
                    continue
                f.write(text[:1500] + "\n")
                total_chars += min(len(text), 1500)
                count += 1
                if count % 10000 == 0:
                    log.info(f"  Kode: {count:,} file ({total_chars/1e6:.1f}M chars)")
                if count >= CODE_ROWS or total_chars >= MAX_CHARS:
                    break
            log.info(f"OK Kode: {count:,} file")
        except Exception as e:
            log.warning(f"FAIL kode: {e}")

    size_mb = os.path.getsize(TEXT_PATH) / 1e6
    log.info(f"Corpus total: {size_mb:.1f} MB")
    return TEXT_PATH


def train_tokenizer(text_path: str):
    try:
        import sentencepiece as spm
    except ImportError:
        raise SystemExit("Install sentencepiece: pip install sentencepiece")

    log.info(f"Training BPE tokenizer vocab={VOCAB_SIZE}...")

    user_defined = ",".join(CHAT_TOKENS)

    spm.SentencePieceTrainer.train(
        input                  = text_path,
        model_prefix           = MODEL_PATH.replace(".model", ""),
        vocab_size             = VOCAB_SIZE,
        model_type             = "bpe",
        character_coverage     = 0.9990,
        pad_id                 = 0,
        unk_id                 = 1,
        bos_id                 = 2,
        eos_id                 = 3,
        pad_piece              = "<pad>",
        unk_piece              = "<unk>",
        bos_piece              = "<bos>",
        eos_piece              = "<eos>",
        user_defined_symbols   = user_defined,
        shuffle_input_sentence = True,
        num_threads            = 4,
        input_sentence_size    = 8_000_000,
        max_sentence_length    = 4192,
        byte_fallback          = True,
    )

    log.info(f"Model saved: {MODEL_PATH}")
    log.info(f"Vocab saved: {VOCAB_PATH}")


def test_tokenizer(model_path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    test_strings = [
        "Cara membuat nasi goreng adalah dengan menyiapkan bahan-bahan.",
        "The quick brown fox jumps over the lazy dog.",
        "def hello_world():\n    print('Hello, world!')\n    return True",
        "<pengguna>Apa itu kecerdasan buatan?</pengguna><asisten>",
    ]

    log.info(f"Vocab size aktual: {sp.GetPieceSize()}")
    for s in test_strings:
        ids = sp.Encode(s)
        pieces = sp.EncodeAsPieces(s)
        log.info(f"\n  Input : {s[:60]}...")
        log.info(f"  Tokens: {len(ids)} | Pieces: {pieces[:12]}...")


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
