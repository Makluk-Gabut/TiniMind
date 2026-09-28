"""
filter.py — Filter & pembersih komprehensif untuk data pretraining teks
=============================================================================
Berlaku untuk sumber teks web-crawl apapun (CulturaX, mC4, OSCAR, dll),
bukan spesifik satu dataset. Beberapa lapis:

  LAPIS 0 — Pembersihan HTML/CSS: buang sisa tag HTML, atribut style/class,
            dan blok CSS yang nyangkut di teks (umum terjadi di web-crawl
            yang scraping-nya kurang bersih). Ini MEMBERSIHKAN teks,
            bukan buang dokumen -- kecuali densitas markup-nya sangat
            tinggi (dokumen praktis raw HTML, bukan prosa), baru dibuang.
  LAPIS 1 — Bahasa: cuma terima dokumen Indonesia atau Inggris. Bahasa lain
            (Arab, Cina, dll yang kadang nyelip di CulturaX) dibuang total.
  LAPIS 2 — Panjang: dokumen < 500 kata ATAU > 10.000 kata dibuang. Terlalu
            pendek biasanya potongan navigasi web/spam; terlalu panjang
            biasanya scrape error (nge-gabung banyak halaman jadi satu).
  LAPIS 3 — Spam simbol: dokumen dengan rasio karakter non-alfabet terlalu
            tinggi (simbol/emoji bertubi-tubi, format rusak) dibuang.
  LAPIS 4 — Kata kotor/kasar: dokumen dengan densitas kata kasar (ID+EN)
            di atas threshold dibuang. INI FILTER PROTEKTIF (membuang),
            bukan daftar buat tujuan lain.
  LAPIS 5 — Marketing/tutorial spam: keyword filter dari analisis
            sebelumnya (download/daftar/member/dll), sama seperti versi lalu.
  LAPIS 6 (skor, bukan buang) — Preferensi topik edukatif/sains: dokumen
            yang kata-katanya condong ke sains/edukasi diprioritaskan
            duluan dalam urutan pengambilan (bukan filter keras, cuma
            bikin data "bagus" lebih sering kepilih duluan).

CATATAN JUJUR:
  - Ini filter heuristik (aturan + keyword), BUKAN classifier ML. Bakal
    ada false positive (dokumen bagus kebuang) dan false negative
    (dokumen jelek lolos). Tapi jauh lebih baik daripada tanpa filter.
  - Pembersihan HTML/CSS pakai regex, bukan HTML parser beneran (BeautifulSoup
    dll) -- lebih cepat & tanpa dependency tambahan, tapi bisa saja miss
    kasus markup yang sangat tidak standar.
  - langdetect (deteksi bahasa) itu PROBABILISTIK dan kadang salah untuk
    teks pendek/campuran -- bukan 100% akurat.

Cara pakai:
    pip install langdetect
    python filter.py --target-tokens 3000000000 \\
        --tokenizer /path/to/tokenizer.model --output-dir /path/to/data
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import unicodedata

import numpy as np
import hashlib

# ─── LAPIS 10: Blacklist judi/dewasa/spam (BEDA dari marketing keywords) ─
# Marketing keywords (download/daftar/dll) itu bahasa tutorial umum yang
# BISA muncul di artikel legit (tutorial teknologi asli). Blacklist ini
# BEDA -- kata-kata di sini nyaris SELALU muncul di konten judi online/
# situs dewasa, jarang banget di konten legit. Threshold lebih ketat
# (cukup 1 match, bukan density) karena keyword-nya lebih spesifik/jelas.

GAMBLING_ADULT_KEYWORDS = [
    "slot gacor", "maxwin", "togel", "zeus slot", "scatter hitam",
    "situs slot", "rtp slot", "bandar togel", "judi online",
    "bokep", "porn", "pinjol ilegal", "obat pembesar",
]


def has_gambling_adult_content(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in GAMBLING_ADULT_KEYWORDS)


# ─── LAPIS 11: URL/domain blacklist + link-count spam ────────────────────

SUSPICIOUS_TLDS = [".xyz", ".top", ".click", ".gacor", ".live", ".bet"]

URL_PATTERN = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+")


def has_suspicious_domain(text: str) -> bool:
    urls = URL_PATTERN.findall(text.lower())
    return any(any(tld in url for tld in SUSPICIOUS_TLDS) for url in urls)


MAX_LINKS = 10


def has_excessive_links(text: str) -> bool:
    return len(URL_PATTERN.findall(text)) > MAX_LINKS


# ─── LAPIS 12: Kata diulang berlebih + tidak ada tanda baca ─────────────
# BEDA dari has_repeated_symbol_spam (yang cek SIMBOL berulang) -- ini
# cek KATA yang sama mendominasi seluruh dokumen (contoh: "mantap mantap
# mantap..." atau review spam yang copas 1 kata ratusan kali).

MAX_SINGLE_WORD_RATIO = 0.40  # kalau 1 kata yang sama = >40% dari total kata, buang


def has_dominant_repeated_word(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 10:
        return False
    from collections import Counter
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return (most_common_count / len(words)) > MAX_SINGLE_WORD_RATIO


def has_no_punctuation(text: str) -> bool:
    """Dokumen tanpa titik/koma/tanda tanya sama sekali biasanya bukan
    prosa normal -- kemungkinan besar list/tabel yang salah ke-scrape,
    atau spam kata kunci berturut-turut."""
    return not re.search(r"[.,?!]", text)


# ─── LAPIS 7: Deduplikasi (exact + near-duplicate) ───────────────────────
# CulturaX sering ada artikel yang SAMA PERSIS atau HAMPIR SAMA muncul di
# banyak situs beda (copas berita, artikel SEO yang di-syndicate ulang).
# Model yang lihat teks yang sama berkali-kali cenderung overfit ke situ,
# bukan belajar generalisasi -- efeknya mirip kasus "6.5 epoch data yang
# sama" yang sudah pernah dibahas, tapi ini levelnya per-dokumen individual.
#
# Dua jenis deteksi:
#   (a) EXACT duplicate: hash SHA256 dari teks yang sudah dinormalisasi
#       (lowercase, whitespace dirapikan) -- nangkep kalau ada yang
#       benar-benar identik.
#   (b) NEAR duplicate: MinHash sederhana pakai shingle (potongan n-kata
#       berurutan) -- nangkep dokumen yang MIRIP tapi tidak identik persis
#       (misal beda dikit di judul/tanggal tapi isi sama).
#
# State (set hash yang sudah pernah dilihat) disimpan di OBJECT DedupTracker
# di bawah -- harus dipakai SATU INSTANCE yang sama sepanjang proses
# filtering (bukan dibuat ulang tiap dokumen), makanya beda dari fungsi
# lain yang stateless.

def normalize_for_hash(text: str) -> str:
    """Normalisasi teks sebelum di-hash: lowercase, whitespace dirapikan,
    biar duplikat yang beda kapitalisasi/spasi tetap terdeteksi sama."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_hash(text: str) -> str:
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_shingles(text: str, n: int = 5) -> set:
    """Pecah teks jadi shingle (n-kata berurutan) buat near-dup detection."""
    words = normalize_for_hash(text).split()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def minhash_signature(shingles: set, num_hashes: int = 32) -> tuple:
    """MinHash signature sederhana -- representasi ringkas dari set shingle,
    dipakai buat estimasi kemiripan (Jaccard similarity) tanpa nyimpen
    seluruh shingle mentah (yang bisa sangat besar buat dokumen panjang)."""
    if not shingles:
        return tuple()
    signature = []
    for seed in range(num_hashes):
        min_hash = min(
            hashlib.md5(f"{seed}_{s}".encode("utf-8")).digest()
            for s in shingles
        )
        signature.append(min_hash)
    return tuple(signature)


class DedupTracker:
    """State buat deduplikasi -- HARUS satu instance yang sama dipakai
    sepanjang proses filtering (dipassing ke passes_all_filters), bukan
    dibuat ulang tiap dokumen."""

    def __init__(self, near_dup_threshold: float = 0.85, max_signatures: int = 500_000):
        self.exact_hashes = set()
        self.minhash_signatures = []  # list of (signature, ) -- dibatasi max_signatures biar RAM gak meledak
        self.near_dup_threshold = near_dup_threshold
        self.max_signatures = max_signatures

    def is_duplicate(self, text: str) -> tuple[bool, str]:
        # Cek exact duplicate dulu (murah, cepat)
        h = exact_hash(text)
        if h in self.exact_hashes:
            return True, "exact_duplicate"
        self.exact_hashes.add(h)

        # Cek near-duplicate (lebih mahal, cuma jalan kalau bukan exact dup)
        shingles = get_shingles(text)
        sig = minhash_signature(shingles)
        if not sig:
            return False, ""

        for existing_sig in self.minhash_signatures:
            similarity = sum(a == b for a, b in zip(sig, existing_sig)) / len(sig)
            if similarity >= self.near_dup_threshold:
                return True, "near_duplicate"

        # Simpan signature buat perbandingan dokumen berikutnya, dengan cap
        # biar RAM gak meledak untuk dataset yang sangat besar
        if len(self.minhash_signatures) < self.max_signatures:
            self.minhash_signatures.append(sig)

        return False, ""

    def save(self, path: str):
        """Simpan state ke disk -- WAJIB dipanggil berkala kalau proses
        akan berjalan lintas banyak sesi (resume). Tanpa ini, restart
        script = dedup_tracker lupa semua dokumen yang sudah diproses,
        dan dokumen di awal stream akan diproses ULANG sebagai 'baru'
        (duplikat beneran, bukan cuma boros waktu)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "exact_hashes": self.exact_hashes,
                "minhash_signatures": self.minhash_signatures,
            }, f)

    @classmethod
    def load(cls, path: str, near_dup_threshold: float = 0.85, max_signatures: int = 500_000):
        """Load state dari disk kalau ada, kalau tidak ada return instance baru
        (kasus pertama kali jalan, belum pernah save)."""
        import pickle
        import os
        tracker = cls(near_dup_threshold=near_dup_threshold, max_signatures=max_signatures)
        if os.path.exists(path):
            with open(path, "rb") as f:
                state = pickle.load(f)
            tracker.exact_hashes = state["exact_hashes"]
            tracker.minhash_signatures = state["minhash_signatures"]
        return tracker


# ─── LAPIS 8: Boilerplate non-konten (navigasi, footer, cookie notice) ───
# Ini BUKAN HTML tag (jadi lolos LAPIS 0), tapi tetap bukan prosa yang mau
# dipelajari model -- pola teks yang umum muncul di elemen non-konten web.

BOILERPLATE_PATTERNS = [
    re.compile(r"\b(beranda|home)\s*[\|/>]\s*(tentang|about)", re.IGNORECASE),
    re.compile(r"\bbaca\s+juga\s*:", re.IGNORECASE),
    re.compile(r"\b(cookie|kebijakan\s+privasi)\s+(kami|ini)\s+(menggunakan|digunakan)", re.IGNORECASE),
    re.compile(r"\b(share|bagikan)\s*:\s*(facebook|twitter|whatsapp)", re.IGNORECASE),
    re.compile(r"\ball\s+rights?\s+reserved\b", re.IGNORECASE),
    re.compile(r"\bhak\s+cipta\s+dilindungi\b", re.IGNORECASE),
    re.compile(r"\bsubscribe\s+(to\s+)?(our\s+)?newsletter\b", re.IGNORECASE),
    re.compile(r"\bberlangganan\s+newsletter\b", re.IGNORECASE),
    re.compile(r"^\s*(menu|navigasi)\s*[:|]", re.IGNORECASE | re.MULTILINE),
]

MAX_BOILERPLATE_HITS_PER_1000_WORDS = 3.0


def boilerplate_density(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    hits = sum(len(pattern.findall(text)) for pattern in BOILERPLATE_PATTERNS)
    return hits / len(words) * 1000


def is_boilerplate_heavy(text: str) -> bool:
    return boilerplate_density(text) > MAX_BOILERPLATE_HITS_PER_1000_WORDS


# ─── LAPIS 9: PII (nomor telepon, email) — di-REDACT, bukan buang dokumen ─
# Berbeda dari lapis lain: PII bukan alasan buang seluruh dokumen (artikel
# yang isinya bagus tapi kebetulan ada 1 nomor telepon di komentar gak perlu
# dibuang semua), cukup DIGANTI placeholder biar model gak belajar/menghafal
# data pribadi orang.

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_ID_PATTERN = re.compile(
    r"\b(?:\+62|62|0)8[1-9][0-9]{6,10}\b"  # nomor HP Indonesia (08xx / +628xx)
)


def redact_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_ID_PATTERN.sub("[TELEPON]", text)
    return text


# ─── LAPIS 0: Pembersihan sisa HTML/CSS ──────────────────────────────────

# Tag HTML lengkap dengan isinya yang HARUS dibuang total (bukan cuma
# tag-nya, tapi isinya juga -- ini biasanya bukan konten yang mau dibaca)
HTML_BLOCK_TAGS_TO_STRIP = re.compile(
    r"<(script|style|noscript|iframe|svg)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

# Tag HTML biasa -- buang tag-nya doang, isinya (teks di dalam) dipertahankan
HTML_TAG = re.compile(r"<[^>]+>")

# Entity HTML umum -> karakter aslinya
HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&apos;": "'", "&hellip;": "...",
    "&mdash;": "—", "&ndash;": "-", "&rsquo;": "'", "&lsquo;": "'",
    "&rdquo;": '"', "&ldquo;": '"',
}

# Blok CSS yang nyangkut sebagai teks polos (bukan di dalam tag <style>,
# tapi keluar sebagai teks biasa -- kasus umum di scraping yang parsing-nya
# kurang bersih), contoh: ".classname { color: red; font-size: 12px; }"
CSS_RULE_BLOCK = re.compile(
    r"[.#]?[\w-]+(?:\s*[,>+~]\s*[.#]?[\w-]+)*\s*\{[^{}]*\}",
)

# Inline style/class attribute yang nyangkut sebagai teks polos (bukan di
# dalam tag, tag-nya udah ke-strip tapi atributnya kadang nyangkut duluan)
INLINE_STYLE_REMNANT = re.compile(
    r'\b(?:style|class)\s*=\s*["\'][^"\']*["\']',
    re.IGNORECASE,
)

# URL mentah yang nyangkut (biasanya dari href="..." yang tag-nya udah
# ke-strip tapi URL-nya kepisah jadi teks lepas)
BARE_URL_FRAGMENT = re.compile(r"https?://\S+")


def clean_html_css(text: str) -> str:
    """Bersihkan sisa HTML/CSS dari teks. Return teks yang sudah dibersihkan
    (bukan buang dokumen -- itu urusan is_html_css_spam di bawah)."""
    # 1. Buang blok script/style/dsb BESERTA isinya
    text = HTML_BLOCK_TAGS_TO_STRIP.sub(" ", text)
    # 2. Buang tag HTML biasa, teks di dalamnya dipertahankan
    text = HTML_TAG.sub(" ", text)
    # 3. Convert entity HTML umum
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)
    # 4. Buang blok CSS yang nyangkut sebagai teks polos
    text = CSS_RULE_BLOCK.sub(" ", text)
    # 5. Buang sisa atribut style/class yang nyangkut
    text = INLINE_STYLE_REMNANT.sub(" ", text)
    # 6. Rapikan whitespace berlebih akibat semua penghapusan di atas
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_css_density(text: str) -> float:
    """Densitas markup HTML/CSS yang MASIH ADA (buat deteksi dokumen yang
    praktis raw HTML, bukan prosa -- dihitung SEBELUM dibersihkan)."""
    if not text:
        return 0.0
    markup_chars = 0
    markup_chars += sum(len(m.group()) for m in HTML_TAG.finditer(text))
    markup_chars += sum(len(m.group()) for m in CSS_RULE_BLOCK.finditer(text))
    return markup_chars / len(text)


MAX_HTML_CSS_DENSITY = 0.15  # kalau >15% dokumen (sebelum dibersihkan) adalah markup, buang total


def is_html_css_spam(text: str) -> bool:
    """True kalau dokumen praktis raw HTML/CSS dump, bukan prosa yang
    kebetulan ada sedikit sisa markup (yang itu cukup dibersihkan, gak perlu
    dibuang -- lihat clean_html_css)."""
    return html_css_density(text) > MAX_HTML_CSS_DENSITY


# ─── LAPIS 1: Deteksi bahasa ─────────────────────────────────────────────

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0  # hasil konsisten, bukan random tiap run
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

ALLOWED_LANGUAGES = {"id", "en"}


def is_allowed_language(text: str) -> bool:
    """Deteksi bahasa dengan beberapa perbaikan biar gak gampang false positive:

    1. Ambil BEBERAPA sample (bukan cuma 1 titik tengah) dari bagian beda
       dokumen -- teks panjang sering campur bahasa (kutipan asing,
       istilah teknis), 1 sample doang gampang salah nangkep bagian yang
       kebetulan bukan representatif.
    2. Sample lebih besar (2000 karakter, dari 1000) -- langdetect makin
       akurat makin banyak teks yang dikasih.
    3. Voting: dokumen LOLOS kalau MAYORITAS sample terdeteksi ID/EN,
       bukan harus SEMUA sample. Ini nurunin false-positive buat dokumen
       yang emang ID/EN asli tapi kebetulan ada 1 sample kena salah deteksi.
    4. Dokumen pendek (<800 karakter) di-skip filter bahasa ini sama
       sekali -- langdetect terlalu gak akurat di teks pendek, MENDING
       diloloskan (biar lapis lain yang nyaring) daripada salah buang.
    """
    if not LANGDETECT_AVAILABLE:
        return True

    if len(text) < 800:
        return True  # terlalu pendek buat dipercaya langdetect, skip filter ini

    sample_size = 2000
    positions = [0.15, 0.5, 0.85]  # ambil dari awal, tengah, akhir dokumen
    votes_allowed = 0
    votes_total = 0

    for frac in positions:
        start = int(len(text) * frac)
        sample = text[start:start + sample_size]
        if len(sample) < 200:  # sample kebetulan kepotong pendek, skip
            continue
        try:
            lang = detect(sample)
            votes_total += 1
            if lang in ALLOWED_LANGUAGES:
                votes_allowed += 1
        except LangDetectException:
            continue  # gagal deteksi 1 sample bukan berarti gagal semua, lanjut ke sample lain

    if votes_total == 0:
        return True  # semua sample gagal dideteksi -- lebih aman loloskan, biar lapis lain yang nyaring

    # LOLOS kalau MAYORITAS (bukan seluruh) sample terdeteksi ID/EN
    return votes_allowed / votes_total >= 0.5


# ─── LAPIS 2: Panjang dokumen ────────────────────────────────────────────

MIN_WORDS = 500
MAX_WORDS = 10_000


def word_count_ok(text: str) -> bool:
    n = len(text.split())
    return MIN_WORDS <= n <= MAX_WORDS


# ─── LAPIS 3: Spam simbol / karakter rusak ───────────────────────────────

MAX_NON_ALPHA_RATIO = 0.30

# Simbol yang WAJAR muncul di teks teknis/matematika/sains -- jangan
# dihitung sebagai "spam" walau banyak. Sebelumnya cuma whitelist tanda
# baca dasar, sekarang ditambah simbol matematika, mata uang, dan yang
# umum di artikel sains/teknis.
NORMAL_SYMBOLS = set(".,!?;:'\"-()[]{}/%&@#")
TECHNICAL_SYMBOLS = set("+=<>*^~°µ±÷×√∑∏∫πθ$€£¥_|\\")
ALL_ALLOWED_SYMBOLS = NORMAL_SYMBOLS | TECHNICAL_SYMBOLS


def symbol_spam_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_alpha = 0
    for ch in text:
        if ch.isalpha() or ch.isspace() or ch.isdigit():
            continue
        if ch in ALL_ALLOWED_SYMBOLS:
            continue
        non_alpha += 1
    return non_alpha / len(text)


def has_repeated_symbol_spam(text: str, min_run: int = 5) -> bool:
    """Deteksi POLA spam yang lebih spesifik: karakter SIMBOL YANG SAMA
    diulang berturut-turut min_run+ kali (contoh: '!!!!!', '=====',
    '#####'). Ini nangkep spam beneran tanpa nge-flag teks teknis yang
    kebetulan padat simbol tapi bervariasi (rumus matematika, misalnya).
    """
    run_char = None
    run_len = 0
    for ch in text:
        if ch.isalnum() or ch.isspace():
            run_char = None
            run_len = 0
            continue
        if ch == run_char:
            run_len += 1
            if run_len >= min_run:
                return True
        else:
            run_char = ch
            run_len = 1
    return False


def is_symbol_spam(text: str) -> bool:
    # Dokumen dianggap symbol-spam kalau SALAH SATU dari dua kondisi ini:
    # (a) rasio simbol non-wajar terlalu tinggi, ATAU
    # (b) ada pola karakter simbol yang sama berulang 5x+ berturut-turut
    #     (indikasi spam/format rusak yang lebih pasti daripada rasio doang)
    return symbol_spam_ratio(text) > MAX_NON_ALPHA_RATIO or has_repeated_symbol_spam(text)


# ─── LAPIS 4: Kata kasar/kotor (filter PROTEKTIF, untuk MEMBUANG) ────────
# Daftar ringkas kata kasar umum ID+EN, dipakai HANYA untuk deteksi &
# pembuangan konten, bukan untuk tujuan lain. Threshold density rendah
# supaya satu-dua kata gak langsung buang seluruh dokumen panjang.

PROFANITY_ID = [
    "anjing", "bangsat", "kontol", "memek", "ngentot", "goblok", "tolol",
    "bajingan", "asu", "babi", "jancok", "kampret", "brengsek", "sialan",
]
PROFANITY_EN = [
    "fuck", "shit", "bitch", "asshole", "cunt", "bastard", "dick",
]
PROFANITY_ALL = PROFANITY_ID + PROFANITY_EN
MAX_PROFANITY_DENSITY = 3.0  # per 1000 kata


def profanity_density(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    text_lower = text.lower()
    hits = sum(text_lower.count(w) for w in PROFANITY_ALL)
    return hits / len(words) * 1000


def is_profanity_spam(text: str) -> bool:
    return profanity_density(text) > MAX_PROFANITY_DENSITY


# ─── LAPIS 5: Marketing/tutorial spam (dari analisis sebelumnya) ────────

MARKETING_KEYWORDS = [
    "download", "daftar", "member", "beli", "jual", "online", "gratis",
    "terbaik", "promo", "diskon", "login", "akun", "aplikasi", "situs",
    "website", "bonus", "hadiah", "klik", "kunjungi", "pendaftaran",
    "mendaftar", "transfer", "pembayaran", "deposit", "withdraw",
]
MAX_MARKETING_DENSITY = 15.0


def marketing_density(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    text_lower = text.lower()
    hits = sum(text_lower.count(kw) for kw in MARKETING_KEYWORDS)
    return hits / len(words) * 1000


def is_marketing_spam(text: str) -> bool:
    return marketing_density(text) > MAX_MARKETING_DENSITY


# ─── LAPIS 6: Skor preferensi topik edukatif/sains (BUKAN filter keras) ─
# Dipakai buat SORTING, bukan buang -- dokumen dengan skor tinggi
# diproses/diambil duluan sebelum dokumen skor rendah, sampai target
# token tercapai. Kalau target belum tercapai dan dokumen skor-rendah-
# tapi-lolos-filter-keras masih dibutuhkan, tetap dipakai (lebih baik
# dari pada buang token budget).

EDUCATIONAL_KEYWORDS = [
    "penelitian", "ilmiah", "eksperimen", "teori", "rumus", "universitas",
    "profesor", "jurnal", "sains", "matematika", "fisika", "biologi",
    "kimia", "sejarah", "analisis", "metode", "hipotesis", "data",
    "research", "scientific", "theory", "university", "journal",
    "science", "mathematics", "physics", "biology", "chemistry", "history",
    "analysis", "method", "hypothesis",
]


def educational_score(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    text_lower = text.lower()
    hits = sum(text_lower.count(kw) for kw in EDUCATIONAL_KEYWORDS)
    return hits / len(words) * 1000


# ─── Fungsi filter gabungan ───────────────────────────────────────────────

def passes_all_filters(text: str, dedup_tracker: "DedupTracker") -> tuple[bool, str, str]:
    """Return (lolos: bool, alasan_buang: str, teks_bersih: str).
    teks_bersih adalah versi text SETELAH dibersihkan HTML/CSS + PII
    di-redact -- ini yang harus dipakai buat tokenisasi, bukan text asli.

    dedup_tracker: instance DedupTracker yang SAMA harus dipassing terus
    sepanjang proses filtering satu dataset (nyimpen state dokumen yang
    udah pernah dilihat). Jangan buat instance baru tiap dokumen.
    """
    if is_html_css_spam(text):
        return False, f"html_css_spam({html_css_density(text):.2f})", text

    text = clean_html_css(text)

    if not word_count_ok(text):
        n = len(text.split())
        return False, f"panjang_invalid({n}_kata)", text

    if is_symbol_spam(text):
        return False, f"symbol_spam({symbol_spam_ratio(text):.2f})", text

    if is_boilerplate_heavy(text):
        return False, f"boilerplate({boilerplate_density(text):.1f}/1000kata)", text

    if has_gambling_adult_content(text):
        return False, "gambling_adult_content", text

    if has_suspicious_domain(text):
        return False, "suspicious_domain", text

    if has_excessive_links(text):
        return False, f"excessive_links(>{MAX_LINKS})", text

    if has_dominant_repeated_word(text):
        return False, "dominant_repeated_word", text

    if has_no_punctuation(text):
        return False, "no_punctuation", text

    if is_profanity_spam(text):
        return False, f"profanity({profanity_density(text):.1f}/1000kata)", text

    if is_marketing_spam(text):
        return False, f"marketing_spam({marketing_density(text):.1f}/1000kata)", text

    if not is_allowed_language(text):
        return False, "bahasa_tidak_diizinkan", text

    is_dup, dup_reason = dedup_tracker.is_duplicate(text)
    if is_dup:
        return False, dup_reason, text

    return True, "", text


# ─── Main pipeline ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Filter komprehensif data CulturaX untuk pretraining")
    p.add_argument("--target-tokens", type=int, default=3_000_000_000)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunk-size", type=int, default=10_000_000)
    p.add_argument("--buffer-docs", type=int, default=5000,
                    help="Berapa dokumen ditampung dulu sebelum di-sort by "
                         "educational_score dan diproses. Buffer lebih besar "
                         "= sorting lebih efektif tapi lebih banyak RAM.")
    args = p.parse_args()

    if not LANGDETECT_AVAILABLE:
        print("PERINGATAN: langdetect tidak terinstall (pip install langdetect).")
        print("Filter bahasa (LAPIS 1) akan di-skip, dokumen bahasa lain "
              "kemungkinan tetap lolos.\n")

    import sentencepiece as spm
    from datasets import load_dataset

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"Tokenizer vocab: {sp.GetPieceSize()}")

    os.makedirs(args.output_dir, exist_ok=True)
    existing = sorted(glob.glob(f"{args.output_dir}/chunk_*.bin"))
    tokens_done = sum(os.path.getsize(f) // 2 for f in existing)
    chunk_idx = len(existing)
    print(f"Token sudah ada: {tokens_done/1e9:.2f}B / {args.target_tokens/1e9:.1f}B")

    ds = load_dataset("uonlp/CulturaX", "id", split="train", streaming=True, trust_remote_code=True)

    buf = []
    doc_buffer = []  # buffer dokumen yang LOLOS filter, buat di-sort dulu

    stats = {"seen": 0, "kept": 0}
    dedup_tracker = DedupTracker()  # satu instance ini dipakai sepanjang proses, JANGAN dibuat ulang per-dokumen
    reject_reasons = {}

    def flush_buffer_sorted():
        """Sort buffer dokumen yang lolos filter berdasarkan educational_score
        (tinggi ke rendah), lalu tokenize & tulis ke chunk secara berurutan."""
        nonlocal buf, chunk_idx, tokens_done
        doc_buffer.sort(key=lambda t: educational_score(t), reverse=True)
        for text in doc_buffer:
            toks = sp.Encode(text)
            buf.extend(toks)
            tokens_done += len(toks)
            while len(buf) >= args.chunk_size:
                path = f"{args.output_dir}/chunk_{chunk_idx:04d}.bin"
                np.array(buf[:args.chunk_size], dtype=np.uint16).tofile(path)
                pct_kept = stats["kept"] / stats["seen"] * 100 if stats["seen"] else 0
                print(f"Saved chunk_{chunk_idx:04d}.bin | total: {tokens_done/1e9:.2f}B | "
                      f"lolos filter: {pct_kept:.1f}%")
                buf = buf[args.chunk_size:]
                chunk_idx += 1
            if tokens_done >= args.target_tokens:
                break
        doc_buffer.clear()

    for doc in ds:
        stats["seen"] += 1
        text = doc.get("text", "").strip()

        if not text:
            continue

        ok, reason, cleaned_text = passes_all_filters(text, dedup_tracker)
        if not ok:
            reject_reasons[reason.split("(")[0]] = reject_reasons.get(reason.split("(")[0], 0) + 1
            if stats["seen"] % 20000 == 0:
                total_rejected = stats["seen"] - stats["kept"]
                print(f"  [progress] dilihat: {stats['seen']:,} | lolos: {stats['kept']:,} | "
                      f"dibuang: {total_rejected:,} ({total_rejected/stats['seen']*100:.1f}%)")
                print(f"  [alasan]  {dict(sorted(reject_reasons.items(), key=lambda x: -x[1])[:5])}")
            continue

        stats["kept"] += 1
        doc_buffer.append(cleaned_text)

        if len(doc_buffer) >= args.buffer_docs:
            flush_buffer_sorted()

        if tokens_done >= args.target_tokens:
            print("Target tercapai!")
            break

    # Flush sisa buffer kalau masih ada dan target belum tercapai
    if doc_buffer and tokens_done < args.target_tokens:
        flush_buffer_sorted()

    if buf:
        path = f"{args.output_dir}/chunk_{chunk_idx:04d}.bin"
        np.array(buf, dtype=np.uint16).tofile(path)
        print(f"Final: chunk_{chunk_idx:04d}.bin")

    print(f"\n{'='*60}")
    print(f"RINGKASAN AKHIR")
    print(f"{'='*60}")
    print(f"Dokumen dilihat : {stats['seen']:,}")
    print(f"Dokumen dipakai : {stats['kept']:,} ({stats['kept']/stats['seen']*100:.1f}%)")
    print(f"Total token     : {tokens_done/1e9:.2f}B")
    print(f"\nAlasan pembuangan (top 10):")
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason:20s}: {count:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
