# TiniMind

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-AGPL-green)
![Status](https://img.shields.io/badge/Status-Beta-orange)

> A lightweight decoder-only Transformer built from scratch in PyTorch with modern architectures and an Indonesian tokenizer.

# ⚠️ **Beta Phase**

### TiniMind is currently under active development. The architecture, APIs, training pipeline, and project structure may change as development progresses.
---

# Why TiniMind?

TiniMind is an experimental Small Language Model (SLM) focused on Indonesian.

The primary goal of this project is to build a lightweight, educational, and efficient Transformer implementation that demonstrates modern language model architectures while remaining easy to understand and modify.

Rather than relying heavily on existing frameworks, TiniMind is implemented from scratch to provide a deeper understanding of how modern decoder-only language models work internally.

A custom Indonesian BPE tokenizer is included to improve tokenization efficiency for Indonesian text compared to general-purpose tokenizers.

---

# Features

- ✅ Decoder-only Transformer built entirely in PyTorch
- ✅ Grouped Query Attention (GQA)
- ✅ Rotary Positional Embeddings (RoPE)
- ✅ RMSNorm
- ✅ Flash Attention using PyTorch `scaled_dot_product_attention`
- ✅ SwiGLU Feed Forward Network
- ✅ Multi-size config (~125M, ~303M, ~490M, ~1.01B)
- ✅ Custom Indonesian SentencePiece Tokenizer (32k Vocabulary)
- ✅ Standalone Training Script
- ✅ Resume Training from Checkpoints
- ✅ Mixed Precision Training (BF16 / FP16 / FP32)
- ✅ Cosine Learning Rate Scheduler with Warmup
- ✅ Modular project structure for experimentation

---

# Current Status

|        Component         |    Status    |
|--------------------------|--------------|
| Indonesian Tokenizer     | Stable       |
| Decoder-only Transformer | Stable       |
| Configuration System     | Stable       |
| Training Script          | Stable       |
| Training Notebook        | Working      |
| Evaluation Benchmark     | Planned      |
| Pretrained Model         | In Progress  |
| Stable Release           | In Progress  |

---

# Roadmap

- [x] Indonesian BPE Tokenizer
- [x] Decoder-only Transformer
- [x] Flash Attention
- [x] Grouped Query Attention
- [x] RMSNorm
- [x] Quantization
- [x] Complete Training Pipeline
- [x] Standalone Training Script
- [x] Resume Training
- [ ] Evaluation Benchmark
- [ ] Hugging Face Integration
- [ ] Pretrained Model Release
- [ ] GGUF Export
- [x] LoRA Support (experimental)

---

# Repository Structure

|           File            |                        Description                        |
|---------------------------|-----------------------------------------------------------|
| `model_v2.py`             | Main Transformer architecture implementation              |
| `config.py`               | Model configuration and hyperparameters                   |
| `train.py`                | Standalone training script with checkpoint resume support |
| `train_tokenizer_indo.py` | Indonesian tokenizer training script                      |
| `TiniMind_Training.ipynb` | Interactive notebook for experimentation and tutorials    |
| `requirements.txt`        | Python dependencies                                       |
| `LICENSE`                 | AGPL License                                              |

---

# Installation

```bash
git clone https://github.com/Makluk-Gabut/TiniMind.git
cd TiniMind
pip install -r requirements.txt
```

---

# Training

### 1. Train the tokenizer

```bash
python train_tokenizer_indo.py
```

### 2. Configure the model

Edit the model configuration in `config.py` and choose a config based on your hardware:

|    Config    | Params |    Hardware    |
|--------------|--------|----------------|
| `tiny_130m`  | ~125M  | T4 / any GPU   |
| `prod_300m`  | ~303M  | T4 16GB        |
| `prod_500m`  | ~490M  | A100 40GB+     |
| `prod_1b`    | ~1.01B | A100 80GB+     |

### 3. Start training

```bash
python train.py \
    --config prod_300m \
    --data-dir ./data \
    --output-dir ./output/pretrain \
    --dtype fp16
```

> **Note:** Use `--dtype fp16` for T4/V100. Use `--dtype bf16` for A100/H100.

### 4. Resume training from checkpoint

```bash
python train.py \
    --config prod_300m \
    --data-dir ./data \
    --output-dir ./output/pretrain \
    --dtype fp16 \
    --resume ./output/pretrain/step_xxxxxxx_loss_x.xxxx.pt
```

### 5. Interactive experimentation

For interactive experimentation and development, use the Jupyter notebook:

```
TiniMind_Training.ipynb
```

### 6. Generate text (CLI)

```bash
python generate.py \
    --checkpoint /path/to/step_0010000_loss_3.3357.pt \
    --tokenizer /path/to/indo_bpe_32k.model \
    --prompt "Apa itu kecerdasan buatan?"
```

### 7. Generate text (programmatic)

```python
from generate import load_model_and_tokenizer, generate

model, sp, device = load_model_and_tokenizer(ckpt_path, tok_path)
text = generate(model, sp, "<penggunna>...</penggunna><asisten>", device)
```

### 8. Quantize model

```bash
python quantize.py \
    --checkpoint /path/to/step_0010000_loss_2.5.pt \
    --output /path/to/tinimind_int8.pt \
    --mode dynamic
```

Or with bitsandbytes (8-bit):

```bash
python quantize.py \
    --checkpoint /path/to/step_0010000_loss_2.5.pt \
    --output /path/to/tinimind_int8.pt \
    --mode bnb8bit \
    --device cuda
```

---

# Requirements

- Python 3.10+
- PyTorch 2.x
- CUDA-compatible GPU (recommended)
- SentencePiece
- bitsandbytes

---

# Philosophy

TiniMind is not intended to compete with large commercial language models.

Instead, this project exists as a personal research and learning project focused on understanding how modern Transformer architectures actually work.

Every component in this repository was implemented because I wanted to learn how it works internally—not just how to use it.

The codebase is intentionally kept modular and readable so anyone interested in language models can explore, modify, and learn from it.

---

# License

This project is licensed under the **AGPL License**.

See the [LICENSE](LICENSE) file for more information.

---

# Dev's Note

Some comments are written in Indonesian. This notebook serves both as project documentation and my personal development notebook, so you may occasionally find mixed-language comments. The code, variable names, and APIs remain in English.

And if you've made it this far, thank you for taking the time to explore TiniMind.

This project has been my personal playground for learning and experimenting with modern Transformer architectures over the past eight months.

I'm building this project entirely on my own while still attending school. Most of the development happened after classes, during weekends, or whenever I had free time.

There were countless bugs, failed experiments, broken training runs, and moments where I seriously questioned whether things would ever work. Looking back, every mistake ended up teaching me something valuable. The hardest part wasn't just building the model, but understanding why certain architectural choices matter and when to apply them.

Over time, TiniMind has evolved from an experimental notebook into a standalone training pipeline capable of running on local machines and cloud environments while keeping the codebase educational and easy to understand.

TiniMind is still in its Beta phase, so expect rough edges, unfinished features, and things that will continue to evolve over time.

This repository represents not only a software project, but also my learning journey into AI and language model development.

If you find this project useful, interesting, or learned something from it, consider giving it a ⭐.

## Related Project: Gabut Playground

Another project that helped shape TiniMind is [Gabut Playground](https://github.com/Makluk-Gabut/Gabut-Playground).

It's my personal sandbox where I experiment with ideas, prototypes, and random AI-related projects before deciding whether they're worth integrating into TiniMind. Many experiments that eventually became features in TiniMind started there first.

If you're curious about what happens behind the scenes, feel free to check it out!
