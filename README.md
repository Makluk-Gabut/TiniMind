
# TiniMind

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Beta-orange)

> A lightweight decoder-only Transformer built from scratch in PyTorch.

# ⚠️ **Beta Phase**

TiniMind is currently under active development. The architecture, APIs, training pipeline, and project structure may change as development progresses.
---

# Why TiniMind?

TiniMind is an experimental Small Language Model (SLM) focused on Indonesian.

The primary goal of this project is to build a lightweight, educational, and efficient Transformer implementation that demonstrates modern language model architectures while remaining easy to understand, modify, and extend.

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
- ✅ Custom Indonesian SentencePiece Tokenizer (32k Vocabulary)
- ✅ Stable Quantization
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
| Documentation            | Stable       |
| Quantization             | Stable       |
| Evaluation Benchmark     | Planned      |
| Pretrained Model         | Planned      |
| Stable Release           | In Progress  |

---

# Roadmap

- [x] Indonesian BPE Tokenizer
- [x] Decoder-only Transformer
- [x] Flash Attention
- [x] Grouped Query Attention
- [x] RMSNorm
- [x] Stable Quantization
- [x] Complete Training Pipeline
- [x] Standalone Training Script
- [x] Resume Training
- [ ] Evaluation Benchmark
- [ ] Hugging Face Integration
- [ ] Pretrained Model Release
- [ ] GGUF Export
- [ ] LoRA Support

---

# Repository Structure

|           File            |                        Description                        |
|---------------------------|-----------------------------------------------------------|
| `model_v2.py`             | Main Transformer architecture implementation              |
| `config.py`               | Model configuration and hyperparameters                   |
| `train.py`                | Standalone training script with checkpoint resume support |
| `quantize.py`             | Quantization utilities                                    |
| `train_tokenizer_indo.py` | Indonesian tokenizer training script                      |
| `TiniMind_Training.ipynb` | Interactive notebook for experimentation and tutorials    |
| `requirements.txt`        | Python dependencies                                       |
| `LICENSE`                 | MIT License                                               |

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

Edit the model configuration inside:

```
config.py
```

### 3. Start training

Run the standalone training script:

```bash
python train.py \
    --config prod_500m \
    --data-dir ./data \
    --output-dir ./output/checkpoints
```

Resume training from a checkpoint:

```bash
python train.py \
    --config prod_500m \
    --data-dir ./data \
    --output-dir ./output/checkpoints \
    --resume ./output/checkpoints/step_xxxxxxx_loss_x.xxx.pt
```

For interactive experimentation and development, you can still use:

```
TiniMind_Training.ipynb
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

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for more information.

---

# Dev's Note

If you've made it this far, thank you for taking the time to explore TiniMind.

This project has been my personal playground for learning and experimenting with modern Transformer architectures over the past eight months.

I'm building this project entirely on my own while still attending school. Most of the development happened after classes, during weekends, or whenever I had free time.

There were countless bugs, failed experiments, broken training runs, and moments where I seriously questioned whether things would ever work. Looking back, every mistake ended up teaching me something valuable.

Over time, TiniMind has evolved from an experimental notebook into a standalone training pipeline capable of running on local machines and cloud environments while keeping the codebase educational and easy to understand.

Fun fact: I accidentally burned my finger on my PC fan while writing this README. Somehow documenting the project turned out to be harder than building the model itself. 😭

TiniMind is still in its Beta phase, so expect rough edges, unfinished features, and things that will continue to evolve over time.

This repository represents not only a software project, but also my learning journey into AI and language model development.

If you find this project useful, interesting, or learned something from it, consider giving it a ⭐.

Another project that helped shape TiniMind is Gabut Playground.

It's my personal sandbox where I experiment with ideas, prototypes, and random AI-related projects before deciding whether they're worth integrating into TiniMind. Many experiments that eventually became part of this repository started there.

If you're curious about what happens behind the scenes, feel free to check it out.

Gabut Playground: https://github.com/Makluk-Gabut/Gabut-Playground
