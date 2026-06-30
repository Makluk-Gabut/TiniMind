````md
# TiniMind

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Beta-orange)

> A lightweight decoder-only Transformer built from scratch in PyTorch.

> ⚠️ **Beta Phase**

TiniMind is currently under active development. The architecture, APIs, training pipeline, and project structure may change as development progresses. Some features are still experimental and may not function as expected.

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
- ✅ Implemented Quantization *(currently under development)*
- ✅ Modular project structure for experimentation

---

# Current Status

|        Component         |    Status    |
|--------------------------|--------------|
| Indonesian Tokenizer     |  Stable      |
| Decoder-only Transformer |  Stable      |
| Configuration System     |  Stable      |
| Training Notebook        |  Working     |
| Documentation            |  Improving   |
| Quantization             |  Improving   |
| Evaluation Benchmark     |  Planned     |
| Pretrained Model         |  Planned     |
| Stable Release           |  In Progress |

---

# Roadmap

- [x] Indonesian BPE Tokenizer
- [x] Decoder-only Transformer
- [x] Flash Attention
- [x] Grouped Query Attention
- [x] RMSNorm
- [x] Stable Quantization
- [ ] Complete Training Pipeline
- [ ] Evaluation Benchmark
- [ ] Hugging Face Integration
- [ ] Pretrained Model Release
- [ ] GGUF Export
- [ ] LoRA Support

---

# Repository Structure

|           File            |                   Description                      |
|---------------------------|----------------------------------------------------|
| `model_v2.py`             | Main Transformer architecture implementation       |
| `config.py`               | Model configuration and hyperparameters            |
| `quantize.py`             | Experimental quantization utilities                |
| `train_tokenizer_indo.py` | Indonesian tokenizer training script               |
| `TiniMind_Training.ipynb` | Notebook for tokenizer training and model training |
| `requirements.txt`        | Python dependencies                                |
| `LICENSE`                 | MIT License                                        |

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

Open the notebook:

```
TiniMind_Training.ipynb
```

Run every cell sequentially.

---

# Requirements

- Python 3.10+
- PyTorch 2.x
- CUDA-compatible GPU (recommended)
- SentencePiece
- bitsandbytes *(only required for experimental quantization)*

---

# Philosophy

TiniMind is **not** intended to compete with large commercial language models.

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

Fun fact: I accidentally burned my finger on my PC fan while writing this README. Somehow documenting the project turned out to be harder than building the model itself. 😭

TiniMind is still in its Beta phase, so expect rough edges, unfinished features, and things that will continue to evolve over time.

This repository represents not only a software project, but also my learning journey into AI and language model development.

If you find this project useful, interesting, or learned something from it, consider giving it a ⭐. It genuinely motivates me to keep improving TiniMind.

Thanks for stopping by
````
