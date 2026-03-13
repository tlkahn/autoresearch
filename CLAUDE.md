# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**autoresearch** is an autonomous AI research framework where an LLM agent independently conducts neural network research by modifying `train.py`, training for a fixed 5-minute budget, and keeping improvements or discarding failures in a loop.

- **`prepare.py`** — Fixed data prep & evaluation harness. **DO NOT MODIFY.**
- **`train.py`** — Model architecture & training loop. **This is the only file the agent modifies.**
- **`program.md`** — Human-written agent instructions defining the experimental loop.

## Commands

```bash
uv sync                          # Install dependencies
uv run prepare.py                # One-time: download data shards + train BPE tokenizer (~2 min)
uv run train.py                  # Run single training experiment (~5 min, requires GPU)
uv run train.py > run.log 2>&1   # Run and capture output
grep "^val_bpb:\|^peak_vram_mb:" run.log  # Extract key metrics from log
```

There is no test suite or linter configured.

## Architecture

### Data Flow

`prepare.py` downloads parquet shards from HuggingFace (`climbmix-400b-shuffle`) and trains a BPE tokenizer (vocab_size=8192). All cached to `~/.cache/autoresearch/`.

`train.py` imports from `prepare.py`: `MAX_SEQ_LEN`, `TIME_BUDGET`, `Tokenizer`, `make_dataloader()`, `evaluate_bpb()`. It defines the GPT model, optimizer, and training loop, then runs until `TIME_BUDGET` (300s) expires and reports `val_bpb` (bits per byte, lower is better).

### Model (`train.py`)

- **GPT** with configurable depth, heads, embeddings, and sliding window attention pattern (`WINDOW_PATTERN="SSSL"`)
- **CausalSelfAttention** using Flash Attention 3 + RoPE + optional value embeddings
- **MLP** with ReLU² activation
- **MuonAdamW** hybrid optimizer: Muon (polar orthogonalization) for 2D weight matrices, AdamW for embeddings/scalars

### Key Constants (in `prepare.py`, immutable)

- `MAX_SEQ_LEN = 2048`, `TIME_BUDGET = 300`, `VOCAB_SIZE = 8192`
- `EVAL_TOKENS = 40 * 524288` (~21M tokens for validation)
- Validation uses a pinned shard (6542) for reproducibility

### Tunable Hyperparameters (in `train.py`)

`ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN`, `DEPTH`, `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`, `EMBEDDING_LR`, `MATRIX_LR`, `WEIGHT_DECAY`, `WARMUP_RATIO`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`

## Experimental Workflow (from `program.md`)

1. Branch: `git checkout -b autoresearch/<date>`
2. Initialize `results.tsv` with header
3. Loop: modify `train.py` → commit → `uv run train.py` → extract `val_bpb` → log to `results.tsv` → keep if improved, `git reset` if not
4. One idea per commit. Agent runs indefinitely.

## Environment

- Python 3.10+, single NVIDIA GPU (H100-class), CUDA 12.8+, PyTorch 2.9.1 with bfloat16
- Dependency manager: `uv`
- Key deps: `torch`, `kernels` (Flash Attention 3), `tiktoken`, `rustbpe`
