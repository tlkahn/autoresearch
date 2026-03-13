"""
Alternative implementation of prepare.py using mainstream libraries.

Replacements:
  requests + pyarrow.parquet  →  huggingface datasets
  rustbpe + tiktoken          →  huggingface tokenizers
  custom generator            →  torch.utils.data.IterableDataset

Public API (imported by train.py) is identical:
  MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

Usage:
    python prepare_hf.py                  # full prep (download + tokenizer)
    python prepare_hf.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch_hf/.
"""

import os
import sys
import time
import math
import argparse

import torch
from torch.utils.data import IterableDataset

from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer as _HFTokenizer, Regex
from tokenizers import models, trainers, pre_tokenizers, decoders

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify — identical to prepare.py)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_hf")
TRAIN_DIR = os.path.join(CACHE_DIR, "train_dataset")
VAL_DIR = os.path.join(CACHE_DIR, "val_dataset")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

DATASET_ID = "karpathy/climbmix-400b-shuffle"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download (HuggingFace datasets — replaces requests + pyarrow)
# ---------------------------------------------------------------------------

def download_data(num_shards, download_workers=8):
    """Download and cache train/val splits using HuggingFace datasets."""
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        train_ds = load_from_disk(TRAIN_DIR)
        val_ds = load_from_disk(VAL_DIR)
        print(f"Data: already prepared ({len(train_ds)} train, {len(val_ds)} val rows) at {CACHE_DIR}")
        return

    num_train = min(num_shards, MAX_SHARD)
    train_files = [f"shard_{i:05d}.parquet" for i in range(num_train) if i != VAL_SHARD]
    val_files = [f"shard_{VAL_SHARD:05d}.parquet"]

    if not train_files:
        print("Data: need at least 1 training shard.")
        sys.exit(1)

    print(f"Data: downloading {len(train_files)} train + 1 val shard via HuggingFace datasets...")

    train_ds = load_dataset(DATASET_ID, data_files=train_files, cache_dir=CACHE_DIR, split="train")
    train_ds.save_to_disk(TRAIN_DIR, num_proc=min(download_workers, len(train_files)))
    print(f"  Train: {len(train_ds)} rows saved to {TRAIN_DIR}")

    val_ds = load_dataset(DATASET_ID, data_files=val_files, cache_dir=CACHE_DIR, split="train")
    val_ds.save_to_disk(VAL_DIR)
    print(f"  Val:   {len(val_ds)} rows saved to {VAL_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training (HuggingFace tokenizers — replaces rustbpe + tiktoken)
# ---------------------------------------------------------------------------

def _text_iterator(ds, max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from a HF dataset for tokenizer training."""
    nchars = 0
    for row in ds:
        doc = row["text"]
        if len(doc) > doc_cap:
            doc = doc[:doc_cap]
        nchars += len(doc)
        yield doc
        if nchars >= max_chars:
            return


def train_tokenizer():
    """Train BPE tokenizer using HuggingFace tokenizers, save as tokenizer.json."""
    tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_path) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_DIR):
        print("Tokenizer: training data not found. Run download_data() first.")
        sys.exit(1)

    train_ds = load_from_disk(TRAIN_DIR)

    if len(train_ds) == 0:
        print("Tokenizer: training dataset is empty. Download more data first.")
        sys.exit(1)

    # --- Build HF tokenizer ---
    print("Tokenizer: training BPE (HuggingFace tokenizers)...")
    t0 = time.time()

    tok = _HFTokenizer(models.BPE())

    # Pre-tokenizer: regex split (same pattern as original), then byte-level
    # encoding so BPE operates on bytes — matching tiktoken's byte-level BPE.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tok.train_from_iterator(_text_iterator(train_ds), trainer=trainer)
    tok.save(tokenizer_path)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_path}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(tok.get_vocab_size()):
        decoded = tok.decode([token_id])
        if decoded in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(decoded.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = tok.encode(test).ids
    decoded = tok.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={tok.get_vocab_size()})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Drop-in replacement for prepare.py's Tokenizer, backed by HF tokenizers."""

    def __init__(self, hf_tok):
        self._tok = hf_tok
        self.bos_token_id = hf_tok.token_to_id(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        tok = _HFTokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        return cls(tok)

    def get_vocab_size(self):
        return self._tok.get_vocab_size()

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):  # noqa: ARG002
        prepend_id = None
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self._tok.token_to_id(prepend)

        if isinstance(text, str):
            ids = self._tok.encode(text).ids
            if prepend_id is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            encoded_batch = self._tok.encode_batch(text)
            ids = [enc.ids for enc in encoded_batch]
            if prepend_id is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self._tok.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)

# ---------------------------------------------------------------------------
# Dataloader (PyTorch IterableDataset — replaces raw pyarrow iteration)
# ---------------------------------------------------------------------------

class _DocumentStream(IterableDataset):
    """
    Infinite document stream backed by a HuggingFace Arrow dataset.

    Replaces the original's raw pyarrow.ParquetFile iteration with a
    PyTorch IterableDataset wrapping datasets.load_from_disk().
    The dataset is memory-mapped, so RAM usage stays low.
    """

    def __init__(self, split):
        assert split in ("train", "val")
        self.split = split

    def __iter__(self):
        path = TRAIN_DIR if self.split == "train" else VAL_DIR
        ds = load_from_disk(path)
        epoch = 1
        while True:
            for row in ds:
                yield row["text"], epoch
            epoch += 1


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from HuggingFace dataset."""
    stream = _DocumentStream(split)
    batch = []
    current_epoch = 1
    for text, epoch in stream:
        current_epoch = epoch
        batch.append(text)
        if len(batch) == tokenizer_batch_size:
            yield batch, current_epoch
            batch = []
    # unreachable — stream is infinite


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).

    Same interface as prepare.py — yields (inputs, targets, epoch).
    Internally uses HuggingFace datasets instead of raw pyarrow.
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer (HuggingFace edition)")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8,
                        help="Number of parallel workers for dataset processing")
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")
