# `prepare_hf.py` 详解：用主流库重写数据准备模块

`prepare_hf.py` 是 `prepare.py` 的替代实现，使用 HuggingFace `datasets` + HuggingFace `tokenizers` + PyTorch `IterableDataset` 三大主流库，提供完全相同的公开接口，让 `train.py` 无需改动即可切换使用。

## 一、为什么需要替代实现？

原版 `prepare.py` 使用了一套"小而精"的工具链：`requests` 手动下载、`pyarrow` 读取 Parquet、`rustbpe` 训练分词器、`tiktoken` 做推理。这些选择性能优秀，但不是主流方案。

`prepare_hf.py` 的目标是用"大家都在用"的库实现同样的功能。就像同一道菜，原版用的是专业厨师的定制刀具，这个版本用的是超市里人人都能买到的标准厨具。功能一样，工具不同。

下面是两个版本的对照。

```
┌──────────────────┬─────────────────────────┬──────────────────────────────┐
│  功能            │  prepare.py (原版)       │  prepare_hf.py (替代版)       │
├──────────────────┼─────────────────────────┼──────────────────────────────┤
│  数据下载         │  requests + 手动重试     │  datasets.load_dataset()     │
│  数据存储/读取    │  pyarrow.parquet        │  datasets Arrow 格式          │
│  分词器训练       │  rustbpe (Rust)         │  tokenizers BpeTrainer       │
│  分词器推理       │  tiktoken (Rust)        │  tokenizers (Rust)           │
│  数据迭代         │  自定义生成器            │  torch IterableDataset       │
│  缓存目录         │  ~/.cache/autoresearch/ │  ~/.cache/autoresearch_hf/   │
└──────────────────┴─────────────────────────┴──────────────────────────────┘
```

## 二、数据下载：`datasets.load_dataset()`

原版用 `requests.get()` 逐个下载 Parquet 文件，自己实现重试、原子写入、多进程并行。替代版只需要一行。

```python
from datasets import load_dataset, load_from_disk

train_ds = load_dataset(
    "karpathy/climbmix-400b-shuffle",
    data_files=train_files,       # ["shard_00000.parquet", ...]
    cache_dir=CACHE_DIR,
    split="train",
)
train_ds.save_to_disk(TRAIN_DIR)  # 持久化为 Arrow 格式
```

上面代码中，[HuggingFace datasets](https://github.com/huggingface/datasets) 在底层自动处理了：

（1）**HTTP 下载与缓存**：下载过的文件不会重复下载，自动存入 `cache_dir`。

（2）**Parquet → Arrow 转换**：Parquet 文件被转换为 Apache Arrow 格式，支持内存映射（memory-mapped），不需要把整个数据集加载进 RAM。

（3）**完整性校验**：通过文件哈希确保数据一致性。

验证集的处理方式也很直观——单独下载 `shard_06542.parquet` 并保存到 `VAL_DIR`。训练和验证数据完全隔离，运行时通过 `load_from_disk()` 按需加载。

```python
# 运行时加载（内存映射，几乎零开销）
train_ds = load_from_disk(TRAIN_DIR)
val_ds = load_from_disk(VAL_DIR)
```

## 三、分词器训练：HuggingFace `tokenizers`

原版的分词器训练分两步：用 `rustbpe` 训练，再导出为 `tiktoken` 格式。替代版用 HuggingFace [`tokenizers`](https://github.com/huggingface/tokenizers) 库一步完成。

```python
from tokenizers import Tokenizer as _HFTokenizer, Regex
from tokenizers import models, trainers, pre_tokenizers, decoders

tok = _HFTokenizer(models.BPE())

# 预分词：先用 GPT-4 风格的正则切分，再转为字节级表示
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,                                # 8192
    special_tokens=SPECIAL_TOKENS,                        # 4 个保留 token
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), # 256 字节基础字符
    show_progress=True,
)

tok.train_from_iterator(text_iterator, trainer=trainer)
tok.save("tokenizer.json")
```

上面代码中，有几个关键设计：

（1）**`pre_tokenizers.Sequence`**：链式组合两个预分词器。先用正则表达式切分（与原版相同的 `SPLIT_PATTERN`），再用 `ByteLevel` 将每个片段转换为字节级表示。这样 BPE 在字节上操作，与原版 `tiktoken` 的行为一致。

（2）**`ByteLevel` 编解码**：将每个字节映射为一个 Unicode 字符（例如空格 `0x20` → `Ġ`）。这确保任意字节序列都能被分词，不会因遇到未知字符而失败。`decoders.ByteLevel()` 负责反向映射。

（3）**`initial_alphabet`**：确保全部 256 个字节值从一开始就在词表中，BPE 合并在此基础上进行。

训练产出保存为 `tokenizer.json`（HuggingFace 标准格式），而非原版的 `tokenizer.pkl`（pickle）。JSON 格式的好处是可读性强、跨语言兼容。

## 四、`Tokenizer` 封装类

为了让 `train.py` 无感知切换，`prepare_hf.py` 提供了与原版完全相同的 `Tokenizer` 接口。

```python
class Tokenizer:
    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        tok = _HFTokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        return cls(tok)

    def encode(self, text, prepend=None, num_threads=8):
        if isinstance(text, str):
            ids = self._tok.encode(text).ids
        elif isinstance(text, list):
            ids = [enc.ids for enc in self._tok.encode_batch(text)]
        return ids

    def decode(self, ids):
        return self._tok.decode(ids)
```

上面代码中，对比原版 `tiktoken` 的调用方式：

```
原版：  enc.encode_ordinary(text)        → [int]
替代版：self._tok.encode(text).ids       → [int]

原版：  enc.encode_ordinary_batch(texts)  → [[int]]
替代版：self._tok.encode_batch(texts)     → [Encoding] → 取 .ids
```

`encode_batch` 在 Rust 层自动并行化（使用 Rayon），不需要手动指定线程数。`num_threads` 参数保留只是为了接口兼容。

## 五、数据加载：PyTorch `IterableDataset`

原版的数据迭代直接操作 `pyarrow.ParquetFile`，逐个读取 row group。替代版用 PyTorch 的 `IterableDataset` 封装 HuggingFace 数据集。

```python
from torch.utils.data import IterableDataset

class _DocumentStream(IterableDataset):
    """无限文档流，包装 HuggingFace Arrow 数据集。"""

    def __init__(self, split):
        self.split = split
        self._ds = None  # 延迟加载

    def _load(self):
        path = TRAIN_DIR if self.split == "train" else VAL_DIR
        self._ds = load_from_disk(path)

    def __iter__(self):
        if self._ds is None:
            self._load()
        epoch = 1
        while True:
            for row in self._ds:
                yield row["text"], epoch
            epoch += 1
```

上面代码中，`_DocumentStream` 是一个标准的 PyTorch `IterableDataset`。它的特点是：

（1）**延迟加载**：数据集在第一次迭代时才加载（`_load`），避免不必要的内存占用。

（2）**无限迭代**：外层 `while True` 确保数据可以被无限遍历，每轮遍历完成后 `epoch` 自增。

（3）**内存映射**：`load_from_disk()` 返回的数据集是内存映射的，不需要把所有文本加载进 RAM。

best-fit 装箱逻辑与原版完全相同——这部分算法与数据源无关，只关心"给我一批文档"。替代版只是换了文档的来源。

## 六、两个版本的差异

虽然接口一致，但两个版本存在一些内在差异。

**分词结果不同**：HuggingFace `tokenizers` 和 `rustbpe` 是不同的 BPE 实现。即使使用相同的训练数据和词表大小，学到的合并规则也会略有不同。因此，两个版本训练出的分词器**不可互换**——用 `prepare.py` 的分词器训练的模型，不能用 `prepare_hf.py` 的分词器评估，反之亦然。

**缓存目录隔离**：替代版使用 `~/.cache/autoresearch_hf/`，与原版的 `~/.cache/autoresearch/` 互不干扰。两者可以共存。

**序列化格式**：原版用 Python pickle（`.pkl`），替代版用 JSON（`.json`）。JSON 更通用，但 pickle 加载更快。

## 七、如何使用

安装依赖（需要额外安装 `datasets` 和 `tokenizers`）。

```bash
uv add datasets tokenizers
```

运行完整的数据准备流程。

```bash
uv run prepare_hf.py
```

只下载少量分片用于测试。

```bash
uv run prepare_hf.py --num-shards 4
```

在 `train.py` 中切换使用替代版（只需改一行 import）。

```python
# 原版
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# 替代版
from prepare_hf import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
```

## 八、参考链接

- [huggingface/datasets — GitHub](https://github.com/huggingface/datasets)：HuggingFace 数据集加载与处理库
- [huggingface/tokenizers — GitHub](https://github.com/huggingface/tokenizers)：高性能分词器训练与推理库
- [PyTorch IterableDataset 文档](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)：PyTorch 可迭代数据集基类
- [karpathy/climbmix-400b-shuffle — HuggingFace](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)：本项目使用的预训练数据集

（完）
