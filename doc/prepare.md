# `prepare.py` 详解：自动化研究框架的数据准备模块

`prepare.py` 是 autoresearch 项目的数据准备脚本，负责从 HuggingFace 下载训练数据分片，并训练一个 BPE 分词器，为后续的神经网络训练实验提供基础设施。

## 一、`prepare.py` 解决什么问题？

训练一个语言模型，首先要解决两个问题：（1）去哪里拿训练数据？（2）怎么把文本变成模型能读懂的数字？

`prepare.py` 就是干这两件事的。你可以把它想象成厨房的"备菜"环节——正式开火（训练模型）之前，先把食材洗好切好摆整齐。它只需要运行一次，之后的所有训练实验都复用它的产出。

整个流程如下。

```
+---------------------+     +---------------------+     +-------------------+
|  HuggingFace 远程   |     |  ~/.cache/           |     |  train.py         |
|  Parquet 分片       | --> |  autoresearch/data/  | --> |  导入并使用       |
|  (climbmix-400b)    |     |  autoresearch/       |     |  Tokenizer,       |
+---------------------+     |  tokenizer/          |     |  make_dataloader,  |
                             +---------------------+     |  evaluate_bpb     |
                                                         +-------------------+
```

上面示例中，左边是远程数据源，中间是本地缓存，右边是训练脚本。`prepare.py` 负责把数据从左边搬到中间，同时在中间训练好分词器。

## 二、核心常量

`prepare.py` 定义了一组**不可修改**的常量，它们是整个实验框架的"宪法"。

```python
MAX_SEQ_LEN = 2048          # 上下文窗口长度
TIME_BUDGET = 300           # 训练时间预算（秒），即 5 分钟
EVAL_TOKENS = 40 * 524288   # 验证集评估使用的 token 数（约 2100 万）
VOCAB_SIZE = 8192           # 词表大小
```

上面代码中，`MAX_SEQ_LEN` 决定了模型一次能"看到"多长的文本；`TIME_BUDGET` 限制每次实验只跑 5 分钟，保证快速迭代；`VOCAB_SIZE` 设为 8192，比 GPT-4 的 10 万级词表小得多，这是为了在小规模实验中降低嵌入层的参数量。

另外，数据源来自 Andrej Karpathy 在 HuggingFace 上发布的 [`climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) 数据集，总共有 6543 个 Parquet 分片（编号 0 到 6542）。其中，**最后一个分片（6542）被固定用作验证集**，不参与训练，确保评估结果可复现。

## 三、数据下载

下载逻辑在 `download_data()` 和 `download_single_shard()` 两个函数中实现。

```python
def download_single_shard(index):
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            return True
        except (requests.RequestException, IOError) as e:
            # 指数退避重试
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False
```

上面代码中，有几个值得注意的设计：

（1）**幂等性**：如果文件已存在，直接跳过，不会重复下载。

（2）**原子写入**：先写到 `.tmp` 临时文件，成功后再 `os.rename`。这样即使下载中途断了，也不会留下一个残缺的正式文件。这就像你网购时，快递不会直接扔到你家里，而是先放到驿站，你确认收货后才算完成。

（3）**指数退避重试**：失败后等 2、4、8、16 秒再试，最多 5 次。这是分布式系统中处理瞬时故障的经典模式。

外层 `download_data()` 使用 `multiprocessing.Pool` 并行下载，默认 8 个 worker。

```python
def download_data(num_shards, download_workers=8):
    # ...
    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)
```

## 四、BPE 分词器训练

分词器（Tokenizer）的作用是把文本切成一个个"词元"（token）。BPE（Byte Pair Encoding）是目前主流大模型使用的分词算法。

它的核心思想很简单：从单个字节开始，反复合并出现频率最高的相邻字节对，直到词表达到目标大小。就像你压缩文件一样——把重复出现的模式用更短的符号替代。

训练代码如下。

```python
def train_tokenizer():
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)  # 8192 - 4 = 8188
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)
```

上面代码中，[`rustbpe`](https://github.com/karpathy/rustbpe) 是 Karpathy 用 Rust 编写的高性能 BPE 训练库。它负责"训练"——从语料中学习合并规则。训练完成后，合并规则被导出为 [`tiktoken`](https://github.com/openai/tiktoken) 格式：

```python
enc = tiktoken.Encoding(
    name="rustbpe",
    pat_str=pattern,
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens,
)
with open(tokenizer_pkl, "wb") as f:
    pickle.dump(enc, f)
```

上面代码中，`tiktoken` 是 OpenAI 开源的高性能分词推理库。训练用 `rustbpe`，推理用 `tiktoken`——各取所长，这是一个很聪明的工程决策。

此外，还会生成一个 `token_bytes.pt` 文件，记录每个 token 对应的 UTF-8 字节数。这是后面计算 **bits per byte（BPB）** 指标的关键。

```python
for token_id in range(enc.n_vocab):
    token_str = enc.decode([token_id])
    if token_str in special_set:
        token_bytes_list.append(0)     # 特殊 token 不计入字节数
    else:
        token_bytes_list.append(len(token_str.encode("utf-8")))
```

最后，用一句中文测试确保分词器能正确往返编解码。

```python
test = "Hello world! Numbers: 123. Unicode: 你好"
encoded = enc.encode_ordinary(test)
decoded = enc.decode(encoded)
assert decoded == test
```

## 五、运行时工具：`Tokenizer` 类

`prepare.py` 不仅是一个独立脚本，它同时也是一个**模块**，被 `train.py` 导入使用。`Tokenizer` 类就是提供给训练脚本的接口。

```python
class Tokenizer:
    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def encode(self, text, prepend=None, num_threads=8):
        # 支持单条文本和批量文本
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
        return ids
```

上面代码中，`from_directory` 是一个工厂方法，从磁盘加载预训练好的分词器。`encode` 方法支持两种输入：单条字符串或字符串列表（批量编码），批量编码时利用多线程加速。

## 六、数据加载器：best-fit 装箱算法

`make_dataloader()` 是整个文件中最精妙的部分。它负责把变长的文档打包成固定长度的训练样本。

这个问题本质上是一个**装箱问题**（bin packing）：你有一堆大小不同的物品（文档），要装进固定容量的箱子（序列长度 `T+1`）里，尽量不浪费空间。

```
行容量 = T + 1 = 2049 个 token

┌─────────────────────────────────────────────────────┐
│ [BOS]Doc_A(800) │ [BOS]Doc_C(500) │ [BOS]Doc_F(749) │   ← 恰好填满
├─────────────────────────────────────────────────────┤
│ [BOS]Doc_B(1200)│ [BOS]Doc_D(848..裁剪)            │   ← 最短文档被裁剪以填满
└─────────────────────────────────────────────────────┘
```

算法的核心逻辑如下。

```python
while pos < row_capacity:
    # 在缓冲区中找能完整放入的最大文档
    best_idx = -1
    best_len = 0
    for i, doc in enumerate(doc_buffer):
        doc_len = len(doc)
        if doc_len <= remaining and doc_len > best_len:
            best_idx = i
            best_len = doc_len

    if best_idx >= 0:
        doc = doc_buffer.pop(best_idx)
        row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc)
        pos += len(doc)
    else:
        # 没有文档能放下——裁剪最短文档来填满剩余空间
        shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
        doc = doc_buffer.pop(shortest_idx)
        row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining])
        pos += remaining
```

上面代码中，策略分两步：（1）优先找能完整放下的最大文档（best-fit）；（2）如果没有文档能放下，裁剪最短的一篇来精确填满。这样做的好处是 **100% 利用率**，没有任何 padding 浪费。

每个文档前面都会加一个 `BOS`（Begin of Sequence）标记，模型可以据此识别文档边界。

数据在 CPU 的 pinned memory 上组装，然后通过 `non_blocking=True` 异步传输到 GPU，实现 CPU/GPU 流水线并行。

## 七、评估指标：Bits Per Byte (BPB)

最后一个核心组件是 `evaluate_bpb()` 函数。

```python
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
```

上面代码中，**BPB（bits per byte）** 是一个与词表大小无关的评估指标。它衡量的是：模型平均需要多少比特来编码原始文本中的每个字节。

为什么不直接用 loss（交叉熵）？因为不同的分词器有不同的词表大小，同样的文本被切成的 token 数量不同，导致 loss 不可比。BPB 绕过了这个问题，回到了最底层的"字节"作为统一度量单位。

计算公式是：

```
BPB = total_nats / (ln(2) × total_bytes)
```

其中 `total_nats` 是所有目标 token 的交叉熵损失之和（以 nat 为单位），`total_bytes` 是这些 token 对应的 UTF-8 字节总数。特殊 token（字节数为 0）被排除在外。

## 八、如何使用

安装依赖。

```bash
uv sync
```

运行完整的数据准备流程（下载 10 个分片 + 训练分词器）。

```bash
uv run prepare.py
```

如果只想下载少量分片用于测试，可以这样做。

```bash
uv run prepare.py --num-shards 4
```

下载全部分片（6543 个）。

```bash
uv run prepare.py --num-shards -1
```

所有产出缓存在 `~/.cache/autoresearch/` 目录下，目录结构如下。

```
~/.cache/autoresearch/
├── data/
│   ├── shard_00000.parquet
│   ├── shard_00001.parquet
│   ├── ...
│   └── shard_06542.parquet      ← 固定验证集
└── tokenizer/
    ├── tokenizer.pkl            ← tiktoken 格式的分词器
    └── token_bytes.pt           ← 每个 token 的字节数查找表
```

## 九、参考链接

- [karpathy/rustbpe — GitHub](https://github.com/karpathy/rustbpe)：Karpathy 编写的 BPE 分词器训练库
- [openai/tiktoken — GitHub](https://github.com/openai/tiktoken)：OpenAI 的高性能 BPE 分词推理库
- [karpathy/climbmix-400b-shuffle — HuggingFace](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)：本项目使用的预训练数据集
- [Byte Pair Encoding — Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding)：BPE 算法的原理介绍

（完）
