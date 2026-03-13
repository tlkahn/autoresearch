`train.py` 是 autoresearch 项目的核心训练脚本，它在单张 GPU 上从零开始预训练一个 GPT 语言模型，并在固定的 5 分钟时间预算内完成整个训练过程。

## 一、这个脚本解决什么问题？

训练一个语言模型通常需要大量的代码和复杂的分布式配置。`train.py` 的目标正好相反——把所有东西塞进一个文件里，用一张 GPU、一条命令就能跑起来。

运行方式如下。

```bash
uv run train.py
```

整个脚本大约 630 行，包含了模型定义、优化器实现、超参数配置和训练循环四个部分。你可以把它想象成一个「迷你炼丹炉」：数据从一端进去，经过 5 分钟的「冶炼」，另一端输出一个可以预测下一个词的模型。

脚本的整体数据流如下。

```
+-------------+     +------------+     +-----------+     +------------+
|  prepare.py | --> | Dataloader | --> |  GPT 模型  | --> | val_bpb 评估|
| (数据+分词器) |     | (批量数据)   |     | (前向+反向) |     | (验证指标)   |
+-------------+     +------------+     +-----------+     +------------+
```

上面的流程中，`prepare.py` 负责下载数据和训练分词器（这是一个独立脚本，不能修改），`train.py` 负责剩下的所有事情。

## 二、模型架构：GPT 的核心组件

`train.py` 定义了一个标准的 GPT 模型，由三个核心模块组成。

（1）**`CausalSelfAttention`**——因果自注意力层。这是 Transformer 的核心，负责让每个 token「看到」它前面的所有 token。

（2）**`MLP`**——前馈神经网络层。它对注意力层的输出做非线性变换，使用了 **ReLU²**（先 ReLU 再平方）作为激活函数。

（3）**`Block`**——一个 Transformer 块，将注意力层和 MLP 层组合起来，并通过残差连接相加。

模型的基本结构如下。

```
Input tokens
    │
    ▼
┌─────────┐
│Embedding│  ← 词嵌入：token → 向量
└────┬────┘
     │ (RMS Norm)
     ▼
┌─────────────────────┐
│ Block 0              │
│  ├─ Attention + RoPE │  ← 因果自注意力 + 旋转位置编码
│  └─ MLP (ReLU²)      │  ← 前馈网络
├─────────────────────┤
│ Block 1              │
│  ├─ Attention + RoPE │
│  └─ MLP (ReLU²)      │
├─────────────────────┤
│ ...                  │  ← 共 DEPTH 层（默认 8 层）
├─────────────────────┤
│ Block N              │
└────┬────────────────┘
     │ (RMS Norm)
     ▼
┌─────────┐
│ lm_head │  ← 线性层：向量 → 词汇表概率
└────┬────┘
     │ (softcap=15)
     ▼
  Logits → Loss
```

下面是 `GPTConfig` 数据类的定义。

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

上面代码中，`sequence_len` 是模型能处理的最大序列长度，`n_layer` 是 Transformer 的层数，`n_head` 是注意力头数，`n_embd` 是模型的隐藏维度。`window_pattern` 控制滑动窗口注意力的模式——后面会详细讲。

## 三、三个关键技术细节

### 3.1 旋转位置编码（RoPE）

传统的 Transformer 使用固定的位置编码来告诉模型每个 token 的位置。**RoPE**（Rotary Position Embedding）采用了一种更优雅的方式：它通过旋转向量的角度来编码位置信息。

就好比你站在钟表中心看指针——12 点和 1 点的「距离」，和 3 点和 4 点的「距离」是一样的，因为它们的角度差相同。RoPE 利用的正是这种性质，让模型天然地感知 token 之间的相对距离。

核心实现如下。

```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

上面代码中，将隐藏维度拆成两半（`x1` 和 `x2`），然后用三角函数对它们做旋转。这正是 RoPE 论文中描述的二维旋转操作。

### 3.2 滑动窗口注意力

标准的自注意力让每个 token 关注序列中所有前面的 token，计算量与序列长度的平方成正比。滑动窗口注意力是一种优化策略：大部分层只关注最近的一半上下文（`S` = short），少数层关注全部上下文（`L` = long）。

`WINDOW_PATTERN = "SSSL"` 表示：每 4 层中，前 3 层使用短窗口（1024 tokens），最后 1 层使用全窗口（2048 tokens）。

```python
def _compute_window_sizes(self, config):
    pattern = config.window_pattern.upper()
    long_window = config.sequence_len       # 2048
    short_window = long_window // 2         # 1024
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    window_sizes[-1] = (long_window, 0)     # 最后一层强制使用全窗口
    return window_sizes
```

上面代码中，`window_sizes[-1]` 确保最后一层始终使用全窗口，这样模型最终输出时能看到完整的上下文。

### 3.3 Value Embedding（值嵌入）

这是来自 ResFormer 的技巧。普通的注意力层中，Value（V）矩阵只来自隐藏状态的线性变换。Value Embedding 额外引入了一个独立的嵌入表，让模型直接从原始 token ID 获取额外的 V 信息，并通过一个可学习的门控（gate）混合进来。

```python
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    v = v + gate.unsqueeze(-1) * ve
```

上面代码中，`sigmoid` 输出 0~1，乘以 2 后范围是 0~2。初始化时 gate 权重为零，`sigmoid(0) = 0.5`，乘以 2 等于 1.0——这意味着训练开始时，Value Embedding 以中性的比例混入，不会破坏模型的初始状态。

## 四、优化器：MuonAdamW

这个脚本使用了一个混合优化器 **`MuonAdamW`**，它将两种优化算法组合在一起：

（1）**AdamW**——用于 1D 参数（词嵌入、输出层、标量参数）。这是深度学习中最经典的优化器之一。

（2）**Muon**——用于 2D 矩阵参数（注意力层和 MLP 的权重矩阵）。Muon 是 2024 年提出的新型优化器，核心思想是对梯度的动量做**极分解**（Polar Decomposition）——提取梯度方向中的「纯旋转」分量，丢弃缩放信息。

打个比方：AdamW 像是一个按地图走路的人，会根据坡度调整步伐；Muon 更像一个指南针，它只关心「方向」，不关心「多陡」。Muon 使用 Newton-Schulz 迭代来近似极分解。

```python
# Polar express orthogonalization
X = g.bfloat16()
X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
if g.size(-2) > g.size(-1):
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B
```

上面代码中，通过 5 次迭代（`ns_steps=5`），将梯度矩阵逐步逼近一个正交矩阵。`polar_express_coeffs` 中存储的是预先计算好的多项式系数。

此外，Muon 还包含了两个重要的补充机制：

（1）**NorMuon 方差缩减**——使用二阶矩来归一化更新步长，类似 Adam 的自适应学习率。

（2）**Cautious weight decay**——只对与梯度方向一致的参数施加衰减（`mask = (g * stacked_params) >= 0`），避免「惩罚」那些已经在朝正确方向移动的参数。

## 五、超参数配置

脚本的超参数集中定义在文件中部，方便实验时修改。

```python
# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # 每个注意力头的维度
WINDOW_PATTERN = "SSSL" # 滑动窗口模式

# Optimization
TOTAL_BATCH_SIZE = 2**19  # ~524K tokens/step
EMBEDDING_LR = 0.6        # 词嵌入学习率
MATRIX_LR = 0.04           # 矩阵参数学习率 (Muon)
WEIGHT_DECAY = 0.2         # 权重衰减
WARMUP_RATIO = 0.0         # 学习率预热比例
WARMDOWN_RATIO = 0.5       # 学习率衰减比例

# Model size
DEPTH = 8                  # Transformer 层数
DEVICE_BATCH_SIZE = 128    # 单卡 batch size
```

上面代码中，模型维度通过 `DEPTH × ASPECT_RATIO` 自动计算得到（默认 `8 × 64 = 512`），然后向上取整到 `HEAD_DIM` 的整数倍。这种设计让模型大小和层数之间保持合理的比例关系。

学习率调度是基于时间进度（而非 step 数）的。

```python
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
```

上面代码中，`progress = training_time / TIME_BUDGET`，范围从 0 到 1。学习率先预热、后平稳、最后衰减——这是目前主流的梯形调度策略。

## 六、训练循环

训练循环是整个脚本的「主引擎」。

```python
while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # 更新学习率和优化器
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # 时间到了就停止
    if step > 10 and total_training_time >= TIME_BUDGET:
        break
```

上面代码中，有几个值得注意的细节：

（1）**梯度累积**——当显存不够放下完整的大 batch 时，将多个小 batch 的梯度累加后再更新参数。`grad_accum_steps = TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE × MAX_SEQ_LEN)`。

（2）**前 10 步不计时**——`if step > 10` 排除了模型编译（`torch.compile`）的预热时间，确保 5 分钟预算只用于实际训练。

（3）**NaN 快速失败**——如果 loss 变成 NaN 或超过 100，立即终止，避免浪费时间。

（4）**GC 管理**——第 0 步之后冻结 Python 的垃圾回收器（`gc.freeze()`），因为 Python GC 在训练中可能造成约 500ms 的卡顿，每 5000 步才手动回收一次。

## 七、评估与输出

训练结束后，脚本切换到评估模式。

```python
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
```

**`val_bpb`**（validation bits per byte）是核心评估指标。它表示模型平均需要多少比特来编码验证集中的每个字节——越低越好。这个指标比 perplexity 更直观：1 bpb 意味着平均每个字节只需 1 比特就能表示。

最终输出格式如下。

```
---
val_bpb:          1.234567
training_seconds: 300.0
total_seconds:    320.5
peak_vram_mb:     12345.6
mfu_percent:      45.00
total_tokens_M:   150.0
num_steps:        286
num_params_M:     50.0
depth:            8
```

上面示例中，`mfu_percent` 是模型浮点利用率（Model FLOPs Utilization），反映了 GPU 的实际利用效率——H100 的 BF16 峰值算力是 989.5 TFLOPS，这个指标表示训练用到了多少。

## 八、完整执行流程

把以上所有部分串起来，脚本的完整执行流程如下。

```
1. 设置环境变量、导入依赖
2. 加载 Flash Attention 3（Hopper GPU 专用加速）
3. 从 prepare.py 导入数据接口
4. 根据超参数构建 GPTConfig
5. 在 meta device 上创建模型 → 搬到 GPU → 初始化权重
6. 创建 MuonAdamW 优化器
7. torch.compile 编译模型
8. 创建数据加载器，预取第一个 batch
9. 进入训练循环（5 分钟时间预算）
   ├─ 前向传播 + 反向传播（带梯度累积）
   ├─ 更新学习率调度
   ├─ 优化器更新参数
   └─ 检查时间是否用完
10. 在验证集上计算 val_bpb
11. 打印最终指标
```

## 九、参考链接

- [Muon: An optimizer for hidden layers in neural networks — Keller Jordan](https://kellerjordan.github.io/posts/muon/)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1)

（完）
