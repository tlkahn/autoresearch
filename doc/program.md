`program.md` 是 autoresearch 项目的「实验规程」，它定义了一套让 LLM 智能体（Agent）自主进行神经网络研究的完整工作流——从环境搭建、实验执行，到结果记录和版本管理。

## 一、这个项目到底在做什么？

一句话概括：**让 AI 自己做 AI 研究。**

传统的机器学习研究流程是这样的：人类研究员提出假设，修改代码，跑实验，看结果，再提出新假设……这个循环可能持续数天甚至数周。autoresearch 的核心想法是——把这个循环完全交给 LLM 智能体来执行。

你可以把它想象成一个永不疲倦的实习研究员：它坐在一台带 GPU 的电脑前，不停地修改模型代码、跑训练、看结果、决定保留还是回滚，然后继续下一轮实验。人类研究员甚至可以去睡觉，第二天醒来查看一夜之间积累的实验记录。

整体流程如下。

```
              ┌─────────────────────────────────┐
              │         LLM 智能体 (Agent)        │
              │                                   │
              │  1. 提出实验想法                     │
              │  2. 修改 train.py                  │
              │  3. git commit                     │
              │  4. uv run train.py (5分钟)         │
              │  5. 检查 val_bpb                    │
              │  6. 改善了？保留 : 回滚               │
              │  7. 记录到 results.tsv              │
              │  8. 回到第 1 步                     │
              │                                   │
              └───────────┬───────────────────────┘
                          │ 永不停止
                          ▼
                    LOOP FOREVER
```

## 二、val_bpb 是什么？为什么要优化它？

**`val_bpb`** 是 validation bits per byte 的缩写，即「验证集上的每字节比特数」。它衡量的是语言模型对未见过的文本的预测能力——平均需要多少个比特来编码验证集中的每个字节。

打个比方：假设你在猜朋友接下来要说什么。如果你猜得很准，你只需要很少的信息来「确认」答案（比特数低）；如果你猜得很差，你就需要很多额外信息来纠正（比特数高）。`val_bpb` 越低，说明模型越「聪明」，预测能力越强。

为什么选择 bpb 而不是更常见的 perplexity？因为 bpb 不依赖分词器——无论你用什么 tokenizer，最终都归一化到字节级别，这使得不同实验之间的结果具有可比性。

## 三、实验的规则与约束

`program.md` 为智能体设定了清晰的边界。

**能做的事：**

（1）修改 `train.py`——这是唯一允许编辑的文件。模型架构、优化器、超参数、训练循环、batch size、模型大小……全部可以改。

**不能做的事：**

（1）不能修改 `prepare.py`——它包含固定的数据加载、分词器和评估函数，是实验的「标尺」。

（2）不能安装新的依赖包——只能使用 `pyproject.toml` 里已有的库。

（3）不能修改评估方式——`evaluate_bpb()` 函数是真理标准。

这些限制就像科学实验中的对照条件：数据集固定、评估方式固定、时间预算固定（5 分钟），唯一的变量就是 `train.py` 里的代码。

此外还有两条软约束。

（1）**显存（VRAM）**——允许适度增加，但不能暴涨。一个微小的 `val_bpb` 提升如果要多吃 10GB 显存，可能不值得。

（2）**简洁性原则**——同等效果下，代码越简单越好。删掉一段代码而指标不变？这算是胜利。加了 20 行 hack 只换来 0.001 的提升？可能不值得。

## 四、实验循环的完整流程

实验在一个独立的 git 分支上进行（如 `autoresearch/mar5`）。下面是单次实验的完整步骤。

```bash
# 1. 智能体修改 train.py（比如调整学习率）
# 2. 提交修改
git commit -am "increase matrix LR to 0.05"

# 3. 运行训练（输出重定向，避免刷屏）
uv run train.py > run.log 2>&1

# 4. 提取关键指标
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

上面命令中，训练输出被重定向到 `run.log`，这是因为训练过程会产生大量日志，如果直接输出到终端会「淹没」智能体的上下文窗口。

提取出 `val_bpb` 后，智能体做出决策。

```
如果 val_bpb 比之前低（更好）：
    → 保留这次 commit，记录 status = "keep"

如果 val_bpb 相同或更高（没改善）：
    → git reset 回滚到上一次的 commit，记录 status = "discard"

如果训练崩溃（OOM、bug 等）：
    → 检查错误日志，尝试修复或放弃，记录 status = "crash"
```

## 五、结果记录格式

每次实验的结果都记录到 `results.tsv` 文件中（注意是 Tab 分隔，不是逗号）。格式如下。

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

上面示例中，五列分别是：

（1）`commit`——git 提交的短哈希（7 位），用于追溯代码变更。

（2）`val_bpb`——验证集上的比特每字节，崩溃时记为 `0.000000`。

（3）`memory_gb`——峰值显存占用（GB），由 `peak_vram_mb / 1024` 得到。

（4）`status`——`keep`（保留）、`discard`（丢弃）或 `crash`（崩溃）。

（5）`description`——简短描述本次实验做了什么。

值得注意的是，`results.tsv` 本身不被 git 追踪——它只是智能体的「实验日记」，不参与版本管理。

## 六、搭建实验环境

在开始实验循环之前，需要完成一次性的环境搭建。

```bash
# 1. 创建实验分支
git checkout -b autoresearch/mar13

# 2. 确认数据已准备好
ls ~/.cache/autoresearch/
# 如果为空，需要先运行：
uv run prepare.py

# 3. 初始化结果记录文件
echo -e "commit\tval_bpb\tmemory_gb\tstatus\tdescription" > results.tsv

# 4. 运行基线实验（不做任何修改）
uv run train.py > run.log 2>&1
grep "^val_bpb:" run.log
```

上面步骤中，第 4 步是关键——必须先用原始代码跑一次基线（baseline），后续所有实验都和这个基线做对比。

## 七、关键设计决策

`program.md` 中有几个设计决策值得深入理解。

### 7.1 为什么用时间预算而不是 step 数？

固定 5 分钟的训练时间（而非固定训练步数），意味着智能体可以自由地调整 batch size 和模型大小。用更大的 batch 会减少步数但每步处理更多数据；用更小的模型会跑更多步但每步学得少。这迫使智能体在「模型容量」和「训练量」之间做权衡——这正是真实 ML 研究中最核心的问题之一。

### 7.2 为什么永不停止？

`program.md` 第 112 行明确写道：

> **NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue.

这个设计体现了完全自主的理念。一个典型场景是：用户晚上启动智能体，然后去睡觉。每次实验约 5 分钟，一小时能跑 12 次，8 小时睡眠期间可以完成大约 100 次实验。用户醒来后，查看 `results.tsv` 就能看到一整夜的研究成果。

### 7.3 版本控制作为实验管理

项目巧妙地利用 git 来管理实验状态：

- 每次修改都是一个 commit——可以精确追溯每个实验做了什么
- 改善了就保留 commit，分支向前推进
- 没改善就 `git reset`，回到上一个好的状态
- 实验分支与主分支隔离，不影响稳定代码

这种做法比传统的实验管理工具（如 MLflow、Weights & Biases）更轻量——不需要额外的服务器或账号，一切就在 git 历史里。

## 八、智能体的「研究策略」

虽然 `program.md` 没有规定智能体应该尝试什么，但从约束条件中可以推断出有效的策略方向。

（1）**超参数搜索**——最安全的起点：调整学习率、batch size、模型深度和宽度。

（2）**架构改进**——更有潜力但风险更高：尝试不同的注意力模式、激活函数、归一化方法。

（3）**优化器调优**——修改 Muon 或 AdamW 的内部参数，或尝试新的学习率调度策略。

（4）**做减法**——删除不必要的组件。如果去掉 Value Embedding 后 `val_bpb` 不变，那就是净赚——代码更简单了。

（5）**组合策略**——如果实验 A 提升了 0.003，实验 B 提升了 0.002，它们的组合也许能提升 0.004 或更多。

整个过程和人类研究员的思维模式完全一致——只是速度快了很多倍。

## 九、参考链接

- [autoresearch 项目代码仓库](https://github.com/) — 项目源码（本地仓库）
- [Bits per byte (BPB) 指标解释 — Hugging Face](https://huggingface.co/docs/transformers/perplexity) — 理解语言模型评估指标
- [Git 工作流基础 — Atlassian](https://www.atlassian.com/git/tutorials/comparing-workflows) — 理解分支管理策略
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361) — 理解模型大小与训练量的权衡

（完）
