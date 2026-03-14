# LLM 推理参数详解 / LLM Inference Parameters Guide

> 理解这些参数如何影响模型输出，是用好本地 LLM 的基础。
>
> Understanding how these parameters affect model output is fundamental to using local LLMs effectively.

---

## 1. Temperature（温度）

**作用：** 控制输出的随机性/创造性。

**What it does:** Controls the randomness/creativity of the output.

**原理：** 模型在生成每个 token 时，会对所有候选 token 计算概率分布。Temperature 通过缩放 logits（原始分数）来调整这个分布：

**How it works:** When generating each token, the model computes a probability distribution over all candidate tokens. Temperature scales the logits (raw scores) to reshape this distribution:

```
概率 / probability = softmax(logits / temperature)
```

- **temperature < 1**：概率分布变得更"尖锐"，高概率的 token 更突出 → 输出更确定、更保守
  The distribution becomes "sharper" — high-probability tokens dominate → more deterministic, conservative output
- **temperature = 1**：使用原始概率分布
  Uses the original probability distribution as-is
- **temperature > 1**：概率分布变得更"平坦"，低概率的 token 也有机会被选中 → 输出更随机、更有创意
  The distribution becomes "flatter" — low-probability tokens get a chance → more random, creative output

**取值范围 / Range：** 0 ~ 2（Ollama 默认 / default: 0.8）

| 场景 / Use Case | 建议值 / Suggested | 原因 / Why |
|------|--------|------|
| RAG / 问答 / 事实提取 | 0.1 - 0.3 | 需要准确，减少幻觉 / Accuracy needed, reduce hallucination |
| 通用对话 / General chat | 0.5 - 0.8 | 平衡准确性和自然度 / Balance accuracy and naturalness |
| 创意写作 / Creative writing | 0.7 - 1.2 | 需要多样性和创造力 / Diversity and creativity needed |
| 代码生成 / Code generation | 0.1 - 0.3 | 代码需要精确 / Code requires precision |

**示例对比 / Example comparison：**

```python
# temperature=0.1 → 每次输出几乎相同 / nearly identical output each time
"春天是万物复苏的季节。"
"春天是万物复苏的季节。"

# temperature=1.5 → 每次输出都不同 / different output each time
"春天像一首温柔的诗，唤醒沉睡的大地。"
"春风拂过田野，花朵竞相绽放。"
```

---

## 2. num_ctx（上下文窗口大小 / Context Window Size）

**作用：** 设置模型一次能处理的最大 token 数量，包括输入（prompt + 历史）和输出。

**What it does:** Sets the maximum number of tokens the model can process at once, including both input (prompt + history) and output.

**原理：** Transformer 模型的注意力机制需要对上下文中的所有 token 进行两两计算，内存占用与 num_ctx 的平方成正比：

**How it works:** The Transformer's attention mechanism computes pairwise relationships between all tokens in the context. Memory usage scales quadratically with num_ctx:

```
内存 / Memory ∝ num_ctx²
```

所以上下文越大，内存消耗增长非常快。

The larger the context, the faster memory consumption grows.

**Qwen3:8B 的情况 / Qwen3:8B specs：**

- 模型最大支持 / Model maximum：128K tokens
- Ollama 默认值 / Ollama default：2048
- 本地推荐值 / Local recommended：4096 - 8192（平衡效果和速度 / balancing quality and speed）

| num_ctx | 大约可容纳 / Approx. capacity | 适用场景 / Use case | 内存需求 / Memory |
|---------|-----------|---------|---------|
| 2048 | ~1500 字中文 / ~1500 EN words | 简短对话 / Short chat | 低 / Low |
| 4096 | ~3000 字中文 / ~3000 EN words | 一般问答 / General QA | 中 / Medium |
| 8192 | ~6000 字中文 / ~6000 EN words | RAG（含检索内容 / with retrieved docs） | 中高 / Medium-High |
| 32768 | ~24000 字中文 / ~24K EN words | 长文档分析 / Long document analysis | 高 / High |
| 131072 | ~100000 字中文 / ~100K EN words | 极长文本 / Very long text（需要大量 RAM / requires lots of RAM） | 极高 / Very High |

**注意事项 / Notes：**
- 超过 num_ctx 的内容会被截断（模型看不到）
  Content beyond num_ctx is truncated — the model cannot see it
- RAG 场景需要预留空间给检索到的文档，建议 8192+
  RAG tasks need room for retrieved documents — recommend 8192+
- 本地运行时，num_ctx 太大会显著降低推理速度
  Locally, a very large num_ctx significantly slows inference

```python
# Ollama Python 设置方式 / How to set in Ollama Python
ollama.chat(
    model="qwen3:8b",
    messages=[...],
    options={"num_ctx": 8192}
)
```

---

## 3. top_p（Nucleus Sampling / 核采样）

**作用：** 在模型生成下一个 token 之前，先过滤掉那些"不太可能"的候选词，只从"有可能"的词中做选择。

**What it does:** Before the model picks the next token, top_p filters out unlikely candidates and only samples from the "probable" ones.

### 为什么需要 top_p？/ Why do we need top_p?

想象模型在生成"今天天气真___"的下一个字。模型内部会给词表里每个词打分（概率）：

Imagine the model is generating the next character after "今天天气真___". Internally, it scores every token in the vocabulary:

```
"好"   = 0.40   ← 最可能 / most likely
"不"   = 0.20
"的"   = 0.15
"棒"   = 0.10
"热"   = 0.05
"冷"   = 0.03
"糟"   = 0.02
"妙"   = 0.01
"紫"   = 0.001  ← 几乎不可能 / almost impossible
"桌"   = 0.0005 ← 完全不合理 / makes no sense
...（词表里还有几万个词，概率都极低）
   (...tens of thousands more tokens with tiny probabilities)
```

如果不做过滤，模型有极小的概率选到"紫"或"桌"这种完全不合理的词，导致输出质量下降。

Without filtering, there's a tiny but non-zero chance the model picks nonsensical tokens like "紫" or "桌", degrading output quality.

**top_p 的作用就是画一条线：只保留概率累加到 p 的那些词，其余全部丢弃。**

**top_p draws a cutoff line: keep only the tokens whose cumulative probability reaches p, discard the rest.**

### 详细计算过程 / Step-by-step calculation

以 `top_p = 0.9` 为例 / Using `top_p = 0.9` as an example:

```
第一步：按概率从高到低排序
Step 1: Sort tokens by probability (high to low)

Token    概率/Prob    累积概率/Cumulative
─────    ────────    ──────────────────
"好"      0.40        0.40
"不"      0.20        0.60
"的"      0.15        0.75
"棒"      0.10        0.85
"热"      0.05        0.90  ← 到这里累积达到 0.90 ≥ top_p
                              cumulative reaches 0.90 ≥ top_p
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ← 以下全部丢弃 / everything below is discarded
"冷"      0.03        0.93
"糟"      0.02        0.95
"妙"      0.01        0.96
"紫"      0.001       ...
"桌"      0.0005      ...

第二步：只从前 5 个 token 中采样
Step 2: Sample only from the top 5 tokens

最终候选池 / Final candidate pool:
  {"好", "不", "的", "棒", "热"}

概率重新归一化 / Probabilities re-normalized:
  "好" = 0.40/0.90 = 0.444
  "不" = 0.20/0.90 = 0.222
  "的" = 0.15/0.90 = 0.167
  "棒" = 0.10/0.90 = 0.111
  "热" = 0.05/0.90 = 0.056
```

#### 为什么要归一化？/ Why re-normalize?

截断后剩余 token 的概率加起来只有 0.90，不是 1.0。但采样要求所有概率之和必须等于 1（"必定选中某一个"）。

After truncation, the remaining tokens' probabilities sum to only 0.90, not 1.0. But sampling requires all probabilities to sum to exactly 1 ("must pick one").

**类比 / Analogy：** 想象一个抽奖箱里有 10 个球，中签率总和 100%。你拿走了 5 个球（top_p 截断），剩下 5 个球的中签率只有 90%。那"剩下的 10% 给谁？"——按比例分给留下来的球，这就是归一化。

Imagine a lottery box with 10 balls, total win probability = 100%. You remove 5 balls (top_p cutoff), and the remaining 5 only sum to 90%. "Who gets the leftover 10%?" — redistribute proportionally to the remaining balls. That's normalization.

```
截断后（加起来 0.90，不是 1.0）       归一化后（加起来 1.0，可以采样）
After cutoff (sums to 0.90)          After normalization (sums to 1.0)

"好" = 0.40  ─── ÷ 0.90 ───→  "好" = 0.444
"不" = 0.20  ─── ÷ 0.90 ───→  "不" = 0.222
"的" = 0.15  ─── ÷ 0.90 ───→  "的" = 0.167
"棒" = 0.10  ─── ÷ 0.90 ───→  "棒" = 0.111
"热" = 0.05  ─── ÷ 0.90 ───→  "热" = 0.056
      ─────                          ─────
      0.900                          1.000 ✓
```

如果不归一化，采样时有 10% 的概率"落空"（选不到任何 token），程序会出错。归一化确保"一定能选到一个"。

Without normalization, there's a 10% chance of "landing on nothing" (no token selected), which would crash the program. Normalization guarantees "exactly one token is always picked".

### 不同 top_p 值的直觉 / Intuition for different top_p values

```
top_p = 0.4 → 只保留 {"好"} → 几乎等于贪心选择，永远选最高概率的
              Keep only {"好"} → almost greedy, always picks the top token

top_p = 0.6 → 保留 {"好", "不"} → 2 个候选，偶尔有变化
              Keep {"好","不"} → 2 candidates, slight variation

top_p = 0.9 → 保留 {"好","不","的","棒","热"} → 5 个候选，自然且多样
              Keep 5 candidates → natural and diverse (DEFAULT)

top_p = 1.0 → 保留全部（包括"紫""桌"）→ 不做任何过滤
              Keep all (including "紫","桌") → no filtering at all
```

### 核心直觉：top_p 是"自适应"的 / Core insight: top_p is adaptive

这是 top_p 最精妙的地方，也是它比 top_k 好的原因：

This is the most elegant aspect of top_p, and why it's better than top_k:

#### 场景 A：模型很确定 / Scenario A: Model is confident

```
"1 + 1 = ___"
  "2" = 0.95    ← top_p=0.9 时只保留这一个！自动变成贪心选择
  "二" = 0.03      With top_p=0.9, only this one survives! Automatically becomes greedy
  "两" = 0.01
  ...
→ 候选池自动缩小到 1 个 / Candidate pool auto-shrinks to 1
```

#### 场景 B：模型不确定 / Scenario B: Model is uncertain

```
"她穿了一件___色的裙子"
  "红" = 0.18
  "蓝" = 0.16
  "白" = 0.14    ← top_p=0.9 时保留 6-7 个词，都是合理的颜色
  "黑" = 0.13       With top_p=0.9, 6-7 tokens survive, all reasonable colors
  "粉" = 0.12
  "绿" = 0.10
  "黄" = 0.08
  ...
→ 候选池自动扩大 / Candidate pool auto-expands
```

**top_k=5 就做不到这种自适应**——它在场景 A 会多保留 4 个不需要的候选，在场景 B 可能又不够用。

**top_k=5 can't do this** — in Scenario A it keeps 4 unnecessary candidates, in Scenario B it might not keep enough.

### 取值范围 / Range

0 ~ 1（Ollama 默认 / default: 0.9）

| top_p 值 / Value | 效果 / Effect |
|----------|------|
| 0.1 - 0.3 | 非常保守，几乎等于贪心解码 / Very conservative, almost greedy decoding |
| 0.5 - 0.7 | 适度多样 / Moderate diversity |
| 0.9 | **推荐默认值**，过滤掉极不可能的 token / **Recommended default** — filters out very unlikely tokens |
| 1.0 | 不过滤（完全由 temperature 控制）/ No filtering (randomness controlled entirely by temperature) |

### 与 temperature 的关系 / Relationship with temperature

两者都控制随机性，但作用的阶段不同：

Both control randomness, but at different stages:

```
原始 logits → [Temperature 缩放] → 概率分布 → [top_p 截断] → 最终采样
Raw logits  → [Temperature scale] → Prob dist → [top_p cutoff] → Final sampling
```

- **Temperature**：改变概率分布的"形状"（尖锐 vs 平坦）
  Reshapes the distribution (sharp vs flat)
- **top_p**：在分布已经确定后，砍掉尾部的低概率候选
  After the distribution is set, chops off the long tail of low-probability candidates

**实际建议 / Practical advice：**

- 通常固定 `top_p=0.9`，只调 `temperature` 就够了
  Usually fix `top_p=0.9` and only tune `temperature`
- 同时大幅调两个参数效果难以预测
  Tuning both aggressively makes behavior hard to predict
- 如果你要极高精度（如 RAG），可以 `temperature=0.1, top_p=0.9`
  For maximum accuracy (e.g. RAG): `temperature=0.1, top_p=0.9`
- 如果你要极高创意，可以 `temperature=1.0, top_p=0.95`
  For maximum creativity: `temperature=1.0, top_p=0.95`

---

## 4. seed（随机种子 / Random Seed）

**作用：** 固定随机数生成器的初始状态，使输出可复现。

**What it does:** Fixes the random number generator's initial state so that output becomes reproducible.

**原理：** LLM 生成 token 时涉及随机采样。设置相同的 seed，采样过程完全一致，输出也就一致。

**How it works:** Token generation involves random sampling. With the same seed, the sampling process is identical, producing identical output.

```python
# 相同 seed → 相同输出 / Same seed → same output
ollama.chat(model="qwen3:8b", messages=[...], options={"seed": 42})
# → "春天是万物复苏的季节。"

ollama.chat(model="qwen3:8b", messages=[...], options={"seed": 42})
# → "春天是万物复苏的季节。"  （完全相同 / exactly the same）

# 不同 seed → 不同输出 / Different seed → different output
ollama.chat(model="qwen3:8b", messages=[...], options={"seed": 123})
# → "春天带来了温暖和生机。"
```

**适用场景 / When to use：**
- 调试和测试：确保每次运行结果一致
  Debugging & testing: ensure consistent results across runs
- A/B 对比：只改一个参数，固定其他变量
  A/B comparison: change one parameter while keeping others fixed
- 可复现的实验：论文/报告中需要重现结果
  Reproducible experiments: results that can be replicated for papers/reports

**注意 / Note：** 只有在 temperature > 0 时 seed 才有意义。temperature=0 时输出本身就是确定的。

seed only matters when temperature > 0. At temperature=0, output is already deterministic.

---

## 5. 其他常用参数 / Other Common Parameters

### top_k

限制只从概率最高的 K 个 token 中采样。

Limits sampling to only the top K most probable tokens.

```
top_k=10 → 只从前 10 个最可能的 token 中选择 / sample from top 10 tokens only
top_k=50 → 从前 50 个中选择 / sample from top 50（Ollama 默认 / default: 40）
```

与 top_p 的区别：top_k 固定数量，top_p 固定概率阈值。top_p 更灵活（在确定场景自动收窄，在不确定场景自动放宽）。

Difference from top_p: top_k fixes the count, top_p fixes the probability threshold. top_p is more adaptive — it narrows automatically in confident scenarios and widens in uncertain ones.

### repeat_penalty（重复惩罚 / Repetition Penalty）

降低已经出现过的 token 再次被选中的概率，防止模型"复读"。

Reduces the probability of tokens that have already appeared, preventing the model from repeating itself.

```
repeat_penalty=1.0 → 不惩罚（可能出现重复）/ No penalty (repetition possible)
repeat_penalty=1.1 → 轻微惩罚 / Mild penalty（Ollama 默认 / default）
repeat_penalty=1.5 → 较强惩罚 / Strong penalty
```

### num_predict（最大生成长度 / Max Generation Length）

限制模型最多生成多少个 token。

Limits the maximum number of tokens the model can generate.

```
num_predict=256  → 短回答 / Short answer
num_predict=1024 → 中等长度 / Medium length
num_predict=-1   → 不限制（直到模型自行停止）/ Unlimited (until model stops naturally)
```

---

## 6. 参数组合建议 / Recommended Parameter Combinations

| 任务类型 / Task | temperature | top_p | num_ctx | 其他 / Other |
|---------|-------------|-------|---------|------|
| RAG 问答 / RAG QA | 0.1 | 0.9 | 8192 | - |
| 代码生成 / Code gen | 0.2 | 0.9 | 4096 | - |
| 通用对话 / General chat | 0.7 | 0.9 | 4096 | - |
| 创意写作 / Creative writing | 1.0 | 0.95 | 4096 | repeat_penalty=1.2 |
| 实验/调试 / Experiment/Debug | 0.5 | 0.9 | 4096 | seed=42 |

```python
# RAG 场景的推荐配置 / Recommended config for RAG
ollama.chat(
    model="qwen3:8b",
    messages=[...],
    options={
        "temperature": 0.1,
        "top_p": 0.9,
        "num_ctx": 8192,
    }
)
```
