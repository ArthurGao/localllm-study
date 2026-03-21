# KV Cache 深度解析
## LLM 推理加速的核心技术

> 本文档配合 Claude Code 生成学习代码。每节末尾附有精准的代码生成提示词。

---

## 目录

1. [KV Cache 是什么：从零理解](#1-kv-cache-是什么从零理解)
2. [KV Cache 的工作原理](#2-kv-cache-的工作原理)
3. [PagedAttention：内存管理革命](#3-pagedattention内存管理革命)
4. [Prefix Caching：前缀复用](#4-prefix-caching前缀复用)
5. [RadixAttention：SGLang 的树形缓存](#5-radixattention-sglang-的树形缓存)
6. [KV Cache 分层存储](#6-kv-cache-分层存储)
7. [主流产品横评](#7-主流产品横评)
8. [实战：vLLM KV Cache 配置](#8-实战vllm-kv-cache-配置)
9. [实战：SGLang RadixAttention](#9-实战sglang-radixattention)
10. [实战：LMCache 分布式 KV Cache](#10-实战lmcache-分布式-kv-cache)
11. [API 层 Prompt Caching（Anthropic / OpenAI）](#11-api-层-prompt-cachinganthropic--openai)
12. [选型指南](#12-选型指南)

---

## 1. KV Cache 是什么：从零理解

### Transformer 中的注意力机制回顾

LLM 生成文字时，每生成一个 token，都要对**所有已有 token** 做注意力计算（Attention）：

```
生成 "我爱北京天安门" 的过程：

生成 "我"   → Attention([])          → "我"
生成 "爱"   → Attention(["我"])       → "爱"
生成 "北"   → Attention(["我","爱"])  → "北"
生成 "京"   → Attention(["我","爱","北"]) → "京"
...
```

Attention 计算的核心是 **Q（Query）、K（Key）、V（Value）** 三个矩阵。

### 没有 KV Cache 的问题

```
没有 KV Cache：

生成第 N 个 token 时：
  对 token 1..N-1 重新计算 K 和 V
  计算量 = O(N²)
  每步都在做重复计算！

  生成 1000 token 的文字：
  token 500 要重新计算前 499 个 token 的 K/V
  token 999 要重新计算前 998 个 token 的 K/V
  → 极其浪费！
```

### KV Cache 的本质

**把每个 token 的 K 和 V 矩阵缓存起来，下次不再重新计算。**

```
有 KV Cache：

生成第 N 个 token 时：
  从缓存读取 token 1..N-1 的 K/V（已计算好）
  只需计算当前 token 的 Q
  和缓存的所有 K/V 做 Attention

  生成量 = O(N)（仅当前 token）
  缓存读取 = O(N)（从内存读）

  → 避免重复计算，速度大幅提升！
```

### KV Cache 的代价

```
优点：减少计算量，加速生成（decode 阶段）
缺点：占用 GPU 显存

一个典型的例子（Llama-3-8B，FP16）：
  每个 token 的 KV Cache 大小
  = 2（K+V）× 层数（32）× 头数（8）× 头维度（128）× 2字节
  ≈ 128 KB / token

  处理 4096 token 的 context：
  4096 × 128 KB = 512 MB（仅 KV Cache！）

  → 长 context + 多并发 = 显存爆炸问题
```

---

## 2. KV Cache 的工作原理

### Prefill 阶段 vs Decode 阶段

```
LLM 推理分两个阶段：

┌─────────────────────────────────────────────────────┐
│  Prefill 阶段（输入处理）                            │
│                                                     │
│  输入：整个 prompt（可能几千 token）                  │
│  操作：并行计算所有输入 token 的 K/V                  │
│  输出：KV Cache + 第一个输出 token                  │
│  特点：计算密集，GPU 满负荷（compute-bound）         │
└─────────────────────────────────────────────────────┘
         │
         │ 保存 KV Cache
         ▼
┌─────────────────────────────────────────────────────┐
│  Decode 阶段（逐 token 生成）                        │
│                                                     │
│  输入：上一个生成的 token                            │
│  操作：读取 KV Cache + 计算新 token 的 Attention     │
│  输出：下一个 token（循环直到 EOS）                  │
│  特点：内存密集，受显存带宽限制（memory-bound）       │
└─────────────────────────────────────────────────────┘
```

### KV Cache 的内存布局（无优化版）

```
早期实现：每个请求分配连续显存

请求 A（512 tokens）: [KKKK...VVVV...] ← 连续显存块
请求 B（256 tokens）: [KKKK...VVVV...] ← 连续显存块
请求 C（1024 tokens）:[KKKK...VVVV...] ← 连续显存块

问题：
1. 碎片化：请求结束后留下不规则空洞
2. 预分配：必须提前按最大长度分配（浪费）
3. 并发差：大请求占用整块显存，小请求等待
```

---

## 3. PagedAttention：内存管理革命

PagedAttention 是 vLLM 提出的核心创新，借鉴操作系统的**虚拟内存分页**思想管理 KV Cache。

### 核心思想

```
操作系统内存分页 → 启发 → PagedAttention

OS 内存管理：
  物理内存切成固定大小的"页"（4KB）
  进程看到的是连续的"虚拟地址"
  实际映射到不连续的物理页
  → 消除碎片，提高利用率

PagedAttention：
  GPU 显存切成固定大小的"KV Block"（如 16 tokens）
  每个请求的 KV Cache 由一组不连续的 Block 组成
  Block Table 记录每个请求拥有哪些 Block
  → 消除显存碎片，提高并发
```

### PagedAttention 内存布局

```
GPU 显存（切成 KV Block，每块 = 16 tokens 的 KV）

物理 Block 池：
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  B0  │  B1  │  B2  │  B3  │  B4  │  B5  │  B6  │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘

请求 A 的逻辑视图（32 tokens，需要 2 个 Block）：
  逻辑 Block 0 → 物理 Block B1
  逻辑 Block 1 → 物理 Block B4

请求 B 的逻辑视图（48 tokens，需要 3 个 Block）：
  逻辑 Block 0 → 物理 Block B0
  逻辑 Block 1 → 物理 Block B3
  逻辑 Block 2 → 物理 Block B6

Block Table（类比操作系统页表）：
  请求 A: [1, 4]
  请求 B: [0, 3, 6]
```

### PagedAttention 的优势

```
1. 零碎片（Near-zero fragmentation）
   传统方式：内存利用率 ~60-80%（碎片浪费）
   PagedAttention：内存利用率 >95%（只有最后一个 Block 可能浪费）

2. Copy-on-Write（写时复制）
   Beam Search 中多条候选序列共享前缀：
   前缀部分的 KV Block 引用计数 +1，不复制
   分叉时才复制（类似 Unix fork()）

3. 提升并发
   vLLM 实测：相比 HuggingFace Transformers 提升 14-24x 吞吐量
```

### 代码概念示意

```python
# PagedAttention 的逻辑（概念示意，非真实 vLLM 代码）

class BlockTable:
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.physical_blocks = {}   # 物理 block 池
        self.table = {}             # request_id -> [block_ids]

    def allocate(self, request_id: str, num_tokens: int):
        """为请求分配 KV Block"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        blocks = self._get_free_blocks(num_blocks)
        self.table[request_id] = blocks
        return blocks

    def get_kv_cache(self, request_id: str, layer_idx: int):
        """获取请求在某层的 KV Cache（逻辑连续，物理不连续）"""
        block_ids = self.table[request_id]
        kv_blocks = [self.physical_blocks[(bid, layer_idx)] for bid in block_ids]
        return kv_blocks  # Attention kernel 处理不连续的 blocks
```

---

> **💡 Claude Code 提示词 — PagedAttention 模拟：**
> ```
> 用 Python 实现一个 PagedAttention 内存管理器的模拟（不需要真实 GPU，纯逻辑模拟）：
>
> 1. KVBlock 类：
>    - block_id, block_size（默认16），ref_count
>    - 存储 key_cache 和 value_cache（用 numpy array 模拟）
>
> 2. BlockAllocator 类：
>    - 初始化：total_blocks 个物理 block
>    - allocate(n) → 分配 n 个 block，返回 block_id 列表
>    - free(block_ids) → 释放（ref_count - 1，到 0 则回收）
>    - fork(block_ids) → Copy-on-Write，ref_count + 1（Beam Search 用）
>    - get_stats() → 返回 {total, used, free, fragmentation_rate}
>
> 3. RequestMemoryManager 类：
>    - 管理多个并发请求的 BlockTable
>    - add_tokens(request_id, n_tokens) → 动态扩展 Block
>    - finish_request(request_id) → 释放所有 Block
>
> 4. 演示场景（打印内存使用变化）：
>    - 同时处理 5 个不同长度的请求（100-500 tokens）
>    - 请求 3 使用 Beam Search（fork 演示 Copy-on-Write）
>    - 请求 1 完成后观察内存释放和复用
>    - 打印每步的内存使用统计
>
> 5. 对比实验：
>    - 同样的请求集合，用"连续内存分配"方式模拟
>    - 对比两种方式的内存利用率和碎片率
> ```

---

## 4. Prefix Caching：前缀复用

Prefix Caching（前缀缓存）是在 PagedAttention 基础上的进一步优化：**相同前缀的请求共享 KV Cache Block，避免重复计算。**

### 核心场景

```
场景 1：System Prompt 复用
  所有请求都有相同的 system prompt（如 "你是一个支付系统专家..."）

  请求 A: [system_prompt] + "查询今天的失败交易"
  请求 B: [system_prompt] + "分析 AM04 错误原因"
  请求 C: [system_prompt] + "生成月度报告"

  Without Prefix Cache: 每个请求都重新计算 system_prompt 的 KV
  With Prefix Cache:     system_prompt 的 KV 只计算一次，3 个请求共享

场景 2：Few-shot 示例复用
  RAG 场景：每个请求都携带相同的示例文档

  请求 A: [Document 1-5] + "问题 A"
  请求 B: [Document 1-5] + "问题 B"

  With Prefix Cache: Document 1-5 的 KV 只算一次

场景 3：多轮对话
  对话历史越来越长，但前几轮是共享的

  轮1结束: cache([sys] + [轮1])
  轮2结束: cache([sys] + [轮1] + [轮2])
  轮3开始: 直接命中 [sys] + [轮1] + [轮2] 的 cache
```

### 缓存 Key 的设计

```
如何判断两段 prompt 前缀是否相同？
→ 对 token 序列做哈希

Block Hash = hash(token_ids_in_block)

示例：
  System Prompt（128 tokens）= 8 个 Block
  Block 0 hash = hash(tokens[0:16])   = "abc123"
  Block 1 hash = hash(tokens[16:32])  = "def456"
  ...

  新请求来了：
  计算其前缀的 Block Hash
  在缓存中查找 → 命中 → 直接复用，跳过 prefill！
```

### vLLM 的 Automatic Prefix Caching（APC）

```python
from vllm import LLM, SamplingParams

# 开启 APC（vLLM 0.4.0+ 默认开启）
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    enable_prefix_caching=True,    # 开启前缀缓存
    max_model_len=8192,
)

# 第一个请求：system prompt + 问题 A
# → 完整 prefill，缓存 system prompt 的 KV
outputs_1 = llm.generate([{
    "prompt": "[system]你是支付专家[/system]\n用户：查询 AM04 错误"
}])

# 第二个请求：相同 system prompt + 问题 B
# → system prompt 部分直接命中缓存，只 prefill 问题 B！
outputs_2 = llm.generate([{
    "prompt": "[system]你是支付专家[/system]\n用户：分析 TTP 流程"
}])

# 查看缓存命中率
stats = llm.get_cache_stats()
print(f"Cache hit rate: {stats['prefix_cache_hit_rate']:.2%}")
```

### Prefix Caching 的局限

```
问题 1：前缀必须完全匹配（token 级别）
  "你是支付专家。" ≠ "你是支付专家！"（末尾标点不同 → 缓存失效）
  → 设计 prompt 时要保持前缀严格一致

问题 2：缓存会被驱逐
  GPU 显存有限，LRU 策略驱逐旧缓存
  → 热点 prompt 需要保持活跃，或配合持久化存储

问题 3：Hash 碰撞（极低概率）
  不同 token 序列碰巧有相同 hash → 错误命中
  → 实际影响极小，可忽略
```

---

> **💡 Claude Code 提示词 — Prefix Caching 模拟：**
> ```
> 用 Python 实现 Prefix Caching 缓存系统的模拟：
>
> 1. KVCacheStore 类（基于 Block Hash）：
>    - block_size: int = 16（每个 Block 的 token 数）
>    - cache: Dict[str, Block]（hash → KV Block）
>    - lru_order: deque（LRU 驱逐队列）
>    - max_blocks: int（最大缓存 Block 数）
>
>    方法：
>    - compute_hash(tokens: List[int]) → str（对 token 序列哈希）
>    - lookup_prefix(tokens: List[int]) → int（返回命中的 token 数）
>    - store(tokens: List[int], kv_data: np.array)（存储 KV）
>    - evict_lru()（LRU 驱逐最旧的 Block）
>
> 2. PrefixCachingSimulator 类：
>    - 模拟请求处理
>    - 计算每个请求的 prefill 节省比例
>    - 跟踪 cache hit rate、eviction count、memory usage
>
> 3. 测试场景（用随机 token id 模拟）：
>    场景 A：100 个请求，都有相同的 512 token system prompt
>    场景 B：多轮对话，每轮在前轮历史上追加 100 token
>    场景 C：RAG 查询，每次携带相同的 5 个文档（2000 token）+ 不同问题
>
> 4. 输出报告：
>    - 每个场景的 cache hit rate
>    - 节省的计算量（命中 token 数 / 总 token 数）
>    - 缓存大小随时间变化的曲线（用 matplotlib 画图）
>
> 5. 对比实验：有/无 Prefix Cache 的总处理时间对比
> ```

---

## 5. RadixAttention：SGLang 的树形缓存

SGLang 的 RadixAttention 将 Prefix Caching 升级为**基于 Radix Tree（基数树）的自动缓存**，无需手动管理，能处理更复杂的共享前缀模式。

### Radix Tree 数据结构

```
Radix Tree（基数树/前缀树）：
  共享相同前缀的序列合并存储

例子：有 4 个请求：
  A: "你好，我叫 Alice" + "问题 A"
  B: "你好，我叫 Alice" + "问题 B"
  C: "你好，我叫 Alice" + "问题 C"
  D: "你好，我叫 Bob"   + "问题 D"

Radix Tree 结构：
         根节点
           │
    [你好，我叫]
           │
      ┌────┴────┐
  [Alice]    [Bob]
      │          │
  ┌───┼───┐    [问题D]
[问A][问B][问C]

KV Cache 存储：
  [你好，我叫] 的 KV → 存一份，4 个请求共享
  [Alice] 的 KV    → 存一份，A/B/C 共享
  [Bob] 的 KV      → 存一份，D 独享
  各问题的 KV      → 各自独立
```

### RadixAttention vs APC 对比

```
vLLM APC（Automatic Prefix Caching）：
  - 基于 Block 级别的 Hash 匹配
  - 需要前缀严格对齐到 Block 边界
  - 静态匹配，提前知道共享模式效果最好

SGLang RadixAttention：
  - 基于 Radix Tree，任意长度前缀都能共享
  - 动态发现共享前缀，无需手动对齐
  - 自动处理多轮对话的渐进式缓存增长
  - 在 agents/tool calls 等非规则模式中优势明显
```

### RadixAttention 实际效果

```
场景：multi-turn chat，每轮历史增长

请求序列（同一用户的多轮对话）：
轮1: [sys]                 + [Q1]      → 完整 prefill
轮2: [sys] + [Q1] + [A1]  + [Q2]      → sys+Q1+A1 命中缓存
轮3: [sys] + [Q1..Q2+A2]  + [Q3]      → 前两轮全命中
...
轮N: 只需 prefill 最新的 Q，其余全部命中

SGLang 论文数据：
  在 multi-turn 场景，相比 vLLM 延迟降低 3.7×
  吞吐量提升 6.4×（heavily cached 场景）
```

---

> **💡 Claude Code 提示词 — Radix Tree KV Cache：**
> ```
> 用 Python 实现 RadixAttention 的 Radix Tree KV Cache 模拟：
>
> 1. RadixTreeNode 类：
>    - tokens: List[int]（本节点存储的 token 序列片段）
>    - children: Dict[int, RadixTreeNode]（子节点，key 是首个 token）
>    - kv_cache: Optional[np.array]（该节点的 KV Cache 数据）
>    - ref_count: int（引用计数）
>    - last_access_time: float（LRU 用）
>
> 2. RadixTree 类：
>    - insert(tokens, kv_data) → 插入新序列的 KV Cache
>      自动找到最长公共前缀，分裂节点，存储新 KV
>    - match_prefix(tokens) → (matched_len, kv_list)
>      返回最长匹配前缀的长度和对应 KV Cache 列表
>    - evict_lru(n_blocks) → 驱逐最少使用的叶节点
>    - visualize() → 打印树结构（ASCII 图）
>
> 3. 场景模拟（3种）：
>    场景1：多轮对话（10 轮，每轮追加 50 token）
>      → 展示缓存命中如何随轮数增长
>    场景2：Few-shot（相同5个示例 + 不同问题，100 个请求）
>      → 统计示例部分的命中率
>    场景3：对比 vLLM Block Hash vs RadixTree
>      → 构造一个 RadixTree 能命中但 Block Hash 无法命中的案例
>      （前缀长度不是 block_size 整数倍的情况）
>
> 4. 可视化输出：
>    - 每次 insert/match 后打印树结构
>    - 最终输出：总命中率、节省计算量、树的节点数统计
> ```

---

## 6. KV Cache 分层存储

随着 context 越来越长（128K、1M token），单纯靠 GPU 显存存 KV Cache 已经不够了。分层存储（Hierarchical KV Cache）应运而生。

### 存储层次结构

```
速度快 ←─────────────────────────────────→ 容量大
成本高                                       成本低

┌──────────────┐
│  GPU HBM     │  速度：~3 TB/s    容量：~80 GB（H100）
│ （热缓存）    │  → 当前活跃请求的 KV Cache
└──────┬───────┘
       │ 溢出
       ▼
┌──────────────┐
│  CPU DRAM    │  速度：~100 GB/s  容量：~1 TB
│ （温缓存）    │  → 最近使用的 KV Cache（PCIe 传输）
└──────┬───────┘
       │ 溢出
       ▼
┌──────────────┐
│  NVMe SSD    │  速度：~10 GB/s   容量：~100 TB
│ （冷缓存）    │  → 历史对话 KV Cache
└──────┬───────┘
       │ 远程存储
       ▼
┌──────────────┐
│  Redis/S3    │  速度：~1 GB/s    容量：无限
│ （持久化）    │  → 跨节点共享、持久化 KV Cache
└──────────────┘
```

### LMCache：企业级分层 KV Cache

LMCache 是目前最完善的分层 KV Cache 解决方案，被 vLLM、SGLang、NVIDIA Dynamo、llm-d、KServe 等主流框架采用。

```
LMCache 架构：

推理引擎（vLLM / SGLang）
    │  KV Connector 接口
    ▼
LMCache Layer
    ├── GPU Cache（本地）
    ├── CPU Cache（本地 offload）
    ├── Disk Cache（本地 NVMe）
    └── Remote Cache（Redis / 分布式存储）
         └── 跨节点共享（多个 GPU 服务器共享同一个 KV Cache 池）
```

### PD Disaggregation（预填充-解码分离）

```
传统部署（Prefill 和 Decode 在同一 GPU）：
  GPU 同时处理 prefill（计算密集）和 decode（内存密集）
  → 资源竞争，利用率低

PD Disaggregation（预填充-解码分离）：
  Prefill GPU：专门做 prefill（高算力 GPU，如 H100）
  Decode GPU：专门做 decode（高带宽 GPU，或更多小 GPU）
  KV Cache 通过高速网络（RDMA/InfiniBand）从 Prefill 传输到 Decode

┌───────────────┐    KV Cache 传输    ┌───────────────┐
│  Prefill GPU  │ ─────────────────► │  Decode GPU   │
│  H100 x 4     │    RDMA / IB       │  A100 x 8     │
│  高算力       │                    │  高带宽/低成本 │
└───────────────┘                    └───────────────┘

优势：
  Prefill 吞吐量提升 2-3×（专注计算）
  Decode 成本降低（可用更多便宜 GPU 分摊）
```

---

> **💡 Claude Code 提示词 — 分层 KV Cache 模拟：**
> ```
> 用 Python 模拟分层 KV Cache 存储系统：
>
> 1. 定义 3 个存储层（用 dict + 访问延迟模拟）：
>    - GPUCache:  容量 100 blocks，访问延迟 0.1ms，LRU 驱逐
>    - CPUCache:  容量 500 blocks，访问延迟 1ms，LRU 驱逐
>    - DiskCache: 容量 2000 blocks，访问延迟 10ms
>
> 2. HierarchicalKVCache 类：
>    - get(token_hash) → 从最快层开始查找，找到则提升到 GPU 层
>    - put(token_hash, kv_data) → 存入 GPU 层，满了驱逐到 CPU，CPU 满了驱逐到 Disk
>    - get_stats() → 每层的 hit_count, miss_count, hit_rate
>    - simulate_latency() → 根据命中层返回模拟延迟
>
> 3. 工作负载模拟（Zipf 分布，模拟热点访问）：
>    - 生成 1000 个 token 序列
>    - 按 Zipf 分布（少数热点被频繁访问）模拟 10000 次请求
>    - 统计每层的命中率
>
> 4. 对比实验：
>    - 只有 GPU Cache（无分层）
>    - GPU + CPU 两层
>    - GPU + CPU + Disk 三层
>    - 对比总平均延迟和缓存命中率
>
> 5. 用 matplotlib 画：
>    - 每层命中率随请求数的变化曲线
>    - 三种配置的平均延迟对比柱状图
> ```

---

## 7. 主流产品横评

### 产品全景图

```
按定位分类：

推理引擎（内置 KV Cache）：
  ├── vLLM        ← 最流行，PagedAttention + APC
  ├── SGLang      ← 高性能，RadixAttention
  ├── TGI         ← HuggingFace 出品，对话友好
  └── Ollama      ← 本地开发，极易上手

KV Cache 增强层（插件式）：
  └── LMCache     ← 分层存储 + 分布式共享 + PD 分离

分布式 KV Cache 传输：
  └── Mooncake    ← 月之暗面出品，RDMA 高速传输

云 API（托管 Prompt Cache）：
  ├── Anthropic   ← Claude Prompt Caching（5 分钟 TTL）
  └── OpenAI      ← GPT-4 Automatic Caching（1 小时 TTL）

生产部署平台：
  └── llm-d       ← Red Hat，K8s 原生，分层 KV offloading
```

### 详细对比表

| 产品 | KV Cache 技术 | 适用场景 | 部署难度 | 开源 |
|------|--------------|---------|---------|------|
| **vLLM** | PagedAttention + APC（FP8 量化）| 通用高吞吐 | ⭐⭐ | ✅ |
| **SGLang** | RadixAttention | Agent / 多轮对话 | ⭐⭐ | ✅ |
| **TGI v3** | Paged Attention + Prefix Cache | 长对话 HuggingFace 生态 | ⭐⭐ | ✅ |
| **Ollama** | 基础 KV Cache | 本地开发/原型 | ⭐ | ✅ |
| **LMCache** | 分层存储（GPU/CPU/Disk/Redis）| 企业级，长上下文 | ⭐⭐⭐ | ✅ |
| **Mooncake** | RDMA 传输引擎，PD 分离 | 大规模分布式部署 | ⭐⭐⭐⭐ | ✅ |
| **Anthropic API** | Prompt Caching（5 min TTL）| 直接调 API | ⭐ | ❌（托管）|
| **OpenAI API** | Auto Caching（1 hour TTL）| 直接调 API | ⭐ | ❌（托管）|

### 核心指标对比

```
吞吐量（tokens/sec，8B 模型，单 A100）：

Ollama:      ~500 tok/s   （基础实现）
HF Transform: ~800 tok/s
TGI v3:      ~3000 tok/s  （长对话场景，prefix cache）
vLLM:        ~4000 tok/s  （PagedAttention + APC）
SGLang:      ~5000+ tok/s （RadixAttention，高缓存命中）

延迟（TTFT，Time to First Token，1K token prompt）：

无缓存：   500ms
vLLM APC 命中：  50ms  （10× 加速）
SGLang RadixAttention：30ms（更细粒度命中）
```

---

## 8. 实战：vLLM KV Cache 配置

### 安装与基础使用

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import init_distributed_environment

# 基础启动（自动开启 PagedAttention + APC）
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",

    # KV Cache 相关配置
    kv_cache_dtype="fp8",              # FP8 量化，节省 50% 显存
    enable_prefix_caching=True,        # 开启 APC（默认已开启）
    max_model_len=32768,               # 最大 context 长度
    gpu_memory_utilization=0.9,        # GPU 显存使用率（留 10% 给激活值）
    swap_space=4,                      # CPU swap 空间（GB）

    # 吞吐量优化
    max_num_batched_tokens=32768,      # 每批最大 token 数
    max_num_seqs=256,                  # 最大并发请求数
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

# 利用 Prefix Cache：把 system prompt 放在最前面，保持严格一致
SYSTEM_PROMPT = """你是一个专业的支付系统助手，熟悉 ISO 20022 标准、
Finacle API、IBM MQ 消息处理，以及 FTH 支付流程。"""

def query(user_question: str) -> str:
    prompt = f"<|system|>{SYSTEM_PROMPT}<|user|>{user_question}<|assistant|>"
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

# 第一个请求：完整 prefill system_prompt
ans1 = query("解释 AM04 错误码")

# 第二个请求：system_prompt 命中缓存，只 prefill 问题部分
ans2 = query("解释 AC01 错误码")
```

### vLLM OpenAI 兼容 Server 启动

```bash
# 启动 vLLM Server（OpenAI 兼容 API）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# 用 OpenAI 客户端调用
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "什么是 ISO 20022？"}
    ]
)
```

### 监控 KV Cache 使用

```python
# 方式 1：通过 /metrics endpoint（Prometheus 格式）
import requests
metrics = requests.get("http://localhost:8000/metrics").text
# 关注指标：
# vllm:gpu_cache_usage_perc     GPU KV Cache 使用率
# vllm:prefix_cache_hit_rate    前缀缓存命中率
# vllm:num_running_seqs         当前运行请求数

# 方式 2：通过 Engine stats（离线推理）
from vllm.engine.metrics import Stats
stats = llm.stat_loggers
```

---

> **💡 Claude Code 提示词 — vLLM 实战：**
> ```
> 用 Python + vLLM 实现一个展示 KV Cache 效果的完整实验：
>
> 1. 环境（如果没有 GPU，用 mock 模式模拟延迟）：
>    - 尝试 import vllm，失败则用 openai 客户端连接远程 vLLM
>    - 或用 time.sleep 模拟 prefill 延迟对比
>
> 2. 实验设计：
>    - 定义一个 500 token 的 system_prompt（用 lorem ipsum 中文填充）
>    - 准备 20 个不同的 user_question
>    - 测量每个请求的 TTFT（Time to First Token）
>
> 3. 对比实验：
>    - 组 A：每次请求都用不同的前缀（无缓存命中）
>    - 组 B：每次请求都保持相同的 system_prompt（缓存命中）
>    - 分别测量 20 个请求的平均 TTFT 和总耗时
>
> 4. 结果可视化（matplotlib）：
>    - 折线图：请求序号 vs TTFT（展示第 2 个请求开始的加速）
>    - 柱状图：有缓存 vs 无缓存的平均 TTFT 对比
>    - 打印：节省的计算量百分比
>
> 5. 最佳实践演示：
>    - 展示 system_prompt 的正确放置方式（前缀一致性）
>    - 展示一个"破坏缓存"的错误方式（在 system 中加时间戳）
>    - 对比两者的命中率差异
> ```

---

## 9. 实战：SGLang RadixAttention

### 安装与配置

```bash
pip install sglang[all]
```

```python
import sglang as sgl

# 启动 SGLang Runtime（自动开启 RadixAttention）
runtime = sgl.Runtime(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    mem_fraction_static=0.88,     # 静态 KV Cache 显存比例
    max_prefill_tokens=16384,     # 最大 prefill token 数
    # RadixAttention 默认开启，无需额外配置
)
sgl.set_default_backend(runtime)

# SGLang 的 @sgl.function 装饰器：自动管理 KV Cache
@sgl.function
def answer_question(s, system_prompt, question):
    s += sgl.system(system_prompt)
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_new_tokens=256))

# 批量请求（RadixAttention 自动发现共享前缀）
SYSTEM = "你是一个金融支付专家，熟悉国际汇款流程。"
questions = [
    "什么是 TTP 支付？",
    "什么是 IMT 支付？",
    "AM04 和 AC01 有什么区别？",
    "ISO 20022 的 MsgId 字段有什么要求？",
]

# 批量执行（自动命中 system prompt 的 KV Cache）
states = answer_question.run_batch(
    [{"system_prompt": SYSTEM, "question": q} for q in questions],
    temperature=0.7,
    num_threads=4,
)

for state, q in zip(states, questions):
    print(f"Q: {q}\nA: {state['answer']}\n")

# 查看 RadixAttention 统计
stats = runtime.get_server_info()
print(f"Radix cache hit tokens: {stats['radix_cache_hit_tokens']}")
print(f"Total input tokens:     {stats['total_input_tokens']}")
```

### SGLang 的结构化生成 + KV Cache

SGLang 特别适合 Agent 场景，因为结构化生成（如固定格式的 tool calls）能极大提高 KV Cache 命中率：

```python
@sgl.function
def run_tool_agent(s, task, tools_description):
    # tools_description 每次请求都相同 → 自动缓存
    s += sgl.system(f"""你是一个工具调用 Agent。
可用工具：{tools_description}
输出格式必须是 JSON。""")

    s += sgl.user(task)

    # 固定前缀 '{"tool":' 也能被 RadixAttention 缓存
    s += sgl.assistant('{"tool": ' + sgl.gen(
        "tool_call",
        max_new_tokens=256,
        stop="}",
        regex=r'"[a-z_]+", "args": \{.*\}'  # 约束输出格式
    ) + "}")
```

---

> **💡 Claude Code 提示词 — SGLang RadixAttention 实验：**
> ```
> 用 Python + SGLang 实现 RadixAttention 缓存效果演示：
>
> （如无 GPU 环境，用 requests 连接远程 SGLang 或模拟延迟）
>
> 1. 设置实验：
>    - 定义 5 个不同长度的 system_prompt（100/200/500/1000/2000 token）
>    - 每个 system_prompt 对应 10 个不同问题
>
> 2. 多轮对话场景（测试 RadixAttention 的核心优势）：
>    - 模拟同一用户进行 10 轮对话
>    - 每轮在历史基础上追加
>    - 测量每轮的 TTFT
>    - 预期：TTFT 保持稳定（而非随对话轮数增加而变慢）
>
> 3. 对比 vLLM APC vs SGLang RadixAttention：
>    - 构造一个"非 block 对齐"的共享前缀场景
>    （前缀长度 = 24 token，block_size = 16，有 8 token 无法命中 Block Hash）
>    - 展示 APC 的命中率 vs RadixAttention 的命中率
>    - 理论计算两者的命中 token 数差异
>
> 4. Agent 工具调用场景：
>    - 定义一个固定的 tools_description（300 token）
>    - 100 个不同的任务描述
>    - 统计 tools_description 部分的缓存节省
>
> 5. 输出：
>    - 每个场景的 cache hit rate
>    - 延迟分布直方图（第 1 次 vs 第 N 次）
> ```

---

## 10. 实战：LMCache 分布式 KV Cache

LMCache 为 vLLM/SGLang 提供分层存储和跨节点 KV Cache 共享能力。

### 安装与配置

```bash
pip install lmcache
```

### 与 vLLM 集成（CPU Offload）

```python
# lmcache_config.yaml
"""
chunk_size: 256          # KV Cache 传输粒度（tokens）
local_cpu:
  enabled: true
  capacity_gb: 20        # CPU 内存用于 KV Cache 的大小

local_disk:
  enabled: true
  path: /tmp/lmcache
  capacity_gb: 100       # 磁盘缓存大小

remote:
  enabled: false         # 生产环境可开启 Redis 远程共享
  # backend: redis
  # url: redis://localhost:6379
"""

# 启动带 LMCache 的 vLLM Server
# vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
#   --kv-cache-transfer-config '{"kv_connector": "LMCacheConnectorV1"}'
```

### LMCache 的典型使用场景

```python
# 场景：RAG 系统中的文档 KV Cache 复用

import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 加载文档（只计算一次 KV，后续复用）
DOCUMENTS = """
[文档 1] ISO 20022 标准概述...（1000 tokens）
[文档 2] Finacle API 接口规范...（1000 tokens）
[文档 3] FTH 支付流程说明...（1000 tokens）
"""

async def query_with_rag(question: str) -> str:
    """
    每次查询都携带相同的文档上下文
    LMCache 确保文档的 KV Cache 被：
    1. 第一次计算后存入 CPU/Disk
    2. 后续请求直接从缓存加载（节省 GPU 计算）
    """
    response = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": f"参考以下文档回答问题：\n{DOCUMENTS}"},
            {"role": "user",   "content": question}
        ]
    )
    return response.choices[0].message.content

# 并发 10 个 RAG 查询（DOCUMENTS 部分只需 prefill 一次）
questions = [f"问题 {i}" for i in range(10)]
answers = await asyncio.gather(*[query_with_rag(q) for q in questions])
```

---

> **💡 Claude Code 提示词 — LMCache 分层缓存实验：**
> ```
> 用 Python 实现 LMCache 风格的分层 KV Cache 系统（完整可运行版本）：
>
> 1. 不需要真实 GPU，用以下方式模拟：
>    - KV Cache 用 numpy array（大小 = tokens × hidden_dim）
>    - 计算延迟 = tokens × 0.1ms（prefill）
>    - 加载延迟 = bytes / bandwidth（根据存储层带宽计算）
>
> 2. 实现完整的 LMCacheSimulator：
>    - GPU Cache:  容量 1GB，带宽 3000 GB/s
>    - CPU Cache:  容量 10GB，带宽 50 GB/s（PCIe）
>    - Disk Cache: 容量 100GB，带宽 5 GB/s（NVMe）
>    - 插入策略：GPU 满时驱逐到 CPU，CPU 满时驱逐到 Disk
>    - 读取策略：命中 CPU/Disk 时，提升到更高层（预热）
>
> 3. PD Disaggregation 模拟：
>    - Prefill 节点：处理输入，生成 KV Cache，通过"网络"传给 Decode 节点
>    - Decode 节点：接收 KV Cache，执行 token 生成
>    - 模拟网络传输延迟（100 Gbps = 12.5 GB/s）
>    - 对比：combined（P+D 在同一节点）vs disaggregated
>
> 4. RAG 场景基准测试：
>    - 10 个文档（每个 500 token）作为固定上下文
>    - 100 个不同问题（每个 50 token）
>    - 测量：无缓存 vs GPU-only vs 三层分层存储 的总延迟
>
> 5. 最终报告：
>    - 每层缓存的命中率
>    - 总延迟节省百分比
>    - GPU 显存节省百分比
>    - 建议的最优配置（根据测试结果）
> ```

---

## 11. API 层 Prompt Caching（Anthropic / OpenAI）

如果你不自托管，直接用云 API，也可以利用 Prompt Caching 降低成本和延迟。

### Anthropic Prompt Caching

```python
import anthropic

client = anthropic.Anthropic()

# 通过 cache_control 标记要缓存的内容
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "你是 FTH 支付系统专家...",
        },
        {
            "type": "text",
            "text": open("fth_documentation.txt").read(),  # 长文档
            "cache_control": {"type": "ephemeral"}         # ← 标记缓存点
        }
    ],
    messages=[
        {"role": "user", "content": "解释 AM04 错误处理流程"}
    ]
)

# 查看缓存使用情况
usage = response.usage
print(f"输入 token:         {usage.input_tokens}")
print(f"缓存命中 token:     {usage.cache_read_input_tokens}")
print(f"新写入缓存 token:   {usage.cache_creation_input_tokens}")

# 费用对比：
# cache_creation: 比普通输入贵 25%（写入缓存）
# cache_read:     比普通输入便宜 90%（命中缓存）
# TTL:            5 分钟（不活跃则驱逐）
```

### Anthropic Caching 的最佳实践

```python
# ✅ 正确：把稳定内容放在最前面（缓存友好）
messages_correct = {
    "system": [
        {"type": "text", "text": STABLE_SYSTEM},     # 稳定
        {"type": "text", "text": LONG_DOCUMENT,
         "cache_control": {"type": "ephemeral"}},    # 稳定文档，缓存
    ],
    "messages": [
        {"role": "user", "content": USER_QUESTION}   # 变化部分放最后
    ]
}

# ❌ 错误：把变化内容放在缓存点之前（破坏缓存）
messages_wrong = {
    "system": [
        {"type": "text", "text": f"当前时间：{datetime.now()}"},  # 每次不同！
        {"type": "text", "text": LONG_DOCUMENT,
         "cache_control": {"type": "ephemeral"}},    # 无法命中，因为前面变了
    ]
}
```

### OpenAI Automatic Caching

```python
from openai import OpenAI

client = OpenAI()

# OpenAI 自动缓存（无需任何配置）
# 规则：超过 1024 token 的 prompt 中，前缀自动被尝试缓存
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": LONG_SYSTEM_PROMPT},  # 自动缓存
        {"role": "user",   "content": question}
    ]
)

# 查看缓存情况
usage = response.usage
print(f"总输入 token:    {usage.prompt_tokens}")
print(f"缓存命中 token:  {usage.prompt_tokens_details.cached_tokens}")
# 缓存命中部分：比普通输入便宜 50%
# TTL：1 小时（比 Anthropic 更长）
```

### Anthropic vs OpenAI Caching 对比

| 特性 | Anthropic | OpenAI |
|------|-----------|--------|
| 触发方式 | 手动标记 `cache_control` | 自动（前缀对齐到 128 token 块）|
| TTL | 5 分钟（活跃刷新）| 1 小时 |
| 最小缓存长度 | 1024 tokens | 1024 tokens |
| 缓存命中折扣 | 90% 折扣 | 50% 折扣 |
| 写入成本 | +25% | 无额外成本 |
| 可控性 | 精确控制缓存点 | 自动，不可控 |

---

> **💡 Claude Code 提示词 — API Prompt Caching 实战：**
> ```
> 用 Python 实现 Anthropic Prompt Caching 的完整使用示例和效果测量：
>
> 1. 基础实验：
>    - 准备一个 3000 token 的文档（用 requests 抓取一个公开文章）
>    - 准备 10 个不同问题
>    - 第 1 次请求：cache_creation（建立缓存），记录 token 用量和费用
>    - 第 2-10 次请求：cache_read（命中缓存），对比费用
>    - 打印：每次请求的 input/cache_read/cache_creation token 数
>    - 计算：有缓存 vs 无缓存的总费用对比
>
> 2. 多轮对话缓存：
>    - 实现一个支持 Prompt Caching 的对话类 CachedChatSession
>    - 每轮对话后更新 cache_control 的位置（指向最新的对话历史末尾）
>    - 演示 5 轮对话，打印每轮的缓存命中 token 数
>
> 3. 缓存失效演示：
>    - 演示什么情况会导致缓存失效：
>      a. 在缓存内容前插入动态内容（如时间戳）
>      b. 修改缓存内容中的任意 token
>    - 对比失效前后的 cache_read vs cache_creation
>
> 4. 费用计算器：
>    - 函数 calculate_cost(n_requests, doc_tokens, question_tokens)
>    - 计算：纯 API 调用费用 vs 启用 Prompt Caching 的费用
>    - 画出费用随请求数增加的对比曲线（matplotlib）
>    - 标出"盈亏平衡点"（多少次请求后缓存开始省钱）
>
> 使用真实的 claude-opus-4-6 价格计算
> ```

---

## 12. 选型指南

### 决策树

```
你的场景是什么？
│
├── 直接调用云 API（不自托管）
│     ├── Anthropic Claude → 使用 Prompt Caching + cache_control
│     └── OpenAI GPT      → 保持 prompt 前缀稳定，自动命中缓存
│
├── 本地开发 / 原型
│     └── Ollama（最简单，一行命令启动）
│
├── 生产部署（自托管）
│     │
│     ├── 场景：通用高吞吐，固定 system prompt
│     │     └── vLLM + APC（enable_prefix_caching=True）
│     │
│     ├── 场景：多轮对话 / Agent / 非规则前缀
│     │     └── SGLang + RadixAttention（自动发现共享前缀）
│     │
│     ├── 场景：长 context（128K+）/ 历史对话持久化
│     │     └── vLLM 或 SGLang + LMCache（分层存储）
│     │
│     └── 场景：大规模分布式（多节点，高并发）
│           └── llm-d + LMCache + Mooncake（K8s 原生 + RDMA 传输）
│
└── 混合场景
      └── vLLM（主引擎）+ LMCache（存储层）+ Mooncake（传输层）
```

### 针对 Eduacan 项目的建议

```
Eduacan（NZQA 教育平台，RAG 架构）：

需求分析：
  - 每个请求携带相同的考纲文档（长 context）
  - 多轮对话（学生和 AI 反复问答）
  - 预算有限，需要控制 API 成本
  - 本地开发阶段用 Ollama

推荐方案：

开发阶段：
  Ollama（本地）+ Haiku（云端）
  用 Anthropic Prompt Caching 标记考纲内容

生产阶段（学生量增长后）：
  SGLang（自托管推理）
  + LMCache CPU Offload（考纲文档 KV 持久化到 CPU/Disk）
  + Prompt Cache 覆盖：相同考纲的 KV 只计算一次

预期收益：
  考纲部分（~5000 token）只 prefill 一次/会话
  每次问答节省 ~90% 的考纲 prefill 计算
  API 成本降低约 70-80%（Anthropic Prompt Caching）
```

---

## 附录：关键术语速查

| 术语 | 含义 |
|------|------|
| **KV Cache** | 缓存注意力层的 Key/Value 矩阵，避免重复计算 |
| **Prefill** | 处理整个输入 prompt 的阶段（计算密集）|
| **Decode** | 逐 token 生成的阶段（内存密集）|
| **TTFT** | Time to First Token，第一个 token 出现的延迟 |
| **ITL** | Inter-Token Latency，token 间延迟（吞吐量指标）|
| **PagedAttention** | vLLM 的 KV Cache 内存管理，借鉴 OS 分页机制 |
| **RadixAttention** | SGLang 的树形前缀缓存，自动发现共享前缀 |
| **APC** | Automatic Prefix Caching，vLLM 的前缀缓存 |
| **PD Disaggregation** | Prefill-Decode 分离，不同 GPU 各司其职 |
| **KV Quantization** | 用 FP8/INT8 存储 KV Cache，节省 50% 显存 |
| **Copy-on-Write** | Beam Search 前缀共享机制，分叉时才复制 |
| **Block Table** | KV Block 的逻辑→物理地址映射表 |
| **LRU Eviction** | 最近最少使用驱逐策略，移除冷 KV Cache |
| **cache_control** | Anthropic API 中标记缓存点的字段 |
