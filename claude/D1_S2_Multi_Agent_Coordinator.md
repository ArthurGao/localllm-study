# Domain 1 · 2.2 — 多 Agent 协调：Coordinator-Subagent 模式
> Claude Certified Architect – Foundations | 考试权重：Domain 1 占 27%

---

## 📌 核心概念速查

| 中文 | 英文关键词 | 说明 |
|------|-----------|------|
| 协调者 Agent | Coordinator Agent | 管理所有子 Agent 的中央控制器 |
| 子 Agent | Subagent | 专注于特定任务的专业 Agent |
| 轮辐架构 | Hub-and-spoke Architecture | Coordinator 作为中心，Subagent 作为辐条 |
| 上下文隔离 | Context Isolation | Subagent 不自动继承 Coordinator 的对话历史 |
| 任务分解 | Task Decomposition | 将复杂任务拆分为可执行的子任务 |
| 结果聚合 | Result Aggregation | 将多个 Subagent 的输出合并为统一结果 |
| 迭代精化 | Iterative Refinement | Coordinator 评估合成输出并反复改进 |
| 覆盖缺口 | Coverage Gap | 任务分解过窄导致的主题遗漏 |

---

## 🧠 核心知识点

### 0. 先搞清楚本质：Agent 就是你写的代码 + Claude API 调用

> 很多人以为有某种内置的"多 Agent 框架"自动运行。实际上：

```
Coordinator = 你的代码  调用一次 Claude API（带特定 system prompt）
Subagent A  = 你的代码  再调用一次 Claude API（带不同 system prompt）
Subagent B  = 你的代码  再调用一次 Claude API（带不同 system prompt）

它们之间没有魔法 —— 就是多个独立的、无状态的 API 调用
"Agent" 只是我们给"有特定职责的 API 调用"起的名字
```

**并行运行（Parallel Execution）**：需要你自己实现，用 `asyncio.gather` 同时发出多个 API 调用：

```python
import asyncio
import anthropic

client = anthropic.AsyncAnthropic()   # 使用 Async 版本

# ── 每个 "Agent" 就是一个 async 函数 ──────────────────────────

async def search_subagent(query: str, domain: str) -> dict:
    """Subagent A — 本质：一次独立的 Claude API 调用"""
    response = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system="你是搜索专家，只负责搜索，不负责分析",  # 这个 Agent 的角色
        messages=[{"role": "user", "content": query}]
        # 注意：这里没有 Coordinator 的任何历史 ← 上下文隔离
    )
    return {"domain": domain, "result": response.content[0].text}


async def run_coordinator(topic: str) -> str:
    """Coordinator — 也是一次 Claude API 调用，但负责决策和整合"""

    # ✅ 并行：asyncio.gather 同时发出 3 个 API 调用
    search_results = await asyncio.gather(
        search_subagent(f"{topic} 视觉艺术", "visual"),  # 同时发出
        search_subagent(f"{topic} 音乐",     "music"),   # 同时发出
        search_subagent(f"{topic} 电影",     "film"),    # 同时发出
    )
    # 3 个 API 调用并发，不是顺序等待，节省 ~60% 时间

    # Coordinator 拿到所有结果后，再调用 Claude 做合成
    synthesis = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system="你是综合分析专家",
        messages=[{
            "role": "user",
            # ← 必须显式把 search_results 传进来，Claude 不会自动知道
            "content": f"请综合以下搜索结果，生成报告：\n{search_results}"
        }]
    )
    return synthesis.content[0].text


# ── 顺序 vs 并行 的选择 ────────────────────────────────────────
#
# 可以并行（互相独立，无依赖）：
#   Search A（视觉）─┐
#   Search B（音乐）─┼─ asyncio.gather → 同时发出
#   Search C（电影）─┘
#
# 必须顺序（有依赖）：
#   Search → Analysis → Synthesis
#   Analysis 依赖 Search 的输出，必须等 Search 完成后才能开始
```

**考试里的 "Task 工具" 和 "Claude Agent SDK"**：

```
你手写版（直接用 Messages API）：
  自己写 asyncio.gather
  自己管理 messages 列表
  自己把上一个 Agent 的输出塞进下一个 Agent 的 prompt

Claude Agent SDK 版（更高层封装）：
  用 Task 工具"描述"要做什么
  SDK 在后台帮你发出并发 API 调用
  SDK 帮你管理部分上下文传递

考试考的是原理和架构决策，不考具体 SDK 语法。
```

---

### 0b. Claude 怎么知道有哪些 Agent 可以调用？

> 很多人以为 Agent 有特殊的"注册机制"，实际上就是工具描述。

**两种架构，机制完全不同：**

**架构 A：纯代码调度 — Claude 根本不知道有其他 Agent**

```python
# Claude Coordinator 只是做分析，输出文字
# 你的 Python 代码决定接下来调哪些 Subagent
# Claude 对调度逻辑一无所知

coordinator_response = await client.messages.create(
    system="你是研究协调专家",
    messages=[{"role": "user", "content": f"分析主题：{topic}"}]
    # 没有 tools 参数 → Claude 不知道有任何 Agent 或工具
)

# 你的代码读取输出后，自己决定调哪些函数
results = await asyncio.gather(
    search_subagent("视觉艺术"),   # Python 代码决定的，Claude 不知道
    search_subagent("音乐"),
)
```

**架构 B：Task 工具调度 — Agent 通过 Task 工具的 description 注册（考试重点）**

```python
# Agent 信息写在 Task 工具的 description 里
# Claude 通过调用 Task 工具来"生成"Subagent
# 机制和调用普通工具完全一样

coordinator_tools = [
    {
        "name": "Task",
        "description": """
生成一个专业子 Agent 执行特定任务。

可用的 Agent 类型（agent_name 参数）：
  - search_agent:   负责网络搜索，输入搜索查询，返回相关文章列表
  - analysis_agent: 负责深度分析文档，输入文档内容，返回分析结果
  - synthesis_agent:负责综合多个来源，输入多份分析结果，返回最终报告

重要：prompt 参数必须包含 Agent 完成任务所需的所有上下文，
      因为 Agent 无法自动获取当前对话历史。
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": ["search_agent", "analysis_agent", "synthesis_agent"]
                },
                "prompt": {
                    "type": "string",
                    "description": "给 Agent 的完整指令，必须包含所有上下文"
                }
            },
            "required": ["agent_name", "prompt"]
        }
    }
]

# Coordinator 的 Claude 看到 Task 工具的 description
# → 知道有三种 Agent 可用
# → 自己决定调哪些、以什么参数调用
response = await client.messages.create(
    model="claude-opus-4-5",
    tools=coordinator_tools,        # ← Task 工具在这里，Agent 信息在描述里
    tool_choice={"type": "any"},
    system="你是研究协调专家，使用 Task 工具调度专业 Agent 完成研究任务",
    messages=[{"role": "user", "content": f"研究：{topic}"}]
)

# Claude 返回的 tool_use 可能是：
# Task(agent_name="search_agent",   prompt="搜索视觉艺术...")
# Task(agent_name="search_agent",   prompt="搜索音乐...")
# Task(agent_name="analysis_agent", prompt="分析以下内容...")
# 你的代码接收这些调用，执行对应的 Subagent 函数
```

```
总结：
  普通工具：description 告诉 Claude "这个工具做什么"
  Task 工具：description 告诉 Claude "有哪些 Agent 可用，各自做什么"

  机制完全相同 —— Agent 就是通过 Task 工具的 description "注册"的
  allowedTools 必须包含 "Task"，Coordinator 才能调度 Subagent
```

**两种架构的选择：**

```
用纯代码调度（架构 A）：
  ✅ 流程固定，你已经知道调哪些 Agent
  ✅ 不需要 Claude 动态决定调度逻辑

用 Task 工具调度（架构 B）：
  ✅ 需要 Claude 根据用户请求动态决定调哪些 Agent
  ✅ 考试重点场景
```



---

### 0c. Subagent 在哪里运行？你的机器，不是 Claude 服务器

> 很多人以为 Claude 服务器会自动启动 Subagent。实际上 Claude 服务器只是一个无状态推理引擎（Stateless Inference Engine），什么都不运行。

```
你的机器                              Claude 服务器
─────────────────────────────────────────────────────

1. 你的代码发送 Coordinator 请求
   [messages]              ──────→   接收请求
                                      推理
                           ←──────   返回响应
                                      {tool_use: Task(
                                        agent_name: "search_agent",
                                        prompt: "搜索..."
                                      )}
                                      ↑
                                      Claude 只是"说"它想调用 Task
                                      Claude 服务器不执行任何东西

2. 你的代码收到 tool_use: Task
   → 解析 agent_name
   → 在本机启动 Subagent（发新的 API 请求）
   [subagent messages]     ──────→   接收请求
                                      推理
                           ←──────   返回 Subagent 响应

3. 你的代码收到 Subagent 结果
   → 打包成 tool_result
   → 追加到 Coordinator messages
   → 再次发送给 Claude
   [updated messages]      ──────→   继续推理...
```

**所有编排逻辑（Orchestration）都在你这边运行：**

```
你的机器上运行：                  Claude 服务器只做：
  ✅ Agentic loop 控制流            接收 messages
  ✅ 决定是否执行 Task              返回 tool_use 或 end_turn
  ✅ 启动 Subagent（新 API 调用）   处理完即结束，不保留状态
  ✅ 管理 messages 列表
  ✅ asyncio.gather 并行逻辑
  ✅ 工具执行（查数据库等）
```

**代码印证：**

```python
# 证明：Subagent 在你的代码里启动，不是 Claude 服务器

async def execute_task_tool(task_input: dict) -> str:
    """
    当 Coordinator 的 Claude 返回 tool_use: Task 时
    是你的代码负责执行这个 Task，启动 Subagent
    """
    agent_name = task_input["agent_name"]
    prompt     = task_input["prompt"]

    # 你的代码从你的机器发起新的 HTTP 请求到 Claude API
    response = await client.messages.create(      # ← 你发出的
        model="claude-opus-4-5",
        system=AGENT_DEFINITIONS[agent_name]["system_prompt"],
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# Agentic loop 里处理 Task 调用：
for block in coordinator_response.content:
    if block.type == "tool_use" and block.name == "Task":

        # 完全在你的进程里执行
        # Claude 服务器不知道这一步发生了什么
        result = await execute_task_tool(block.input)

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result
        })
```

**类比：**

```
Claude 服务器  =  一个顾问
                  你问他"接下来该做什么"
                  他说"去找 search_agent 搜索"
                  但他自己不去找，他只是给建议

你的代码      =  实际执行的人
                  收到建议后，真的去启动 search_agent
                  等结果回来，再去问顾问下一步怎么办
```

---

### 1. Hub-and-Spoke 架构图

```
                    ┌─────────────────┐
                    │   Coordinator   │  ← 所有通信经过此节点
                    │   (Hub/中心)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ Search   │  │ Analyze  │  │Synthesize│
       │Subagent  │  │Subagent  │  │Subagent  │
       └──────────┘  └──────────┘  └──────────┘
       
关键原则：
✅ Subagent 之间不直接通信
✅ 所有信息流经 Coordinator
✅ 错误处理集中在 Coordinator
✅ Subagent 上下文相互隔离
```

---

### 2. 上下文隔离（Context Isolation）— 最重要的概念

```python
# ❌ 错误假设：Subagent 会"记得"之前的对话
coordinator_messages = [
    {"role": "user", "content": "研究 AI 对创意产业的影响"},
    {"role": "assistant", "content": "好的，我已找到以下信息：..."},
]
# 此时调用 Subagent，它对上面的对话一无所知！

# ✅ 正确做法：显式传递上下文
subagent_prompt = f"""
你是一个文档分析专家。

【背景信息 - 搜索 Agent 已找到的内容】
{search_results}  # 必须显式注入

请对以上搜索结果进行深度分析...
"""
```

**为什么这很重要？**
- 每个 Subagent 是一个独立的 API 调用
- Claude API 是无状态的（stateless）
- Subagent 只能看到你显式放入其 prompt 的内容

---

### 3. Coordinator 的四大职责

```
┌─────────────────────────────────────────────┐
│              Coordinator 职责                 │
├─────────────────────────────────────────────┤
│ 1. 任务分解 (Task Decomposition)             │
│    将用户请求拆分为专业子任务                  │
│    ⚠️ 分解不能太窄！否则遗漏主题              │
├─────────────────────────────────────────────┤
│ 2. 动态委派 (Dynamic Delegation)             │
│    根据查询复杂度决定调用哪些 Subagent         │
│    不是所有任务都需要走完整流水线              │
├─────────────────────────────────────────────┤
│ 3. 结果聚合 (Result Aggregation)             │
│    合并多个 Subagent 的输出                   │
│    保留来源归属（source attribution）         │
├─────────────────────────────────────────────┤
│ 4. 迭代精化 (Iterative Refinement)           │
│    评估合成输出是否有覆盖缺口                  │
│    如有缺口，重新委派针对性查询                │
└─────────────────────────────────────────────┘
```

---

### 4. 任务分解过窄的风险（Coverage Gap）

这是 Domain 1 的**高频考题场景**，来自考试 Question 7：

```
用户请求：研究"AI 对创意产业的影响"

❌ 错误的 Coordinator 分解（太窄）：
  - 子任务1: "AI 在数字艺术创作中的应用"
  - 子任务2: "AI 在平面设计中的应用"  
  - 子任务3: "AI 在摄影中的应用"
  
结果：音乐、写作、电影制作完全被遗漏！
所有 Subagent 完美执行了任务，但系统输出错误。

✅ 正确的 Coordinator 分解：
  - 子任务1: "AI 对视觉艺术（绘画、摄影、设计）的影响"
  - 子任务2: "AI 对音乐和音频创作的影响"
  - 子任务3: "AI 对写作和文学创作的影响"
  - 子任务4: "AI 对电影和视频制作的影响"
```

**根本原因**：问题在 Coordinator 的分解策略，而非 Subagent 的执行能力。

---

### 5. 迭代精化循环（Iterative Refinement Loop）

```
Coordinator
    │
    ├─→ 生成搜索任务 → Search Subagent
    │
    ├─→ 发送给分析 → Analysis Subagent  
    │
    ├─→ 发送给合成 → Synthesis Subagent → 输出报告草稿
    │
    ├─→ [评估] 报告是否有覆盖缺口？
    │      │
    │      ├─ 有缺口 → 重新委派针对性搜索 → 循环
    │      │
    │      └─ 无缺口 → 最终输出
    │
    └─→ 完成
```

---

## 💡 Claude Code 提示词（生成学习代码）

```
请用 Python 实现一个多 Agent 研究系统（Multi-Agent Research System），要求：

1. 实现 Hub-and-Spoke 架构：
   - Coordinator Agent：负责任务分解、委派、结果聚合
   - Search Subagent：模拟网络搜索（返回虚构但合理的数据）
   - Analysis Subagent：分析文档
   - Synthesis Subagent：综合所有发现

2. 核心要求：
   - Subagent 不自动继承 Coordinator 上下文
   - 上下文必须显式传递
   - 展示迭代精化循环（至少一次再委派）
   - 合成输出保留来源归属

3. 演示以下场景对比：
   - 错误：Coordinator 过窄分解"AI 对创意产业影响"（只覆盖视觉艺术）
   - 正确：Coordinator 全面分解（覆盖视觉、音乐、写作、电影）

4. 使用 Anthropic SDK，添加详细中文注释

文件名：multi_agent_research_system.py
```

---

## 🔍 真实案例对比

### 案例 1：错误实现（Subagent 上下文依赖）

```python
# ❌ 真实开发中的常见错误
class ResearchSystem:
    def __init__(self):
        self.shared_messages = []  # 共享消息列表
    
    def run(self, topic):
        # Coordinator 搜索
        self._add_message("user", f"搜索关于 {topic} 的信息")
        search_result = self._call_claude()
        
        # ❌ 错误：以为 Subagent 能看到之前的对话
        self._add_message("user", "现在分析上面搜索到的内容")
        # 问题：如果这是新的 API 调用（Subagent），
        # 它无法访问上面的 shared_messages！
```

### 案例 2：正确实现（显式上下文传递）

```python
# ✅ 正确实现
def run_coordinator(topic: str):
    # Step 1: Search Subagent（独立调用）
    search_results = run_search_subagent(
        query=topic,
        context=None  # 无需前置上下文
    )
    
    # Step 2: Analysis Subagent（显式注入 Search 结果）
    analysis = run_analysis_subagent(
        documents=search_results,   # ← 显式传递
        context=f"分析主题：{topic}" # ← 显式传递
    )
    
    # Step 3: Synthesis Subagent（显式注入所有前序输出）
    final_report = run_synthesis_subagent(
        search_data=search_results,   # ← 显式传递
        analysis_data=analysis,       # ← 显式传递
        topic=topic                   # ← 显式传递
    )
    
    return final_report
```

### 案例 3：并行 Subagent 执行（来自 Task Statement 1.3）

```python
# 在单个 Coordinator 响应中发出多个 Task 工具调用
# 实现真正的并行执行（而非顺序调用）

coordinator_response_content = [
    {
        "type": "tool_use",
        "name": "Task",
        "input": {
            "agent": "search_subagent",
            "prompt": "搜索 AI 对视觉艺术的影响"
        }
    },
    {
        "type": "tool_use", 
        "name": "Task",
        "input": {
            "agent": "search_subagent",
            "prompt": "搜索 AI 对音乐创作的影响"
        }
    }
    # 两个 Task 同时发出 → 并行执行
]
```

---

## 🎯 完整代码示例

```python
import anthropic
import json
from typing import Optional

client = anthropic.Anthropic()

# ============================================================
# Subagent 执行函数（Context Isolation 演示）
# ============================================================

def run_search_subagent(query: str, domain: str) -> dict:
    """
    搜索 Subagent：完全独立的 API 调用
    上下文：只包含搜索任务，不包含 Coordinator 历史
    """
    print(f"\n[Search Subagent] 搜索: {query}")
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system="你是一个专业的信息搜索专家。请搜索并返回关于给定主题的关键信息。",
        messages=[{
            "role": "user",
            "content": f"""请搜索以下主题：{query}
            
            返回格式（JSON）：
            {{
                "domain": "{domain}",
                "key_findings": ["发现1", "发现2", "发现3"],
                "source": "模拟数据来源URL",
                "date": "2024-03"
            }}"""
        }]
    )
    
    # 解析结果（实际项目中应有更健壮的解析）
    return {"domain": domain, "content": response.content[0].text}


def run_analysis_subagent(
    search_results: list,  # 显式传入的 Search 结果
    topic: str
) -> dict:
    """
    分析 Subagent：接收显式传入的 Search 结果
    ⚠️ 关键：search_results 必须显式传入，不能假设 Subagent 能自动获取
    """
    print(f"\n[Analysis Subagent] 分析 {len(search_results)} 个搜索结果")
    
    # 将 search_results 格式化后注入 prompt
    search_context = "\n".join([
        f"领域：{r['domain']}\n内容：{r['content']}"
        for r in search_results
    ])
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system="你是一个专业的数据分析专家。",
        messages=[{
            "role": "user",
            "content": f"""主题：{topic}

【搜索 Agent 已获取的原始数据】
{search_context}

请对以上数据进行深度分析，识别：
1. 主要趋势
2. 关键影响
3. 覆盖缺口（哪些方面搜索数据不足？）"""
        }]
    )
    
    return {"analysis": response.content[0].text}


def run_synthesis_subagent(
    search_results: list,  # 显式传入
    analysis: dict,         # 显式传入
    topic: str
) -> str:
    """
    合成 Subagent：接收所有前序 Agent 的输出
    必须保留来源归属（source attribution）
    """
    print(f"\n[Synthesis Subagent] 合成最终报告")
    
    # 构建包含所有上下文的 prompt
    prompt = f"""主题：{topic}

【搜索数据】（来源：Search Subagent）
{json.dumps([r['content'] for r in search_results], ensure_ascii=False, indent=2)}

【分析结果】（来源：Analysis Subagent）
{analysis['analysis']}

请综合以上所有信息，生成一份结构化报告：
1. 报告标题
2. 执行摘要
3. 各领域详细发现（保留来源信息）
4. 结论
5. 数据覆盖局限性说明"""
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=3000,
        system="你是一个专业报告撰写专家，注重数据来源的准确引用。",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text


# ============================================================
# Coordinator：动态任务分解演示
# ============================================================

def run_coordinator(topic: str, decomposition_mode: str = "correct") -> str:
    """
    Coordinator Agent
    
    decomposition_mode:
        "narrow"  → 展示错误的过窄分解（只覆盖视觉艺术）
        "correct" → 展示正确的全面分解
    """
    
    print(f"\n{'='*60}")
    print(f"Coordinator 启动 | 主题：{topic}")
    print(f"分解模式：{decomposition_mode}")
    print(f"{'='*60}")
    
    # Step 1: 任务分解（关键步骤）
    if decomposition_mode == "narrow":
        # ❌ 错误分解：只覆盖视觉艺术（考试 Q7 的错误案例）
        search_tasks = [
            ("AI 在数字艺术创作中的应用", "视觉艺术"),
            ("AI 在平面设计中的应用", "平面设计"),
            ("AI 在摄影中的应用", "摄影"),
        ]
        print("\n⚠️ [错误模式] 任务分解过窄，遗漏音乐、写作、电影领域！")
    else:
        # ✅ 正确分解：覆盖所有创意产业
        search_tasks = [
            ("AI 对视觉艺术和设计的影响", "视觉艺术"),
            ("AI 对音乐创作和音频制作的影响", "音乐"),
            ("AI 对写作和文学创作的影响", "写作"),
            ("AI 对电影和视频制作的影响", "影视"),
        ]
        print("\n✅ [正确模式] 全面覆盖创意产业各领域")
    
    # Step 2: 并行执行 Search Subagents（演示：顺序执行模拟）
    print(f"\n[Coordinator] 委派 {len(search_tasks)} 个搜索任务...")
    search_results = []
    for query, domain in search_tasks:
        result = run_search_subagent(query=query, domain=domain)
        search_results.append(result)
    
    # Step 3: Analysis Subagent（显式传递 search_results）
    analysis = run_analysis_subagent(
        search_results=search_results,  # 显式传递！
        topic=topic
    )
    
    # Step 4: 检查覆盖缺口（Iterative Refinement）
    if "覆盖不足" in analysis["analysis"] or "缺少" in analysis["analysis"]:
        print("\n[Coordinator] 检测到覆盖缺口，进行迭代精化...")
        # 针对缺口进行补充搜索（简化演示）
        supplementary = run_search_subagent(
            query="AI 对其他创意领域的影响补充",
            domain="补充研究"
        )
        search_results.append(supplementary)
        # 重新分析
        analysis = run_analysis_subagent(
            search_results=search_results,
            topic=topic
        )
    
    # Step 5: Synthesis Subagent（显式传递所有上下文）
    final_report = run_synthesis_subagent(
        search_results=search_results,  # 显式传递！
        analysis=analysis,               # 显式传递！
        topic=topic
    )
    
    return final_report


# ============================================================
# 运行对比演示
# ============================================================
if __name__ == "__main__":
    topic = "AI 对创意产业的影响"
    
    print("\n" + "="*60)
    print("演示 1：错误的过窄分解（会遗漏音乐、写作、电影）")
    print("="*60)
    narrow_result = run_coordinator(topic, decomposition_mode="narrow")
    
    print("\n" + "="*60)
    print("演示 2：正确的全面分解")
    print("="*60)
    correct_result = run_coordinator(topic, decomposition_mode="correct")
    
    print("\n\n最终报告（正确版本）：")
    print(correct_result[:500] + "...")  # 只打印前 500 字
```

---

## ⚠️ 重点 & 难点

### 难点 1：分辨"子 Agent 失败"还是"Coordinator 分解失败"

**考试陷阱**：当系统输出不完整时，考题倾向于让你去检查 Subagent 的实现，但真正的根因往往是 Coordinator 的任务分解策略。

**判断方法**：
- 每个 Subagent 都"成功完成"了被分配的任务 → 根因在 Coordinator 分解
- 某个 Subagent 报错或返回空结果 → 根因在该 Subagent

### 难点 2：并行 vs 顺序 Subagent 执行

```python
# 顺序执行（低效，但简单）
result1 = run_subagent_a()
result2 = run_subagent_b(result1)  # 依赖 result1

# 并行执行（高效，Coordinator 在单响应中发出多个 Task）
# 适用于独立任务（search task 1 和 search task 2 互不依赖）
import asyncio
results = await asyncio.gather(
    run_subagent_a(),
    run_subagent_b()
)
```

### 难点 3：何时需要迭代精化

- 合成输出有明确的覆盖缺口注释
- 用户验证发现重要领域缺失
- 质量标准检查未通过

---

## 📋 例题练习（考试模拟）

### 例题 1（对应考试 Question 7，难度：★★★）

运行多 Agent 研究系统分析"AI 对创意产业的影响"。每个 Subagent 都成功完成了任务：Search Agent 找到了相关文章，Analysis Agent 正确分析了文档，Synthesis Agent 生成了连贯输出。但最终报告只覆盖了视觉艺术，完全遗漏了音乐、写作和电影制作。

检查 Coordinator 日志发现任务分解为："AI 在数字艺术创作中的应用"、"AI 在平面设计中的应用"、"AI 在摄影中的应用"。

最可能的根本原因是什么？

**A)** Synthesis Agent 缺乏识别接收数据覆盖缺口的指令

**B)** Coordinator Agent 的任务分解过窄，导致 Subagent 分配不覆盖主题的所有相关领域

**C)** Search Agent 的查询不够全面，需要扩展以覆盖更多创意产业部门

**D)** Analysis Agent 因过于严格的相关性标准过滤掉了非视觉创意产业的来源

> **正确答案：B**
> 解析：日志直接揭示了根本原因。分配了正确范围内的任务，问题在于分配了什么。Subagent 在其分配范围内执行正确，A、C、D 错误地指责了在正确分配范围内工作的下游 Agent。

---

### 例题 2（难度：★★☆）

在设计 Coordinator 时，以下哪种方式能确保 Synthesis Subagent 获得 Search Subagent 的发现？

**A)** 将所有 Agent 共享同一个 messages 列表

**B)** 在 Synthesis Subagent 的 system prompt 中告知它可以调用 Search Agent

**C)** 在 Synthesis Subagent 的 prompt 中直接包含 Search Subagent 的完整输出

**D)** 使用全局变量在 Agent 之间共享状态

> **正确答案：C**
> 解析：Subagent 拥有隔离上下文，不自动继承任何其他 Agent 的历史。唯一可靠的方式是显式在 prompt 中传递所需信息。A、D 违反了上下文隔离原则，B 会导致递归调用问题。

---

### 例题 3（难度：★★★）

研究发现 Synthesis Agent 在合并发现时频繁需要验证特定声明。当前流程需要 2-3 次往返（Synthesis → Coordinator → Search → Synthesis），导致延迟增加 40%。评估显示 85% 的验证是简单事实核查。最有效的方案是什么？

**A)** 给 Synthesis Agent 提供一个作用域 `verify_fact` 工具用于简单查询，复杂验证继续通过 Coordinator 委派

**B)** 让 Synthesis Agent 积累所有验证需求，最后批量返回给 Coordinator

**C)** 给 Synthesis Agent 访问所有搜索工具的权限，以便直接处理任何验证需求

**D)** 让 Search Agent 在初始搜索时主动缓存额外上下文

> **正确答案：A**
> 解析：最小权限原则——只给 Synthesis Agent 处理 85% 简单场景所需的工具（`verify_fact`），同时保留复杂验证的现有协调模式。C 违反了职责分离，D 依赖无法可靠预测需求的推测性缓存。

---

## 🔗 关联知识点

- **Task Statement 1.3**：Task 工具与 allowedTools 配置
- **Task Statement 1.6**：动态任务分解策略
- **Task Statement 5.6**：信息来源溯源在多源合成中的保留

---

*文档版本：v1.0 | 对应考试 Domain 1 Task Statement 1.2*
