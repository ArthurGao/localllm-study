# Domain 1 · 2.3 — Subagent 调用、上下文传递与生成
> Claude Certified Architect – Foundations | 考试权重：Domain 1 占 27%

---

## 📌 核心概念速查

| 中文 | 英文关键词 | 说明 |
|------|-----------|------|
| Task 工具 | `Task` tool | 生成（spawn）Subagent 的机制 |
| 允许工具列表 | `allowedTools` | Coordinator 必须包含 "Task" 才能调用 Subagent |
| Agent 定义 | `AgentDefinition` | 配置 Subagent 的描述、system prompt 和工具限制 |
| 上下文传递 | Context Passing | 显式在 Subagent prompt 中注入所需上下文 |
| 并行 Subagent | Parallel Subagents | 单次 Coordinator 响应中发出多个 Task 调用 |
| Fork 会话 | Fork-based Session | 从共享基线创建独立分支探索不同方向 |
| 结构化数据格式 | Structured Data Format | 分离内容与元数据（来源URL、文档名、页码） |
| 来源归属 | Source Attribution | 保留发现与原始来源的映射关系 |

---

## 🧠 核心知识点

### 1. Task 工具与 allowedTools 配置

```python
# Coordinator 必须在 allowedTools 中包含 "Task"
coordinator_config = {
    "name": "coordinator_agent",
    "description": "协调研究任务的主 Agent",
    "system_prompt": "你是一个研究协调专家...",
    "allowedTools": [
        "Task",           # ← 必须包含！否则无法生成 Subagent
        "synthesize",
        "evaluate_coverage"
    ]
}

# Subagent 不需要 "Task"（除非它也需要生成子 Subagent）
search_subagent_config = {
    "name": "search_agent",
    "allowedTools": ["web_search", "fetch_url"]  # 无 "Task"
}
```

**关键考点**：如果 `allowedTools` 中没有 `"Task"`，Coordinator 就无法调用 Subagent，即使描述中说它是协调者。

---

### 2. AgentDefinition 配置结构

```python
from anthropic import AgentDefinition  # 概念性示例

# 每种 Subagent 类型都有独立的 AgentDefinition
search_agent = AgentDefinition(
    name="search_agent",
    description="专门负责网络搜索的 Agent，"
                "输入：搜索查询，输出：相关文章列表和摘要",
    system_prompt="""你是一个专业的信息搜索专家。
    
    你的职责：
    - 执行网络搜索
    - 过滤相关结果
    - 返回结构化的搜索结果（包含来源 URL 和日期）
    
    你不负责：分析或合成发现（这是其他 Agent 的工作）""",
    allowed_tools=["web_search"]
)

analysis_agent = AgentDefinition(
    name="analysis_agent",
    description="专门负责文档深度分析的 Agent",
    system_prompt="""你是一个文档分析专家...""",
    allowed_tools=["read_document", "extract_data"]
)

synthesis_agent = AgentDefinition(
    name="synthesis_agent", 
    description="专门负责综合多来源发现的 Agent",
    system_prompt="""你是一个综合分析专家...""",
    allowed_tools=["verify_fact"]  # 有限的跨角色工具
)
```

---

### 3. 显式上下文传递（最重要实践）

```
原则：Subagent 只能看到你放入其 prompt 的内容

信息传递三要素：
┌─────────────────────────────────────────────────┐
│ 1. 内容 (Content)：前序 Agent 的实际发现         │
│ 2. 元数据 (Metadata)：来源URL、文档名、日期、页码 │
│ 3. 研究目标 (Research Goals)：质量标准           │
└─────────────────────────────────────────────────┘
```

```python
def build_synthesis_prompt(
    topic: str,
    search_results: list,    # 来自 Search Subagent
    analysis_results: dict   # 来自 Analysis Subagent
) -> str:
    """
    构建 Synthesis Subagent 的 prompt
    演示：内容与元数据分离的结构化传递
    """
    
    # 格式化搜索结果（内容 + 元数据）
    formatted_search = "\n".join([
        f"""
[发现 {i+1}]
领域：{r['domain']}
内容：{r['content']}
来源URL：{r.get('source_url', '未知')}       ← 元数据
发布日期：{r.get('date', '未知')}             ← 元数据
相关性评分：{r.get('relevance_score', 'N/A')} ← 元数据
"""
        for i, r in enumerate(search_results)
    ])
    
    return f"""
研究主题：{topic}

【第一阶段：搜索发现】（来源：Search Subagent）
{formatted_search}

【第二阶段：深度分析】（来源：Analysis Subagent）
{analysis_results['content']}

你的任务：综合以上所有信息，生成最终报告。

质量要求：
- 每个声明必须标注来源（保留来源归属）
- 区分有充分支持的发现和有争议的发现
- 明确指出数据覆盖的局限性
"""
```

---

### 4. 并行 Subagent 生成（Parallel Spawning）

这是提高系统效率的关键技术：

```
顺序执行（慢）：          并行执行（快）：
Search Task 1              Search Task 1 ─┐
    ↓                      Search Task 2 ─┼→ 同时执行
Search Task 2              Search Task 3 ─┘
    ↓
Search Task 3
```

**实现方式**：在单个 Coordinator 响应中发出多个 Task 工具调用：

```python
# ✅ 并行：Coordinator 在单次响应的 content 中包含多个 Task 调用
coordinator_content = [
    {
        "type": "tool_use",
        "name": "Task",
        "id": "task_1",
        "input": {
            "agent_name": "search_agent",
            "prompt": "搜索 AI 对视觉艺术的影响"
        }
    },
    {
        "type": "tool_use",  # 同一响应中的第二个 Task
        "name": "Task", 
        "id": "task_2",
        "input": {
            "agent_name": "search_agent",
            "prompt": "搜索 AI 对音乐创作的影响"
        }
    },
    {
        "type": "tool_use",  # 同一响应中的第三个 Task
        "name": "Task",
        "id": "task_3", 
        "input": {
            "agent_name": "search_agent",
            "prompt": "搜索 AI 对电影制作的影响"
        }
    }
]
# 这三个 Task 同时执行，而非顺序执行
```

**对比**：

```python
# ❌ 顺序（低效）：每个 Task 在单独的 turn 中
# Turn 1: 发出 Task 1
# Wait...
# Turn 2: 发出 Task 2（等 Task 1 完成后）
# Wait...
# Turn 3: 发出 Task 3（等 Task 2 完成后）
```

---

### 5. Coordinator Prompt 设计原则

```python
# ❌ 错误：逐步程序指令（限制 Subagent 适应性）
bad_coordinator_prompt = """
你必须按以下顺序执行：
1. 首先调用 search_agent 搜索 'AI 艺术'
2. 然后调用 analysis_agent 分析结果
3. 最后调用 synthesis_agent 生成报告
"""

# ✅ 正确：指定目标和质量标准（给 Subagent 适应空间）
good_coordinator_prompt = """
你的目标：对给定主题进行全面研究，生成有引用来源的报告。

质量标准：
- 覆盖主题的所有主要子领域（不能有重大遗漏）
- 每个声明必须有可追溯的来源
- 发现必须区分有充分支持和有争议的内容
- 如果合成输出有覆盖缺口，进行迭代精化直到满足标准

你可以根据查询复杂度决定调用哪些 Subagent 和调用顺序。
"""
```

---

### 6. Fork 会话管理（Fork-based Session）

```
场景：测试两种重构方案，从同一代码库分析基线出发

共享基线（codebase analysis）
         │
    ┌────┴────┐
    │         │
分支A：     分支B：
微服务架构  模块化单体
  探索        探索
    │         │
  结论A     结论B
    │         │
    └────┬────┘
         │
    Coordinator 比较两个结论
         │
    最终推荐方案
```

```python
# 概念示例：Fork 会话用于并行探索
def compare_refactoring_approaches(codebase_analysis: str):
    # 从同一分析基线 Fork 出两个探索
    fork_a_result = explore_microservices_approach(
        baseline=codebase_analysis  # 共享基线
    )
    fork_b_result = explore_modular_monolith_approach(
        baseline=codebase_analysis  # 同一共享基线
    )
    
    # 比较两个独立分支的结论
    return compare_and_recommend(fork_a_result, fork_b_result)
```

---

## 💡 Claude Code 提示词（生成学习代码）

```
请用 Python + asyncio 实现一个多 Agent 并行研究系统，要求：

1. 实现三种 Subagent（Search/Analysis/Synthesis）：
   - 每个都有独立的 AgentDefinition（system_prompt + allowedTools）
   - 演示 allowedTools 包含/不包含 "Task" 的区别

2. 实现并行 Subagent 生成：
   - Coordinator 在单次响应中发出 3 个并行 Search Task
   - 使用 asyncio.gather 模拟并行执行
   - 对比顺序 vs 并行的耗时差异

3. 实现结构化上下文传递：
   - Search 结果必须包含：content、source_url、date、relevance_score
   - Synthesis prompt 将内容与元数据分离组织
   - 最终报告保留每条声明的来源归属

4. 展示 Fork 会话概念：
   - 从同一代码分析基线 Fork 出两个重构方案探索
   - 最终比较两个 Fork 的结论

文件名：parallel_subagent_system.py
```

---

## 🔍 真实案例：研究系统实现对比

### 案例：来自来源的结构化数据传递

```python
# 实际生产系统中的上下文传递模式

# Search Subagent 返回的结构化数据
search_output = {
    "findings": [
        {
            # 内容部分
            "claim": "生成式 AI 使艺术创作速度提升 300%",
            "evidence": "研究发现专业艺术家使用 AI 工具后...",
            
            # 元数据部分（来源归属）
            "source_url": "https://example.com/ai-art-study",
            "source_name": "MIT Media Lab 2024 研究报告",
            "page": 23,
            "date": "2024-02",
            "credibility": "peer_reviewed"
        },
        {
            "claim": "75% 的音乐人担忧版权问题",
            "evidence": "调查显示...",
            "source_url": "https://music-survey.org/2024",
            "source_name": "全球音乐人协会调查",
            "page": None,
            "date": "2024-01",
            "credibility": "industry_survey"
        }
    ],
    "coverage_gaps": ["电影行业数据不足", "中文内容搜索结果有限"]
}

# Synthesis Subagent 的 prompt 中保留这些元数据
# 确保最终报告中每条声明都可追溯到原始来源
```

---

## 🎯 关键代码：完整 Subagent 调用链

```python
import anthropic
import asyncio
import json
from dataclasses import dataclass
from typing import Optional

client = anthropic.Anthropic()

@dataclass
class AgentResult:
    """标准化 Agent 输出格式，确保上下文传递的一致性"""
    content: str
    metadata: dict  # 来源URL、日期、可信度等
    coverage_gaps: list[str]  # 覆盖缺口注释
    agent_name: str


def run_search_subagent(query: str, domain: str) -> AgentResult:
    """
    Search Subagent
    allowedTools: ["web_search"] - 不包含 "Task"
    """
    print(f"[Search Agent] 执行搜索: {query}")
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system="""你是专业搜索专家。
        
        重要：你的输出必须是严格的 JSON 格式，包含：
        - findings: 发现列表（每条含 claim、evidence、source_url、date）
        - coverage_gaps: 搜索局限性列表""",
        
        messages=[{"role": "user", "content": f"""
请搜索：{query}（领域：{domain}）

返回 JSON 格式：
{{
    "findings": [
        {{
            "claim": "关键发现",
            "evidence": "支持证据",
            "source_url": "来源URL（模拟）",
            "source_name": "来源名称",
            "date": "YYYY-MM"
        }}
    ],
    "coverage_gaps": ["本次搜索的局限性1", "局限性2"]
}}"""}]
    )
    
    # 解析结果（实际项目中应有更健壮的错误处理）
    try:
        parsed = json.loads(response.content[0].text)
    except json.JSONDecodeError:
        parsed = {"findings": [], "coverage_gaps": ["解析失败"]}
    
    return AgentResult(
        content=json.dumps(parsed["findings"], ensure_ascii=False),
        metadata={"domain": domain, "query": query},
        coverage_gaps=parsed.get("coverage_gaps", []),
        agent_name="search_agent"
    )


async def run_parallel_searches(search_tasks: list[tuple]) -> list[AgentResult]:
    """
    并行执行多个 Search Subagent
    模拟 Coordinator 在单次响应中发出多个 Task 调用的效果
    """
    print(f"\n[Coordinator] 并行启动 {len(search_tasks)} 个 Search Subagent...")
    
    loop = asyncio.get_event_loop()
    
    # 并行执行（模拟真实的并行 Task 调用）
    tasks = [
        loop.run_in_executor(None, run_search_subagent, query, domain)
        for query, domain in search_tasks
    ]
    
    results = await asyncio.gather(*tasks)
    print(f"[Coordinator] {len(results)} 个搜索任务完成")
    return list(results)


def run_synthesis_subagent(
    topic: str,
    search_results: list[AgentResult],  # 显式传递
    analysis_result: Optional[AgentResult] = None  # 显式传递
) -> AgentResult:
    """
    Synthesis Subagent
    关键：所有上下文必须显式包含在 prompt 中
    """
    
    # 构建结构化上下文（内容与元数据分离）
    search_context = ""
    for i, result in enumerate(search_results):
        search_context += f"""
【搜索结果 {i+1}】领域：{result.metadata.get('domain', '未知')}
内容：{result.content}
覆盖缺口：{', '.join(result.coverage_gaps)}
---"""
    
    analysis_context = ""
    if analysis_result:
        analysis_context = f"""
【深度分析结果】
{analysis_result.content}"""
    
    prompt = f"""主题：{topic}

{search_context}
{analysis_context}

请综合以上所有信息，生成结构化报告：
1. 各领域关键发现（保留来源引用）
2. 跨领域共同趋势
3. 有争议或不确定的发现（明确标注）
4. 数据覆盖局限性（汇总所有搜索缺口）"""
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=3000,
        system="你是综合分析专家，注重来源归属和不确定性的诚实标注。",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return AgentResult(
        content=response.content[0].text,
        metadata={"topic": topic, "sources_count": len(search_results)},
        coverage_gaps=[gap for r in search_results for gap in r.coverage_gaps],
        agent_name="synthesis_agent"
    )


async def run_full_research_pipeline(topic: str) -> str:
    """
    完整研究流水线（Coordinator 视角）
    演示：并行 Subagent 调用 + 显式上下文传递
    """
    print(f"\n{'='*60}")
    print(f"[Coordinator] 研究主题：{topic}")
    print(f"{'='*60}")
    
    # Step 1: 任务分解（全面覆盖，避免过窄）
    search_tasks = [
        (f"{topic}：视觉艺术和设计领域", "视觉艺术"),
        (f"{topic}：音乐和音频创作领域", "音乐"),
        (f"{topic}：写作和文学创作领域", "写作"),
        (f"{topic}：电影和视频制作领域", "影视"),
    ]
    
    # Step 2: 并行执行 Search Subagents（单次响应发出多 Task）
    search_results = await run_parallel_searches(search_tasks)
    
    # Step 3: 检查覆盖缺口（Coordinator 评估）
    all_gaps = [gap for r in search_results for gap in r.coverage_gaps]
    if all_gaps:
        print(f"\n[Coordinator] 检测到覆盖缺口：{all_gaps[:3]}")
        # 可以在此触发迭代精化（简化：略过）
    
    # Step 4: Synthesis Subagent（显式传递所有 Search 结果）
    final_result = run_synthesis_subagent(
        topic=topic,
        search_results=search_results  # ← 显式传递，关键！
    )
    
    return final_result.content


# 运行
if __name__ == "__main__":
    result = asyncio.run(run_full_research_pipeline("AI 对创意产业的影响"))
    print("\n\n最终报告：")
    print(result[:800])
```

---

## ⚠️ 重点 & 难点

### 难点 1：allowedTools 的 "Task" 配置

**最常见考试陷阱**：
```
如果 Coordinator 的 allowedTools 中没有 "Task"：
→ 无法调用任何 Subagent
→ 系统退化为单 Agent 操作
→ 即使 system prompt 说"你是协调者"也没用
```

### 难点 2：并行 vs 顺序的适用场景

```
适合并行的情况：
✅ 各 Search Task 相互独立（不依赖彼此输出）
✅ 需要快速获取多领域数据
✅ 搜索结果不影响下一个搜索的方向

需要顺序的情况：
📋 Analysis 依赖 Search 的输出（必须顺序）
📋 Synthesis 依赖 Analysis 的输出（必须顺序）
📋 某个 Task 的输出影响下一个 Task 的 prompt
```

### 难点 3：Coordinator Prompt 要指定目标而非步骤

```python
# ❌ 指定步骤（限制灵活性）：
"第一步调用搜索，第二步调用分析..."

# ✅ 指定目标和质量标准（保持灵活性）：
"研究目标是全面覆盖主题，质量标准是每条声明有来源..."
```

---

## 📋 例题练习（考试模拟）

### 例题 1（难度：★★★）

你构建了一个多 Agent 研究系统，其中 Coordinator Agent 的配置如下：

```python
coordinator = AgentDefinition(
    allowedTools=["web_search", "synthesize", "evaluate"]
)
```

系统运行时，Coordinator 无法调用任何 Subagent。最可能的原因是什么？

**A)** system_prompt 中没有明确说明 Coordinator 需要协调其他 Agent

**B)** allowedTools 中缺少 "Task"，这是生成 Subagent 所必需的工具

**C)** Subagent 的 allowedTools 配置错误

**D)** 需要在 API 调用中设置特殊的 coordinator=True 参数

> **正确答案：B**
> 解析：`Task` 工具是生成 Subagent 的唯一机制。`allowedTools` 中没有 "Task"，Coordinator 就无法调用任何 Subagent，无论 prompt 如何描述其角色。

---

### 例题 2（难度：★★★）

你的 Synthesis Subagent 经常产生没有来源归属的摘要。根本原因最可能是什么？

**A)** Synthesis Subagent 的 system_prompt 没有强调引用来源的重要性

**B)** Search Subagent 的输出格式不包含来源元数据（URL、文档名、日期）

**C)** Coordinator 在传递给 Synthesis Subagent 的 prompt 中没有包含来源元数据，只传递了内容

**D)** 应该使用不同的模型来处理合成任务

> **正确答案：C**
> 解析：如果 Coordinator 在向 Synthesis Subagent 传递上下文时只传递了内容（findings）而丢失了元数据（source_url、date 等），Synthesis Agent 就没有数据可以引用。内容与元数据必须同时传递。B 也部分正确，但 C 更直接指向 Coordinator 的传递行为。

---

### 例题 3（难度：★★☆）

以下哪种方式能让三个独立的 Search Subagent 并行执行（而非顺序执行）？

**A)** 在三个单独的 Coordinator 轮次（turn）中分别发出三个 Task 调用

**B)** 在单个 Coordinator 响应的 content 列表中同时包含三个 Task 工具调用

**C)** 使用多线程在 Python 层面同时调用三次 Coordinator

**D)** 设置 parallel=True 参数在 Task 工具定义中

> **正确答案：B**
> 解析：并行 Subagent 生成的关键是在**单个 Coordinator 响应**中发出多个 Task 调用。A 是顺序执行（每个轮次等待前一个完成）。C、D 是不存在的机制。

---

## 🔗 关联知识点

- **Task Statement 1.2**：Coordinator-Subagent 架构（上下文隔离）
- **Task Statement 1.4**：工作流强制执行（allowedTools 的安全意义）
- **Task Statement 2.3**：工具分配原则（Subagent 只获得所需工具）
- **Task Statement 5.6**：信息来源溯源保留

---

*文档版本：v1.0 | 对应考试 Domain 1 Task Statement 1.3*
