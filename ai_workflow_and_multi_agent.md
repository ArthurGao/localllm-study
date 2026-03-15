# AI Workflow 与 Multi-Agent 深度解析

> 本文档用于配合 Claude Code 生成学习代码，每个章节末尾附有 **代码生成提示词**。

---

## 目录

1. [核心概念辨析](#1-核心概念辨析)
2. [AI Workflow 详解](#2-ai-workflow-详解)
   - 2.1 Prompt Chaining
   - 2.2 Parallelization
   - 2.3 Routing
   - 2.4 Map-Reduce
3. [Multi-Agent 详解](#3-multi-agent-详解)
   - 3.1 Orchestrator + Workers
   - 3.2 Peer-to-Peer
   - 3.3 Human-in-the-Loop
4. [Workflow vs Multi-Agent 对比](#4-workflow-vs-multi-agent-对比)
5. [框架选型指南](#5-框架选型指南)
6. [综合实战案例](#6-综合实战案例)

---

## 1. 核心概念辨析

### 什么是 Workflow？

Workflow 是**由开发者预先定义好的任务执行流程**。控制流（走哪条路、执行哪一步）由代码决定，LLM 只负责在各个节点完成具体的文本处理任务。

```
[开发者编排控制流]

Input
  │
  ▼
Step 1: LLM 做分类
  │
  ▼
Step 2: LLM 做翻译
  │
  ▼
Step 3: LLM 做总结
  │
  ▼
Output
```

**关键特征：**
- 控制流确定，执行路径可预测
- LLM 是"工具"，被调用
- 适合重复性、结构化任务

---

### 什么是 Agent？

Agent 是**能够自主感知环境、规划步骤、调用工具、并做出决策的 AI 系统**。控制流由 LLM 自身的推理决定。

```
[LLM 自己决定控制流]

Input
  │
  ▼
┌─────────────────────┐
│      Agent Loop     │
│  Think → Act → Obs  │◄──────┐
│  (LLM 决定下一步)    │       │
└─────────────────────┘       │
  │                           │
  ▼                     (继续循环？)
Output
```

**关键特征：**
- 控制流动态，LLM 自主规划
- LLM 是"决策者"
- 适合复杂、开放性任务

---

### 什么是 Multi-Agent？

Multi-Agent 是**多个 Agent 协同工作的系统**，每个 Agent 有独立的 LLM、工具集、记忆，通过消息传递协作完成复杂任务。

```
Orchestrator Agent
    │
    ├──► Research Agent ──► [web_search, read_url]
    │
    ├──► Code Agent ──────► [write_file, run_code]
    │
    └──► Writer Agent ────► [write_doc, format]
```

---

## 2. AI Workflow 详解

### 2.1 Prompt Chaining（链式调用）

**原理：** 上一个 LLM 调用的输出，作为下一个调用的输入，形成处理链。

**适用场景：**
- 任务需要多个独立步骤，每步依赖上一步结果
- 每步使用不同的 prompt 模板
- 需要在步骤间插入验证逻辑（gate check）

**示意图：**
```
User Query
    │
    ▼
[Step 1] Translate to English
    │  output: english_text
    ▼
[Step 2] Extract key facts
    │  output: facts_list
    ▼
[Step 3] Generate summary
    │  output: summary
    ▼
Final Output
```

**实际例子：新闻分析 Pipeline**
```
输入: 一篇中文新闻
Step 1 → 翻译成英文
Step 2 → 提取：时间、地点、人物、事件
Step 3 → 判断新闻情感倾向（正/负/中立）
Step 4 → 生成 100 字摘要
输出: { translation, entities, sentiment, summary }
```

**Gate Check 模式（带验证的链）：**
```
Step 1 → 生成内容
    │
    ▼
[Gate] 检查内容质量
    ├── Pass → 继续 Step 2
    └── Fail → 返回 Step 1 重新生成（最多 N 次）
```

---

> **💡 Claude Code 代码生成提示词：**
> ```
> 用 Python + LangChain LCEL 实现一个 Prompt Chaining workflow：
> 1. Step 1: 将用户输入的中文文本翻译成英文
> 2. Step 2: 从英文文本中提取 entities（人物/地点/时间）
> 3. Step 3: 生成情感分析（positive/negative/neutral）
> 4. 每步之间加 gate check，若输出为空则重试最多 3 次
> 5. 最终返回 { translation, entities, sentiment }
> 使用 claude-3-5-sonnet，展示完整的 chain 组合方式
> ```

---

### 2.2 Parallelization（并行化）

**原理：** 将独立的子任务分配给多个 LLM 并发执行，最后聚合结果。

**两种子模式：**

**① Sectioning（分段并行）：** 任务本身可以拆分为独立部分
```
大文档
    │
    ├──► [LLM 1] 分析第 1-10 页 ──►┐
    ├──► [LLM 2] 分析第 11-20 页 ──►├──► 聚合 ──► 总报告
    └──► [LLM 3] 分析第 21-30 页 ──►┘
```

**② Voting（投票）：** 同一任务交给多个 LLM，取共识结果
```
同一个问题
    │
    ├──► [LLM 1] 给出答案 A ──►┐
    ├──► [LLM 2] 给出答案 B ──►├──► 投票/评审 ──► 最终答案
    └──► [LLM 3] 给出答案 C ──►┘
```

**实际例子：代码安全审查**
```
一份代码库
    ├── [LLM 1] 检查 SQL 注入漏洞
    ├── [LLM 2] 检查 XSS 漏洞
    ├── [LLM 3] 检查认证/授权问题
    └── [聚合器] 合并所有安全报告
```

---

> **💡 Claude Code 代码生成提示词：**
> ```
> 用 Python + asyncio + Anthropic SDK 实现 Parallelization workflow：
> 1. 接收一段代码作为输入
> 2. 并发启动 3 个 LLM 调用，分别检查：SQL注入、XSS漏洞、认证问题
> 3. 用 asyncio.gather() 等待所有结果
> 4. 聚合输出一份统一的安全报告（JSON 格式）
> 5. 计算并打印总耗时 vs 串行耗时对比
> ```

---

### 2.3 Routing（路由分发）

**原理：** 先用 LLM 或规则对输入进行分类，再路由到对应的专用处理流程。

**示意图：**
```
User Input
    │
    ▼
[Classifier LLM]
    │
    ├── "技术问题" ──► Technical Support Flow
    ├── "账单问题" ──► Billing Flow
    ├── "投诉"     ──► Complaint Flow
    └── "其他"     ──► General Flow
```

**实际例子：客服系统路由**
```
用户消息: "我的支付失败了，钱扣了但订单没生成"

Classifier → { category: "payment_issue", urgency: "high" }

路由至 → Payment Issue Handler
    ├── 查询支付记录
    ├── 检查订单状态
    └── 生成退款建议
```

**多级路由：**
```
Level 1: 语言检测 → 中文 / 英文 / 其他
Level 2: 问题类型 → 技术 / 账单 / 投诉
Level 3: 紧急程度 → 高 / 中 / 低
    └──► 最终路由到具体 handler
```

---

> **💡 Claude Code 代码生成提示词：**
> ```
> 用 Python 实现一个 Routing workflow：
> 1. 定义 4 个处理器：technical_handler、billing_handler、complaint_handler、general_handler
> 2. 实现 classifier：调用 LLM 将用户输入分类，返回结构化 JSON { category, urgency, language }
> 3. 根据分类结果路由到对应 handler
> 4. 每个 handler 是独立的 prompt 模板
> 5. 写 3 个测试用例覆盖不同路由路径
> 使用 Pydantic 定义路由结果的数据结构
> ```

---

### 2.4 Map-Reduce（映射归约）

**原理：** 将大任务 Map 成多个小任务并行处理，再 Reduce 聚合结果。与 Parallelization 的区别在于 Map 阶段的拆分逻辑更动态。

**示意图：**
```
大任务（如：分析 100 份合同）
    │
    ▼
[Map 阶段] 动态拆分
    ├──► 处理合同 1 ──►┐
    ├──► 处理合同 2 ──►│
    ├──► ...           ├──► [Reduce] 聚合 ──► 最终报告
    └──► 处理合同 N ──►┘

```

**实际例子：Eduacan 题目批量生成**
```
输入: NCEA 数学考纲 PDF（50页）

Map:
    ├── 提取第 1-5 页知识点 → 生成 10 道题
    ├── 提取第 6-10 页知识点 → 生成 10 道题
    └── ...

Reduce:
    ├── 去重（相似题目合并）
    ├── 难度分级（Easy / Medium / Hard）
    └── 输出题库 JSON
```

---

> **💡 Claude Code 代码生成提示词：**
> ```
> 用 Python + asyncio 实现 Map-Reduce workflow：
> 1. 输入：一篇长文章（模拟 PDF 内容）
> 2. Map 阶段：将文章按段落拆分，每段并发调用 LLM 提取关键知识点
> 3. Reduce 阶段：
>    - 第一步 Reduce：对每段的知识点去重
>    - 第二步 Reduce：再次调用 LLM 将所有知识点整合成结构化大纲
> 4. 输出 Markdown 格式的知识大纲
> 5. 用 tqdm 显示处理进度
> ```

---

## 3. Multi-Agent 详解

### 3.1 Orchestrator + Workers 模式

**原理：** Orchestrator Agent 负责任务拆解与分发，Worker Agents 负责执行具体子任务，结果汇报给 Orchestrator 做最终整合。

**示意图：**
```
User Request: "帮我调研 LangGraph 并写一份技术报告"
    │
    ▼
┌─────────────────────────────┐
│     Orchestrator Agent      │
│  1. 拆解任务                │
│  2. 分配给各 Worker         │
│  3. 整合结果                │
└─────────────────────────────┘
    │           │           │
    ▼           ▼           ▼
[Research   [Code        [Writer
 Worker]     Worker]      Worker]
搜索资料    跑示例代码    整合成报告
```

**关键设计点：**

| 组件 | 职责 | 工具 |
|------|------|------|
| Orchestrator | 任务规划、分发、整合 | 无外部工具，依靠推理 |
| Research Worker | 信息收集 | web_search, read_url |
| Code Worker | 代码生成与执行 | write_file, run_code |
| Writer Worker | 文档撰写 | write_doc, format_md |

**实际例子：支付系统故障排查 Agent**
```
告警: "FTH 支付成功率下降 30%"
    │
    ▼
Orchestrator: 分析故障可能原因，制定排查计划
    │
    ├──► Log Analyst Worker: 分析最近 1 小时 MQ 日志
    ├──► DB Worker: 查询 Oracle 中失败交易记录
    └──► API Worker: 检查 Finacle API 响应状态
    │
    ▼
Orchestrator: 综合三方报告，定位根因，生成 RCA 报告
```

---

> **💡 Claude Code 代码生成提示词：**
> ```
> 用 Python + LangGraph 实现 Orchestrator + Workers Multi-Agent 系统：
> 1. Orchestrator Agent：接收用户任务，调用 LLM 拆解为子任务列表（JSON）
> 2. 实现 3 个 Worker Agent：
>    - research_worker：模拟搜索（返回固定 mock 数据即可）
>    - analysis_worker：对数据进行分析总结
>    - writer_worker：生成最终 Markdown 报告
> 3. Orchestrator 动态决定调用哪些 Worker，以及调用顺序
> 4. 用 LangGraph StateGraph 管理整个流程状态
> 5. 打印每个 Agent 的思考过程和执行结果
> 场景：用户输入"分析 Python 和 Go 的性能差异"
> ```

---

### 3.2 Peer-to-Peer 对等协作模式

**原理：** 多个 Agent 平等地相互通信，没有中央 Orchestrator。Agent 之间通过消息传递协作，形成对话驱动的协作网络。

**示意图：**
```
Agent A (Proposer)
    │
    │ "我提议方案 X"
    ▼
Agent B (Critic)
    │
    │ "方案 X 有个问题：..."
    ▼
Agent C (Resolver)
    │
    │ "综合 A 和 B 的意见，最终方案是..."
    ▼
Final Output
```

**实际例子：代码 Review 协作**
```
Agent 1 (Developer): 生成初版代码
    ↕
Agent 2 (Reviewer): 指出代码问题
    ↕
Agent 1 (Developer): 修改代码
    ↕
Agent 3 (Tester): 生成测试用例，发现 Bug
    ↕
Agent 1 (Developer): 修复 Bug
    ↕
Agent 3 (Tester): 测试通过 ✓
```

**AutoGen 风格的 Agent 对话：**
```python
# 概念示意
developer.send("这是我写的支付处理代码：...", to=reviewer)
reviewer.send("发现问题：没有处理幂等性", to=developer)
developer.send("已修复，加入了 idempotency_key 检查", to=tester)
tester.send("测试通过，覆盖率 95%", to=all)
```

---

> **💡 Claude Code 代码生成提示词：**
> ```
> 用 Python 实现一个 Peer-to-Peer Multi-Agent 代码审查系统：
> 1. 定义 3 个 Agent 类，每个有自己的 system_prompt 和角色：
>    - DeveloperAgent：负责编写和修改代码
>    - ReviewerAgent：负责发现代码问题（安全/性能/可读性）
>    - TesterAgent：负责生成测试用例并验证
> 2. 实现 Agent 间消息传递机制（Message 类包含 from_agent, to_agent, content）
> 3. 实现对话循环：最多 5 轮，直到 Tester 说 "APPROVED"
> 4. 场景：输入一段有 bug 的 Python 支付处理代码，让 3 个 Agent 协作修复
> 5. 打印完整对话历史
> ```

---

### 3.3 Human-in-the-Loop（人机协作）

**原理：** 在 Agent 执行过程中，在关键节点暂停并等待人类确认或输入，再继续执行。

**适用场景：**
- 高风险操作（删除数据、发送邮件、资金操作）
- 模糊指令需要澄清
- 需要专业判断的决策点

**示意图：**
```
Agent 执行中...
    │
    ▼
[关键节点] 需要确认
    │
    ▼
⏸️  PAUSE ──► 通知人类
                │
                ├── ✅ Approve ──► 继续执行
                ├── ✏️  Modify  ──► 修改参数后继续
                └── ❌ Reject   ──► 终止或重新规划
```

**实际例子：Finacle API 调用前确认**
```
Agent: "我准备调用 Finacle initiateTransaction API"
      "参数：{ amount: 50000, currency: NZD, beneficiary: XXX }"

⏸️ PAUSE → 等待人工确认

操作员: ✅ Confirm

Agent: 继续执行 API 调用
```

**LangGraph Interrupt 实现思路：**
```python
# LangGraph 中用 interrupt 实现
def risky_operation_node(state):
    # 在执行前请求人工确认
    human_approval = interrupt({
        "action": "initiate_payment",
        "amount": state["amount"],
        "message": "请确认此笔支付操作"
    })
    if human_approval == "approved":
        return execute_payment(state)
    else:
        return {"status": "rejected"}
```

---

> **💡 Claude Code 代码生成提示词：**
> ```
> 用 Python + LangGraph 实现 Human-in-the-Loop Agent：
> 1. 构建一个支付审批 Agent，流程：
>    - 解析用户的支付请求（金额、收款方、备注）
>    - 金额 < 1000 → 自动审批
>    - 金额 >= 1000 → 暂停，等待人工审批
>    - 人工输入 approve/reject/modify
>    - 根据人工决定继续或终止
> 2. 使用 LangGraph 的 interrupt() 实现暂停
> 3. 使用 MemorySaver checkpointer 保存状态，支持从断点恢复
> 4. 命令行交互界面
> 5. 展示 approve 和 reject 两个完整流程
> ```

---

## 4. Workflow vs Multi-Agent 对比

### 全面对比表

| 维度 | Workflow | Multi-Agent |
|------|----------|-------------|
| **控制流** | 开发者预定义 | LLM 动态决定 |
| **可预测性** | 高，路径固定 | 低，路径动态 |
| **可解释性** | 强，每步清晰 | 弱，推理过程黑盒 |
| **调试难度** | 容易 | 困难 |
| **成本** | 可精确估算 | 难以预测 |
| **延迟** | 可优化（并行） | 通常更高 |
| **适合任务** | 结构化、重复性 | 复杂、开放性 |
| **容错性** | 需手动设计重试 | Agent 可自主重试 |
| **扩展性** | 修改流程需改代码 | 调整 prompt 即可 |
| **典型框架** | Celery, Temporal | LangGraph, CrewAI |

### 决策树

```
任务路径是否可以提前完全定义？
    │
    ├── 是 → Workflow
    │         │
    │         ├── 步骤是否可并行？→ 是 → Parallelization
    │         ├── 输入类型多样？→ 是 → Routing
    │         ├── 步骤顺序依赖？→ 是 → Prompt Chaining
    │         └── 大量同类任务？→ 是 → Map-Reduce
    │
    └── 否 → Agent / Multi-Agent
              │
              ├── 单一复杂任务？→ Single Agent (ReAct)
              ├── 需要专业分工？→ Orchestrator + Workers
              ├── 需要相互审查？→ Peer-to-Peer
              └── 有高风险操作？→ Human-in-the-Loop
```

### 什么时候 Workflow 比 Agent 更好？

```
✅ 用 Workflow 当：
- 流程固定（如：每日报表生成）
- 成本敏感（精确控制 token 消耗）
- 合规要求（每步需要审计日志）
- 延迟敏感（需要 SLA 保证）

✅ 用 Agent 当：
- 任务路径未知（需要 LLM 探索）
- 任务复杂度超出单次 context window
- 需要自主使用多种工具
- 处理意外情况和边缘 case
```

---

## 5. 框架选型指南

### LangGraph 核心概念速查

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. 定义 State
class AgentState(TypedDict):
    messages: list
    current_task: str
    result: str

# 2. 定义 Nodes（每个 node 是一个函数）
def researcher(state: AgentState):
    # 调用 LLM 搜索信息
    return {"result": "searched info..."}

def writer(state: AgentState):
    # 调用 LLM 写报告
    return {"result": "written report..."}

# 3. 条件路由函数
def should_continue(state: AgentState):
    if state["result"] == "need_more_info":
        return "researcher"  # 回到 researcher
    return "writer"          # 继续 writer

# 4. 构建 Graph
graph = StateGraph(AgentState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge(START, "researcher")
graph.add_conditional_edges("researcher", should_continue)
graph.add_edge("writer", END)

app = graph.compile()
```

### CrewAI 核心概念速查

```python
from crewai import Agent, Task, Crew

# 1. 定义 Agents（有角色、目标、工具）
researcher = Agent(
    role="Senior Researcher",
    goal="找到最新的 AI 技术趋势",
    tools=[search_tool, read_tool]
)

writer = Agent(
    role="Technical Writer",
    goal="将研究结果写成清晰的报告"
)

# 2. 定义 Tasks（有描述、期望输出、执行者）
research_task = Task(
    description="调研 LangGraph 的最新特性",
    expected_output="结构化的调研笔记",
    agent=researcher
)

write_task = Task(
    description="基于调研笔记写技术报告",
    expected_output="Markdown 格式报告",
    agent=writer
)

# 3. 组建 Crew 并执行
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process="sequential"  # 或 "hierarchical"
)

result = crew.kickoff()
```

### Temporal 工作流概念速查（适合支付系统）

```python
# Temporal 的核心价值：Workflow 状态持久化，进程崩溃也能恢复
from temporalio import activity, workflow

# 1. 定义 Activity（实际的业务操作）
@activity.defn
async def call_finacle_api(params: dict) -> dict:
    # 调用 Finacle，失败会自动重试
    return await finacle_client.initiate_transaction(params)

# 2. 定义 Workflow（编排逻辑，持久化执行）
@workflow.defn
class PaymentWorkflow:
    @workflow.run
    async def run(self, payment_request: dict) -> dict:
        # Step 1: 验证
        validated = await workflow.execute_activity(
            validate_payment, payment_request,
            retry_policy=RetryPolicy(max_attempts=3)
        )
        # Step 2: 扣款（exactly-once 语义）
        result = await workflow.execute_activity(
            call_finacle_api, validated,
            retry_policy=RetryPolicy(max_attempts=1)  # 资金操作只试一次
        )
        return result
```

---

## 6. 综合实战案例

### 案例：Eduacan 智能题目生成系统

结合 Workflow + Multi-Agent 的混合架构：

```
用户上传 NCEA 数学考纲 PDF
    │
    ▼
┌─────────────────────────────────────────────┐
│              Workflow Layer                  │
│                                             │
│  [PDF Parser] → [Chunk Splitter]            │
│       ↓                                     │
│  [Map] 并发提取各章节知识点                  │
│       ↓                                     │
│  [Reduce] 整合知识图谱                       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           Multi-Agent Layer                  │
│                                             │
│  Orchestrator: 根据知识图谱规划出题策略       │
│      │                                      │
│      ├── Question Designer Agent            │
│      │   └── 生成题目初稿                   │
│      ├── Difficulty Evaluator Agent         │
│      │   └── 评估难度，标注 Easy/Med/Hard    │
│      └── Quality Checker Agent             │
│          └── 检查题目准确性，拒绝低质量题目  │
└─────────────────────────────────────────────┘
    │
    ▼
题库 JSON → 存入 PostgreSQL + pgvector
```

**混合架构的设计原则：**
```
结构清晰的部分 → Workflow（PDF解析、向量存储）
需要判断的部分 → Agent（出题策略、质量控制）
```

---

### 案例：FTH 支付异常处理 Agent

```
MQ 收到支付响应（AM04 - 余额不足）
    │
    ▼
┌─────────────────────────────────────────────┐
│         Routing Workflow                    │
│  AM04 → Insufficient Funds Handler          │
│  AC01 → Invalid Account Handler             │
│  DUPL → Duplicate Payment Handler           │
└─────────────────────────────────────────────┘
    │ (AM04 路由)
    ▼
┌─────────────────────────────────────────────┐
│      Human-in-the-Loop Agent               │
│                                             │
│  1. 查询客户账户余额历史                     │
│  2. 判断：临时不足 or 账户问题？             │
│  3. 金额 > $10,000 → 暂停，等待人工确认      │
│  4. 生成客户通知建议（短信/邮件）            │
└─────────────────────────────────────────────┘
    │
    ▼
操作员确认 → 执行重试 or 退款
```

---

> **💡 Claude Code 综合代码生成提示词：**
> ```
> 用 Python + LangGraph 实现一个综合示例，展示 Workflow + Multi-Agent 混合架构：
>
> 场景：自动化技术调研报告生成系统
>
> 1. Workflow 部分（固定流程）：
>    - 接收用户输入的技术关键词（如 "LangGraph"）
>    - 并行搜索 3 个来源（mock 数据即可）：官方文档、GitHub、技术博客
>    - Map-Reduce 整合搜索结果为结构化知识库
>
> 2. Multi-Agent 部分：
>    - Orchestrator Agent: 分析知识库，规划报告结构
>    - Analyst Agent: 深度分析技术优缺点
>    - Comparator Agent: 与竞品对比（如 CrewAI）
>    - Writer Agent: 整合输出 Markdown 报告
>
> 3. 技术要求：
>    - 用 LangGraph StateGraph 管理整体流程
>    - 用 TypedDict 定义完整的 State 结构
>    - 包含错误处理和重试逻辑
>    - 每个 Agent 的思考过程可观测（打印到 console）
>    - 最终输出保存为 report.md
>
> 4. 代码结构：
>    workflow/
>    ├── main.py          # 入口
>    ├── state.py         # State 定义
>    ├── workflow_nodes.py # Workflow 节点
>    ├── agent_nodes.py   # Agent 节点
>    └── graph.py         # Graph 组装
> ```

---

## 附录：关键术语速查

| 术语 | 含义 |
|------|------|
| **Agent Loop** | Agent 的 Think→Act→Observe 循环 |
| **Tool Call** | Agent 调用外部工具（API/DB/搜索）|
| **Handoff** | 一个 Agent 将任务移交给另一个 |
| **Interrupt** | LangGraph 中暂停 Agent 等待人工输入 |
| **Checkpointer** | 持久化 Agent 状态，支持断点续跑 |
| **ReAct** | Reasoning + Acting，最经典的 Agent 模式 |
| **State** | LangGraph 中流转于各节点间的数据 |
| **Reducer** | 定义 State 如何被更新的函数 |
| **Gate Check** | Workflow 中的质量验证节点 |
| **Exactly-once** | 任务只执行一次，即使重试（Temporal 的核心保证）|
