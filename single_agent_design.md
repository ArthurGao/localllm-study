# Single Agent 设计与实现深度解析
## 基于 LangChain + LangGraph

> 本文档配合 Claude Code 生成学习代码。每节末尾附有精准的代码生成提示词。

---

## 目录

1. [Agent 是什么：从零理解](#1-agent-是什么从零理解)
2. [Agent 的核心组成](#2-agent-的核心组成)
3. [ReAct 模式：Agent 的思维方式](#3-react-模式agent-的思维方式)
4. [Tools 工具设计](#4-tools-工具设计)
5. [Memory 记忆管理](#5-memory-记忆管理)
6. [LangGraph 实现 Agent](#6-langgraph-实现-agent)
7. [State 设计模式](#7-state-设计模式)
8. [错误处理与重试](#8-错误处理与重试)
9. [Agent 可观测性](#9-agent-可观测性)
10. [完整实战案例](#10-完整实战案例)

---

## 1. Agent 是什么：从零理解

### 传统程序 vs Agent

```
传统程序：
Input → [固定逻辑代码] → Output
开发者决定：做什么、怎么做、什么顺序

Agent：
Input → [LLM 推理] → 决定用什么工具 → 执行 → 观察结果 → 再推理 → Output
LLM 决定：做什么、怎么做、什么顺序
```

### Agent 的本质

**Agent = LLM + Tools + Loop**

```
+-----------------------------------------+
|              Agent Loop                 |
|                                         |
|   +----------+                          |
|   |  Think   |  LLM 分析当前状态        |
|   | (Reason) |  决定下一步行动           |
|   +----+-----+                          |
|        |                                |
|        v                                |
|   +----------+                          |
|   |   Act    |  调用 Tool / 输出答案     |
|   | (Action) |                          |
|   +----+-----+                          |
|        |                                |
|        v                                |
|   +----------+                          |
|   | Observe  |  接收 Tool 执行结果       |
|   |  (Obs)   |  更新上下文              |
|   +----+-----+                          |
|        |                                |
|        +---- 继续循环 or 结束 -----------+
+-----------------------------------------+
```

### Agent 何时结束？

```
LLM 每轮推理后决定：
    +-- 需要更多信息 --> 调用工具 --> 继续循环
    +-- 信息已足够  --> 直接回答 --> 结束循环
```

---

## 2. Agent 的核心组成

```
+------------------------------------------------+
|                    Agent                       |
|                                                |
|  +-------------+   +----------------------+   |
|  |    Brain    |   |        Tools         |   |
|  |   (LLM)     |-->|  web_search          |   |
|  |             |   |  calculator          |   |
|  |  推理、规划  |   |  read_file           |   |
|  |  决策、生成  |   |  write_file          |   |
|  +-------------+   |  call_api            |   |
|         |          +----------------------+   |
|         |                                      |
|  +------v-------------------------------------+|
|  |              Memory                        ||
|  |  Short-term: 当前对话 messages              ||
|  |  Long-term:  向量数据库 / 外部存储           ||
|  +--------------------------------------------+|
|         |                                      |
|  +------v-------------------------------------+|
|  |              State                         ||
|  |  当前任务状态、中间结果、执行历史             ||
|  +--------------------------------------------+|
+------------------------------------------------+
```

### 各组件职责

| 组件 | 职责 | LangGraph 对应 |
|------|------|---------------|
| Brain (LLM) | 推理、规划、生成 | 节点内的 `llm.invoke()` |
| Tools | 执行具体操作 | `@tool` 装饰器定义的函数 |
| Short-term Memory | 当前对话上下文 | `state["messages"]` |
| Long-term Memory | 跨会话持久记忆 | `MemorySaver` / 向量DB |
| State | 全局流转数据 | `TypedDict` 定义的 State |

---

## 3. ReAct 模式：Agent 的思维方式

ReAct = **Re**asoning + **Act**ing

这是最经典的 Agent 模式：LLM 交替进行"推理"和"行动"。

### ReAct 完整执行示例

```
用户: "LangGraph 最新版本是多少？比上个版本新增了什么？"

---------------------------------------------------
Thought 1:
  "我需要搜索 LangGraph 最新版本"

Action 1:   web_search
Input 1:    "LangGraph latest version changelog 2025"
Observation 1: "LangGraph 0.2.x, new features: subgraph streaming..."

---------------------------------------------------
Thought 2:
  "我得到了最新版本，现在需要找上个版本对比"

Action 2:   web_search
Input 2:    "LangGraph 0.1.x features"
Observation 2: "0.1.x features: basic StateGraph, MemorySaver..."

---------------------------------------------------
Thought 3:
  "信息足够，可以总结回答了"

Final Answer:
  "LangGraph 最新版本是 0.2.x，新增了 subgraph streaming..."
---------------------------------------------------
```

### LangChain 中的实现方式

```python
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic

# 1. LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# 2. 从 Hub 拉取 ReAct prompt 模板
prompt = hub.pull("hwchase17/react")

# 3. 创建 ReAct agent
agent = create_react_agent(llm, tools, prompt)

# 4. 包装成 executor（负责循环执行）
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "LangGraph 最新版本是多少？"})
```

### Tool Calling 模式（现代推荐方式）

相比文本解析的 ReAct，Tool Calling 通过结构化 JSON 调用工具，更稳定：

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

# LLM 返回结构化的工具调用，而不是文本：
# {"tool": "web_search", "args": {"query": "LangGraph version"}}
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### ReAct vs Tool Calling 对比

| 维度 | ReAct（文本解析） | Tool Calling（结构化）|
|------|-----------------|----------------------|
| 工具调用方式 | 解析 LLM 输出文本 | 结构化 JSON |
| 稳定性 | 低（依赖格式正确）| 高（原生支持）|
| 可读性 | 高（思维链可见）| 中 |
| 模型要求 | 任何模型 | 需模型支持 Function Calling |
| 推荐场景 | 学习/调试 | 生产环境 |

---

> **💡 Claude Code 提示词 — ReAct Agent：**
> ```
> 用 Python + LangChain 实现两个版本的 Agent 并对比：
>
> 版本 1：传统 ReAct（文本解析）
> - 使用 create_react_agent + hub.pull("hwchase17/react")
> - 定义 2 个 mock 工具：web_search（返回假数据）、calculator（真实计算）
> - verbose=True 展示完整思维链
>
> 版本 2：Tool Calling（现代方式）
> - 使用 create_tool_calling_agent
> - 相同的工具集
> - 对比两个版本的输出格式差异
>
> 测试问题：
> "2024年 Python 最新版本号是多少？把版本号的主版本数字乘以 100 等于多少？"
> （需要先 web_search 获取版本号，再用 calculator 计算）
>
> 使用 claude-3-5-sonnet-20241022，打印每步的 thought/action/observation
> ```

---

## 4. Tools 工具设计

Tools 是 Agent 感知和操作外部世界的唯一途径。工具设计的好坏直接决定 Agent 能力上限。

### 4.1 用 @tool 定义工具

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 方式 1：简单装饰器
@tool
def web_search(query: str) -> str:
    """
    在互联网上搜索最新信息。
    适用场景：查询实时数据、新闻、软件版本、价格信息。
    不适用：数学计算、代码执行、文件操作。
    """
    return f"搜索结果：关于 {query} 的信息..."

# 方式 2：带 Schema 的装饰器（更精确的参数描述）
class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，例如 '2 + 3 * 4'")

@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """计算数学表达式。只用于纯数学计算，不用于其他操作。"""
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# 方式 3：StructuredTool（最完整的控制）
from langchain_core.tools import StructuredTool

def fetch_payment_status(transaction_id: str, include_history: bool = False) -> dict:
    """查询支付交易状态"""
    return {"status": "completed", "amount": 1000}

payment_tool = StructuredTool.from_function(
    func=fetch_payment_status,
    name="fetch_payment_status",
    description="查询 FTH 支付系统中的交易状态。需要提供 transaction_id。",
)
```

### 4.2 工具描述的重要性

LLM 完全依靠 **name** 和 **description** 决定何时调用工具。描述写得好坏极为关键：

```python
# ❌ 差的工具描述
@tool
def search(q: str) -> str:
    """搜索"""  # 太模糊，LLM 不知道何时该用

# ✅ 好的工具描述
@tool
def web_search(query: str) -> str:
    """
    在互联网上搜索最新信息。
    适用场景：查询实时数据、最新新闻、软件版本、价格信息。
    不适用：数学计算、代码执行、文件操作。
    参数 query：搜索关键词，建议使用英文以获得更好结果。
    """
```

### 4.3 工具的分类设计

```
工具按操作类型分类：

读取类（Read）：
    +-- web_search       搜索网页
    +-- read_file        读取文件内容
    +-- query_database   查询数据库
    +-- fetch_api        调用只读 API

写入类（Write）：
    +-- write_file       写入文件
    +-- send_email       发送邮件
    +-- update_database  更新数据库
    +-- call_api         调用写入 API

计算类（Compute）：
    +-- calculator       数学计算
    +-- run_python       执行 Python 代码
    +-- parse_json       解析数据结构
```

**设计原则：**
- 每个工具职责单一（Single Responsibility）
- 工具要有清晰的成功/失败返回
- 写入类工具要考虑幂等性
- 错误时返回描述性错误信息，让 LLM 能判断下一步

### 4.4 工具的错误处理

```python
@tool
def query_oracle_db(sql: str) -> str:
    """查询 Oracle 数据库，返回查询结果。仅支持 SELECT 语句。"""
    try:
        if not sql.strip().upper().startswith("SELECT"):
            return "错误：只允许 SELECT 查询"

        result = db.execute(sql)
        return str(result)

    except ConnectionError:
        return "错误：数据库连接失败，请稍后重试"
    except Exception as e:
        return f"查询失败：{str(e)}"
```

---

> **💡 Claude Code 提示词 — Tools 设计：**
> ```
> 用 Python + LangChain 设计并测试一套完整的工具集：
>
> 实现以下 4 个工具：
> 1. web_search(query: str)
>    - 根据 query 关键词返回不同的 mock 数据
>    - 预设关键词映射：{"python" -> "Python 3.12.x...", "langgraph" -> "0.2.x..."}
>
> 2. calculator(expression: str)
>    - 用 ast.literal_eval 安全计算数学表达式
>    - 处理除零、语法错误等异常
>
> 3. read_file(filepath: str)
>    - 读取本地文件内容
>    - 处理文件不存在、权限错误
>
> 4. get_current_time(timezone: str = "UTC")
>    - 返回指定时区的当前时间（用 pytz）
>
> 要求：
> - 每个工具用 Pydantic BaseModel 定义参数 schema
> - 每个工具有完整 docstring（含适用场景）
> - 写单独的测试函数验证每个工具
> - 最后绑定到 Claude LLM，测试 LLM 是否能正确选择工具
>
> 测试问题：
> - "现在北京时间是几点？"   -> get_current_time
> - "3 的 10 次方是多少？"  -> calculator
> - "读取 ./README.md"      -> read_file
> ```

---

## 5. Memory 记忆管理

### 5.1 记忆的层次

```
+--------------------------------------------+
|             记忆层次结构                    |
|                                            |
|  In-Context Memory（上下文内记忆）           |
|  +--------------------------------------+  |
|  |  messages = [                        |  |
|  |    HumanMessage("你好"),             |  |
|  |    AIMessage("你好！有什么可以帮你"), |  |
|  |    HumanMessage("计算 2+2"),         |  |
|  |    ToolMessage("4"),                 |  |
|  |    AIMessage("2+2 = 4"),            |  |
|  |  ]                                   |  |
|  +--------------------------------------+  |
|  特点：速度快，受 context window 限制       |
|                                            |
|  External Memory（外部持久记忆）             |
|  +--------------------------------------+  |
|  |  MemorySaver（SQLite/内存）           |  |
|  |  PostgreSQL checkpointer             |  |
|  |  向量数据库（pgvector）               |  |
|  +--------------------------------------+  |
|  特点：跨会话持久，可存储大量历史            |
+--------------------------------------------+
```

### 5.2 Short-term Memory：对话历史

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing import Annotated

class AgentState(TypedDict):
    # add_messages reducer：新消息 append，相同 id 的消息会更新
    messages: Annotated[list[BaseMessage], add_messages]
```

**对话历史裁剪（防止超出 context window）：**

```python
from langchain_core.messages import trim_messages

# 按 token 数裁剪
trimmer = trim_messages(
    max_tokens=4000,
    strategy="last",       # 保留最近的消息
    token_counter=llm,
    include_system=True,   # 保留 system message
    allow_partial=False,
)

# 按消息数量裁剪
def trim_to_last_n(messages: list, n: int = 10) -> list:
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs  = [m for m in messages if not isinstance(m, SystemMessage)]
    return system_msgs + other_msgs[-n:]
```

### 5.3 Long-term Memory：跨会话持久化

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# thread_id 区分不同用户会话
config_alice = {"configurable": {"thread_id": "alice_session_1"}}
config_bob   = {"configurable": {"thread_id": "bob_session_1"}}

# Alice 的对话
app.invoke({"messages": [HumanMessage("我叫 Arthur，在 Westpac 做支付系统")]}, config_alice)
app.invoke({"messages": [HumanMessage("我在哪个公司工作？")]}, config_alice)
# Agent 回答："你在 Westpac 工作"（记住了上文）

# Bob 的对话（独立记忆）
app.invoke({"messages": [HumanMessage("我在哪个公司工作？")]}, config_bob)
# Agent 回答："我不知道你在哪个公司"
```

### 5.4 记忆的读写时机

```
请求进入
    |
    v
[读取记忆] <- checkpointer 加载历史 state
    |
    v
[执行节点] Agent 推理 + 工具调用
    |
    v
[写入记忆] <- checkpointer 保存最新 state
    |
    v
返回响应
```

---

> **💡 Claude Code 提示词 — Memory 管理：**
> ```
> 用 Python + LangGraph 实现带完整记忆管理的对话 Agent：
>
> 1. Short-term Memory：
>    - 用 add_messages reducer 管理对话历史
>    - 超过 10 条消息时，保留 system message + 最近 8 条
>    - 每轮打印消息裁剪前后的数量
>
> 2. Long-term Memory（跨会话）：
>    - 使用 MemorySaver 作为 checkpointer
>    - 支持 thread_id 区分不同用户
>
> 3. 对话演示场景：
>    用户 alice（thread: "alice"）:
>      轮1: "我是 Alice，我在 Westpac 做支付系统开发"
>      轮2: "我做什么工作？"        (Agent 应能回答)
>      轮3: "我在哪个公司？"        (Agent 应能回答)
>
>    用户 bob（thread: "bob"）:
>      轮1: "我在哪个公司工作？"    (Agent 应回答不知道)
>
> 4. 代码加详细注释，说明每个记忆操作发生的时机
> 使用 claude-3-5-sonnet-20241022
> ```

---

## 6. LangGraph 实现 Agent

LangGraph 将 Agent 建模为**有状态的图（Graph）**，是实现生产级 Agent 的推荐框架。

### 6.1 LangGraph 核心概念

```
Graph = Nodes + Edges + State

State：流转于各节点间的数据（TypedDict）
Nodes：执行具体逻辑的函数（接收 State，返回 State 更新）
Edges：决定执行顺序（普通边 or 条件边）

      START
        |
        v
   [agent node]  <- LLM 推理，决定是否调用工具
        |
        v (条件边)
   +----+----+
   |         |
   v         v
[tools]    END
  node
   |
   +-------> [agent node]  <- 循环回去继续推理
```

### 6.2 标准 ReAct Agent 的 LangGraph 实现

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Step 1: 定义 State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Step 2: 定义工具
@tool
def web_search(query: str) -> str:
    """搜索网络信息"""
    return f"搜索到关于 {query} 的信息..."

tools = [web_search, calculator]

# Step 3: 绑定工具到 LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
llm_with_tools = llm.bind_tools(tools)

# Step 4: 定义 Agent 节点
def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Step 5: 构建 Graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    tools_condition,   # 检查最后一条消息是否包含 tool_calls
    # 返回 "tools" 或 END
)
graph.add_edge("tools", "agent")  # 工具执行后回到 agent

# Step 6: 编译并运行
app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session_1"}}

result = app.invoke(
    {"messages": [HumanMessage("LangGraph 最新版本是多少？")]},
    config
)
```

### 6.3 执行流程可视化

```
用户输入："LangGraph 最新版本？"
    |
    v
[agent node] 第 1 轮
  LLM 决定：需要搜索
  输出：AIMessage(tool_calls=[web_search("LangGraph latest")])
    |
    v (tools_condition -> "tools")
[tools node]
  执行：web_search("LangGraph latest")
  输出：ToolMessage("0.2.x released...")
    |
    v
[agent node] 第 2 轮
  LLM 决定：有足够信息，直接回答
  输出：AIMessage("LangGraph 最新版本是 0.2.x")
    |
    v (tools_condition -> END)
返回结果
```

### 6.4 自定义条件路由

```python
def my_router(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # 有工具调用 -> 去执行工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # LLM 判断需要人工介入
    if "需要人工确认" in last_message.content:
        return "human_review"

    return END

graph.add_conditional_edges(
    "agent",
    my_router,
    {
        "tools":        "tools",
        "human_review": "human_review_node",
        END:            END
    }
)
```

---

> **💡 Claude Code 提示词 — LangGraph Agent：**
> ```
> 用 Python + LangGraph 从零实现一个完整的 ReAct Agent：
>
> 1. State 定义：
>    - messages: 对话历史（add_messages reducer）
>    - tool_call_count: 工具调用次数（int，默认 0）
>    - final_answer: 最终答案（str，可为 None）
>
> 2. 工具集（3个）：
>    - web_search(query): 根据关键词返回 mock 数据
>      预设映射：{"python" -> "Python 3.12", "langgraph" -> "0.2.x"}
>    - calculator(expression): 真实数学计算
>    - get_time(): 返回当前时间
>
> 3. 节点：
>    - agent_node: LLM 推理，每次调用 tool_call_count + 1
>    - tools_node: ToolNode 执行工具
>    - 安全限制：tool_call_count > 5 时强制结束
>
> 4. Graph：START -> agent -> (条件) -> tools -> agent (循环) -> END
>
> 5. 运行 3 个测试：
>    a. "现在几点了？再加 100 分钟后是几点？" (get_time + calculator)
>    b. "Python 版本号的主版本数字乘以 100 等于多少？" (web_search + calculator)
>    c. "你好" (无需工具，直接回答)
>
> 6. 每步打印：当前节点名、messages 数量、tool_call_count
> 使用 claude-3-5-sonnet-20241022
> ```

---

## 7. State 设计模式

State 是 LangGraph 的核心，所有节点通过 State 通信。

### 7.1 基础 State 结构

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class BasicAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### 7.2 按业务需求扩展 State

```python
class AdvancedAgentState(TypedDict):
    # 对话历史
    messages:        Annotated[list[BaseMessage], add_messages]

    # 任务追踪
    task_description: str
    current_step:     int
    max_steps:        int     # 防无限循环
    status:           str     # "running" | "completed" | "error"

    # 中间结果
    search_results:   list[str]
    calculated_values: dict

    # 错误处理
    error_message:    Optional[str]
    retry_count:      int

    # 元数据
    user_id:          str
    session_start:    str
```

### 7.3 Reducer：控制 State 如何更新

```python
import operator
from typing import Annotated

# add_messages：智能合并消息列表（相同 id 的消息会被更新）
messages: Annotated[list, add_messages]

# operator.add：列表追加
# 节点返回 ["result1"] -> State 变成 [...existing, "result1"]
search_results: Annotated[list[str], operator.add]

# 自定义 reducer：只保留最大值
def keep_max(existing: int, new: int) -> int:
    return max(existing, new)

best_score: Annotated[int, keep_max]

# 无 reducer（直接覆盖，默认行为）
current_step: int
```

### 7.4 State 更新的正确姿势

```python
# 节点只返回需要更新的字段，其余自动保留
def agent_node(state: AdvancedAgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages":     [response],                        # add_messages append
        "current_step": state["current_step"] + 1,        # 覆盖
        # 其他字段无需返回，自动保留原值
    }

# 不要这样：返回完整 State 会导致 messages 被替换而非 append
# def wrong_node(state): state["current_step"] += 1; return state  # 错误！
```

---

> **💡 Claude Code 提示词 — State 设计：**
> ```
> 用 Python + LangGraph 实现展示 State 设计最佳实践的"研究助手" Agent：
>
> State 包含：
> - messages: 对话历史
> - research_notes: 收集到的信息（list，operator.add 追加）
> - tool_calls_log: 工具调用日志（list，记录工具名+结果摘要）
> - iteration_count: 迭代次数（int）
> - confidence_score: 答案置信度 0-100（自定义 keep_max reducer）
> - final_report: 最终报告（str，可为 None）
>
> 3 个节点：
> 1. research_node: 调用 web_search，更新 research_notes 和 tool_calls_log
> 2. evaluate_node: LLM 评估信息是否充分，更新 confidence_score（0-100）
> 3. report_node: 生成最终 Markdown 报告
>
> 路由逻辑：
> - confidence_score < 70 -> 继续 research（最多 3 次）
> - confidence_score >= 70 -> 生成 report -> END
>
> 每步打印完整 State 快照，演示 Reducer 的实际效果
> ```

---

## 8. 错误处理与重试

### 8.1 工具调用失败处理

```python
from langchain_core.messages import ToolMessage

def safe_tool_node(state: AgentState) -> dict:
    """带错误隔离的工具执行节点"""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        try:
            tool = tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        except Exception as e:
            # 不崩溃，把错误作为工具结果返回给 LLM
            tool_results.append(
                ToolMessage(
                    content=f"工具执行失败：{str(e)}。请尝试其他方式。",
                    tool_call_id=tool_call["id"]
                )
            )

    return {"messages": tool_results}
```

### 8.2 无限循环防护

```python
def check_loop_limit(state: AgentState) -> str:
    if state["iteration_count"] >= state["max_iterations"]:
        return "force_end"

    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return END

def force_end_node(state: AgentState) -> dict:
    return {
        "messages": [AIMessage("已达到最大迭代次数，任务终止。")],
        "status": "timeout"
    }
```

### 8.3 LLM 调用重试

```python
from tenacity import retry, stop_after_attempt, wait_exponential

# 方式 1：tenacity 装饰器
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def call_llm_with_retry(messages):
    return llm.invoke(messages)

# 方式 2：LangChain 原生 with_retry
llm_with_retry = llm.with_retry(stop_after_attempt=3)

# 方式 3：主模型失败时切换备用模型（Fallback）
from langchain_openai import ChatOpenAI

primary_llm  = ChatAnthropic(model="claude-3-5-sonnet-20241022")
fallback_llm = ChatOpenAI(model="gpt-4o")

llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])
```

---

> **💡 Claude Code 提示词 — 错误处理：**
> ```
> 用 Python + LangGraph 实现健壮的 Agent，演示完整错误处理：
>
> 1. 不稳定工具：
>    - flaky_search(query): 随机 30% 概率抛出 ConnectionError
>    - 工具内部用 tenacity 重试（最多 3 次，指数退避）
>
> 2. 工具节点错误隔离：
>    - 实现 safe_tool_node（替代默认 ToolNode）
>    - 工具失败时返回描述性错误的 ToolMessage，不崩溃
>    - LLM 收到错误后自主决定换策略
>
> 3. 循环保护：
>    - State 中加 iteration_count
>    - 超过 5 次 -> 强制结束
>
> 4. LLM 重试：
>    - with_retry(stop_after_attempt=2)
>
> 5. 运行 3 种演示：
>    a. 正常流程（工具成功）
>    b. 工具偶发失败后自动恢复
>    c. 触发循环上限
>
> 每步打印带时间戳的详细日志
> ```

---

## 9. Agent 可观测性

### 9.1 流式输出

```python
# 流式获取每个节点的执行结果
for chunk in app.stream(
    {"messages": [HumanMessage("搜索 LangGraph 信息")]},
    config,
    stream_mode="updates"  # 每个节点更新时推送
):
    node_name, state_update = list(chunk.items())[0]
    print(f"\n── 节点：{node_name} ──")
    if "messages" in state_update:
        last_msg = state_update["messages"][-1]
        print(f"   消息类型：{type(last_msg).__name__}")
        print(f"   内容：{str(last_msg.content)[:100]}")

# 打字机效果（token 流）
for chunk in app.stream(input_data, config, stream_mode="messages"):
    if chunk[1].get("langgraph_node") == "agent":
        print(chunk[0].content, end="", flush=True)
```

### 9.2 节点执行日志

```python
import time
from functools import wraps

def log_node(func):
    """节点日志装饰器"""
    @wraps(func)
    def wrapper(state):
        start = time.time()
        print(f"\n[{func.__name__}] 开始")
        print(f"  messages: {len(state.get('messages', []))} 条")

        result = func(state)

        elapsed = time.time() - start
        print(f"[{func.__name__}] 完成 ({elapsed:.2f}s)")
        return result
    return wrapper

@log_node
def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### 9.3 执行历史与状态快照

```python
# 查看当前状态
current_state = app.get_state(config)
print("消息数：", len(current_state.values["messages"]))
print("下一步：", current_state.next)

# 遍历完整执行历史
history = list(app.get_state_history(config))
print(f"\n共 {len(history)} 个检查点：")
for snapshot in history:
    step = snapshot.metadata.get("step", "?")
    node = snapshot.next
    msg_count = len(snapshot.values.get("messages", []))
    print(f"  步骤 {step}: 节点={node}, 消息数={msg_count}")
```

---

> **💡 Claude Code 提示词 — 可观测性：**
> ```
> 用 Python + LangGraph 实现带完整可观测性的 Agent：
>
> 1. 日志装饰器：
>    - 装饰所有节点函数
>    - 打印：进入时间、State 摘要（消息数/工具调用数）、耗时
>
> 2. 流式输出：
>    - stream_mode="updates" 实时显示每个节点执行
>    - stream_mode="messages" 对 agent 节点实现打字机效果
>
> 3. 执行后打印摘要报告：
>    ================================
>    执行摘要
>    ================================
>    任务：[用户输入]
>    状态：completed / error / timeout
>    经过节点：agent -> tools -> agent -> END
>    工具调用：N 次（各工具调用次数）
>    总耗时：X.XX 秒
>    ================================
>
> 4. 打印完整状态历史（每个检查点的消息数量变化）
>
> 测试场景：需要 2-3 次工具调用的问题
> ```

---

## 10. 完整实战案例

### 案例 1：FTH 支付查询 Agent

**场景：** 运营人员用自然语言查询 FTH 支付系统状态

```
输入：
  "查一下今天失败的 NZD 国际转账，超过 $5000 的有哪些？"
    |
    v
+---------------------------------------------+
|           PaymentQueryAgent                 |
|                                             |
|  State:                                     |
|    messages:      对话历史                   |
|    query_results: 查询结果缓存               |
|    generated_sql: 生成的 SQL                |
|    iteration:     迭代计数                   |
|                                             |
|  Tools:                                     |
|    generate_sql(nl_query) -> SQL            |
|    execute_query(sql) -> 结果               |
|    format_report(data) -> Markdown 报告     |
|                                             |
|  Flow:                                      |
|    agent -> tools -> agent -> tools         |
|          -> agent -> format_report -> END   |
+---------------------------------------------+
    |
    v
输出：
  | TransactionID | Amount | Status | Reason |
  |---------------|--------|--------|--------|
  | TXN001234     | $6,500 | FAILED | AM04   |
  | TXN001567     | $8,200 | FAILED | AC01   |
```

**State 设计：**

```python
class PaymentAgentState(TypedDict):
    messages:      Annotated[list[BaseMessage], add_messages]
    query_results: Annotated[list[dict], operator.add]
    generated_sql: Optional[str]
    report:        Optional[str]
    iteration:     int
    error:         Optional[str]
```

---

### 案例 2：Eduacan 题目生成 Agent

**场景：** 输入知识点，自动生成 NCEA 风格习题

```
输入：
  "Generate NCEA Level 2 calculus: derivatives, medium difficulty"
    |
    v
+---------------------------------------------+
|         QuestionGeneratorAgent              |
|                                             |
|  Tools:                                     |
|    search_question_bank(topic)              |
|      -> 相似题目（避免重复）                 |
|    generate_question(topic, difficulty)     |
|      -> 题目草稿                            |
|    validate_question(question, answer)      |
|      -> 质量评分 0-100                      |
|    save_to_db(question_data)                |
|      -> 保存题目                            |
|                                             |
|  路由逻辑：                                  |
|    validate 后 score >= 75 -> save -> END   |
|    validate 后 score < 75  -> 重新 generate |
|    超过 3 次失败             -> END         |
+---------------------------------------------+
    |
    v
输出：
  题目:  "Find f'(x) where f(x) = 3x³ + 2x² - 5x + 1"
  答案:  "f'(x) = 9x² + 4x - 5"
  难度:  Medium
  知识点: [Power Rule, Polynomial Differentiation]
  质量分: 92/100
```

---

> **💡 Claude Code 综合提示词 — 完整实战案例：**
> ```
> 用 Python + LangGraph 实现 Eduacan 题目生成 Agent（完整版）：
>
> State：
> - messages:             对话历史
> - topic:               题目主题（从用户输入提取）
> - difficulty:          难度 Easy/Medium/Hard
> - generated_questions: 生成的题目列表（add reducer）
> - quality_scores:      对应质量分（add reducer）
> - approved_questions:  通过质量检查的题目
> - generation_attempts: 尝试次数
>
> 工具（4个）：
> 1. search_similar_questions(topic, difficulty)
>    -> 返回 mock 相似题目（避免重复）
>
> 2. generate_question(topic, difficulty, avoid_similar)
>    -> 调用 Claude API 生成题目（含题目、答案、解析）
>
> 3. validate_question(question, answer)
>    -> 调用 Claude API 评估质量（返回分数 0-100 + 改进建议）
>
> 4. save_question(question_data)
>    -> 模拟保存到数据库（打印保存成功信息）
>
> 节点与路由：
> - extract_requirements -> search -> generate -> validate
>   -> score >= 75: save -> END
>   -> score < 75:  重回 generate（最多 3 次）
>   -> 超过 3 次:   END + 打印失败信息
>
> 测试输入：
> "Generate 1 NCEA Level 2 Statistics question about
>  normal distribution, medium difficulty"
>
> 代码结构：
> eduacan_agent/
> ├── state.py
> ├── tools.py
> ├── nodes.py
> ├── graph.py
> └── main.py
>
> 使用 claude-3-5-sonnet-20241022
> ```

---

## 附录：LangChain + LangGraph 速查

### 常用导入

```python
# LangChain 核心
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool, StructuredTool
from langchain_anthropic import ChatAnthropic

# LangGraph 核心
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Python 标准库
from typing import Annotated, Optional
from typing_extensions import TypedDict
import operator
```

### 节点函数签名

```python
# 标准节点：接收 State，返回需要更新的字段
def my_node(state: MyState) -> dict:
    return {"field_to_update": new_value}

# 条件边函数：返回下一个节点名或 END
def my_router(state: MyState) -> str:
    return "next_node"  # 或 END
```

### 常用 Reducer 速查

```python
# 消息列表（智能合并，相同 id 更新）
messages: Annotated[list[BaseMessage], add_messages]

# 列表追加
items: Annotated[list, operator.add]

# 计数累加
count: Annotated[int, operator.add]

# 直接覆盖（默认，无需 Annotated）
status: str
current_step: int
```

### Graph 构建模板

```python
graph = StateGraph(MyState)

# 添加节点
graph.add_node("node_a", node_function_a)
graph.add_node("node_b", node_function_b)

# 普通边
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)

# 条件边
graph.add_conditional_edges(
    "node_a",
    router_function,
    {"route_1": "node_b", "route_2": "node_c", "end": END}
)

# 编译（带持久化记忆）
app = graph.compile(checkpointer=MemorySaver())

# 运行
config = {"configurable": {"thread_id": "my_session"}}
result = app.invoke({"messages": [HumanMessage("你好")]}, config)
```
