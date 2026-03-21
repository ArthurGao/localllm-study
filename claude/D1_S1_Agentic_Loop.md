# Domain 1 · 2.1 — Agentic Loop 核心机制
> Claude Certified Architect – Foundations | 考试权重：Domain 1 占 27%

---

## 📌 核心概念速查

| 中文 | 英文关键词 | 说明 |
|------|-----------|------|
| 代理循环 | Agentic Loop | Claude 自主执行任务的主控制流 |
| 停止原因 | `stop_reason` | 决定循环是否继续的关键字段 |
| 工具使用 | `tool_use` | 模型请求执行某个工具时的 stop_reason |
| 轮次结束 | `end_turn` | 模型认为任务完成时的 stop_reason |
| 工具结果 | tool result | 工具执行后返回给模型的数据 |
| 对话历史 | conversation history | 每轮工具结果追加后传回模型的上下文 |
| 模型驱动决策 | model-driven decision-making | Claude 基于上下文推理下一步调用哪个工具 |
| 预配置决策树 | pre-configured decision tree | 与模型驱动相对，固定的工具调用顺序 |

---

## 🧠 核心知识点

### 1. Agentic Loop 生命周期（Life Cycle）

```
┌─────────────────────────────────────────────────────┐
│                   Agentic Loop                       │
│                                                       │
│  1. 发送请求给 Claude (Send Request)                  │
│         ↓                                             │
│  2. 检查 stop_reason (Inspect stop_reason)            │
│         ↓                                             │
│   ┌─────┴──────┐                                     │
│   │            │                                     │
│  tool_use   end_turn                                  │
│   │            │                                     │
│   ↓            ↓                                     │
│  执行工具    终止循环                                  │
│  (Execute)  (Terminate)                               │
│   │                                                   │
│  3. 工具结果追加到对话历史                             │
│     (Append tool result to conversation history)      │
│         ↓                                             │
│  4. 返回下一次迭代 (Loop back to step 1)              │
└─────────────────────────────────────────────────────┘
```

**关键理解**：Claude 本身不执行工具，工具由你的代码执行，结果再传回 Claude。

---

### 2. 两种 stop_reason 的意义

```python
response = client.messages.create(...)

if response.stop_reason == "tool_use":
    # Claude 想要调用工具 → 你去执行 → 结果追加 → 继续循环
    pass
elif response.stop_reason == "end_turn":
    # Claude 认为任务完成 → 循环终止
    pass
```

---

### 3. 如何保证 Claude 一定调用工具？（tool_choice 控制）

**问题**：默认情况下 Claude 可能直接给文字答案，完全跳过工具调用。

```python
# ⚠️ 这种情况完全合法，Claude 不报错：
response.stop_reason  # "end_turn"  ← 没有调用任何工具！
response.content      # [TextBlock(text="订单状态是已发货...")]
# Claude 凭"记忆"或推断直接回答了，绕过了所有工具
```

**根本原因**：默认 `tool_choice` 是 `"auto"`，Claude 自己判断要不要调工具。

---

#### 三种 tool_choice 配置

```python
import anthropic
client = anthropic.Anthropic()

# ❌ 默认 auto：Claude 自己决定，可能直接给答案
response = client.messages.create(
    model="claude-opus-4-5",
    tools=tools,
    tool_choice={"type": "auto"},        # 可能跳过工具！
    messages=messages
)

# ✅ any：强制必须调用某个工具，Claude 自选哪个
response = client.messages.create(
    model="claude-opus-4-5",
    tools=tools,
    tool_choice={"type": "any"},         # 保证 stop_reason="tool_use"
    messages=messages                    # 不会直接返回纯文本答案
)

# ✅ 强制特定工具：必须调用这个，没得选
response = client.messages.create(
    model="claude-opus-4-5",
    tools=tools,
    tool_choice={"type": "tool", "name": "lookup_order"},  # 只能调这个
    messages=messages
)
```

---

#### Agentic Loop 中的正确用法

```python
def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    for _ in range(20):
        # 关键：根据轮次动态切换 tool_choice
        is_first_turn = len(messages) == 1
        tool_choice = {"type": "any"} if is_first_turn else {"type": "auto"}
        # 第一轮：强制调工具（不允许直接回答）
        # 后续轮：auto（让 Claude 决定继续调工具还是 end_turn）

        response = client.messages.create(
            model="claude-opus-4-5",
            tools=tools,
            tool_choice=tool_choice,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return extract_text(response)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = execute_all_tools(response)
        messages.append({"role": "user", "content": tool_results})
```

---

#### 选择原则速查

| `tool_choice` | 保证 | 适用场景 |
|--------------|------|---------|
| `"auto"` | 不保证调工具 | 普通对话，工具是可选辅助 |
| `"any"` | 保证调某个工具 | 必须查实时数据；需要结构化输出 |
| `{"type":"tool","name":"X"}` | 保证调指定工具 | 工作流第一步必须是特定工具 |

> ⚠️ **考试关联**：`tool_choice: "any"` 在 Domain 4（结构化输出）也是高频考点——用于保证提取时返回 JSON 结构而非纯文本。

---

### 5. 工具结果追加到对话历史（Tool Results in Conversation History）

每次工具执行后，必须将结果以特定格式追加到消息列表，Claude 才能基于结果继续推理：

```python
# 对话历史结构示意
messages = [
    {"role": "user", "content": "帮我查一下订单 #12345 的状态"},
    # Claude 响应，请求调用工具
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "tool_abc", "name": "lookup_order", 
         "input": {"order_id": "12345"}}
    ]},
    # 工具执行结果追加 ← 这一步至关重要
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tool_abc",
         "content": '{"status": "shipped", "eta": "2024-03-20"}'}
    ]}
    # 再次调用 Claude，基于工具结果继续推理
]
```

---

### 6. 模型驱动 vs 预配置决策树（重要区别）
**Model-driven Decision-making vs Pre-configured Decision Tree**

#### 核心区别一句话

```
预配置决策树 (Pre-configured Decision Tree)：
  你的代码决定调用哪个工具、什么顺序

模型驱动 (Model-driven Decision-making)：
  Claude 自己决定调用哪个工具、什么顺序
```

---

#### 对比表

| 特性 | 模型驱动 (Model-driven) | 预配置决策树 (Pre-configured) |
|------|------------------------|------------------------------|
| 工具选择者 | Claude 基于上下文推理 | 你的代码硬编码顺序 |
| 灵活性 | 高，不同输入走不同路径 | 低，所有输入走同一流程 |
| 合规性 | 概率性（Probabilistic）~85-95% | 确定性（Deterministic）100% |
| 适用场景 | 高歧义、动态任务 | 安全/合规要求的强制顺序 |
| 例子 | 客户支持多意图处理 | 身份验证 → 财务操作门控 |

---

#### 具体例子：客户退款请求

**场景**：客户说 `"I received a damaged item, order #12345, I want a refund"`

**方式一：预配置决策树 Pre-configured Decision Tree**

```python
# 你的代码写死了执行顺序
# Claude 只负责提取参数，不负责决策流程

def handle_refund_request(user_message: str):

    # 步骤固定，由代码控制，Claude 没有选择权
    order_id = extract_order_id(user_message)         # 代码提取参数

    customer = get_customer(order_id=order_id)        # 代码决定：永远先查客户
    order    = lookup_order(order_id=order_id)        # 代码决定：再查订单
    eligible = check_return_eligibility(order_id)     # 代码决定：再检查资格
    refund   = process_refund(order_id, order.amount) # 代码决定：最后退款

    return generate_response(customer, refund)

# ✅ 顺序 100% 可预测、可强制
# ✅ 适合：身份验证必须在财务操作前的场景
# ❌ 无论用户说什么，都走完整 4 步
# ❌ 用户只想查状态时，也会走退款流程
```

**方式二：模型驱动 Model-driven Decision-making**

```python
# Claude 自己看对话，推理该调用哪个工具
# 你的代码只负责：执行 Claude 请求的工具，把结果还给 Claude

tools = [
    {"name": "get_customer",            "description": "验证客户身份"},
    {"name": "lookup_order",            "description": "查询订单详情"},
    {"name": "check_return_eligibility","description": "检查退款资格"},
    {"name": "process_refund",          "description": "执行退款"},
    {"name": "escalate_to_human",       "description": "升级到人工"},
]

# 你的代码只运行 agentic loop，不决定工具顺序：
messages = [{"role": "user", "content": "I received a damaged item, order #12345"}]

# --- 迭代 1 ---
# Claude 分析："用户有订单号，我先查订单"
# → Claude 自己决定调用 lookup_order（跳过 get_customer！）

# --- 迭代 2 ---
# Claude 看到订单结果："在退款期内，检查资格"
# → Claude 自己决定调用 check_return_eligibility

# --- 迭代 3 ---
# Claude 看到资格通过："可以退款"
# → Claude 自己决定调用 process_refund → end_turn

# ✅ 根据上下文动态选择路径
# ✅ 不同输入走不同工具组合
# ❌ 顺序不能 100% 保证（概率性，存在失败率）
# ❌ 不适合"必须先验证才能操作"的合规场景
```

---

#### 同一系统，不同输入的真实行为对比

```
输入 A: "order #12345 status?"（只查状态）
  预配置: get_customer → lookup_order → check_eligibility → process_refund
          （走完整退款流程，即使用户不要退款！）
  模型驱动: lookup_order → end_turn
          （Claude 判断用户只要查状态，直接结束）

输入 B: "refund my order #12345, item was broken"（要退款）
  预配置: get_customer → lookup_order → check_eligibility → process_refund
  模型驱动: lookup_order → check_eligibility → process_refund
          （Claude 跳过 get_customer，订单号已足够验证）

输入 C: "I'm really unhappy, just get me a human"（要人工）
  预配置: get_customer → lookup_order → ...
          （代码不知道用户要求人工，继续走固定流程）
  模型驱动: escalate_to_human → end_turn
          （Claude 直接识别意图，跳过所有其他工具）
```

---

#### 何时用哪种？选择原则

```
用预配置决策树 (Pre-configured Decision Tree) 当：
  ✅ 顺序有安全/合规要求（身份验证必须在财务操作前）
  ✅ 工具调用顺序错误会有真实后果（财务损失、数据泄露）
  ✅ 例：get_customer 必须在 process_refund 前 → 用 WorkflowGate

用模型驱动 (Model-driven Decision-making) 当：
  ✅ 用户意图多样，流程需要灵活响应
  ✅ 不同情况需要不同的工具组合
  ✅ 例：客户支持 Agent 处理退款/查询/投诉等多种意图

⭐ 生产系统最佳实践：两者结合
  模型驱动负责：理解意图，动态选择工具路径（灵活性）
  预配置门控负责：关键顺序强制执行（安全性）
```

---

> ⚠️ **考试陷阱**：题目描述"12% 案例中 Agent 跳过了 `get_customer`"——
> 这是模型驱动的概率性失败（Probabilistic Failure）。
> 答案永远是**加程序化门控（Programmatic Gate）**，
> 而不是"改进 prompt"或"加 few-shot 示例"，
> 因为任何 prompt 方案都是模型驱动，永远有非零失败率（Non-zero Failure Rate）。

---

## ❌ 必须避免的反模式（Anti-patterns）

### 反模式 1：解析自然语言信号来判断终止

```python
# ❌ 错误做法
response_text = get_assistant_text(response)
if "任务完成" in response_text or "done" in response_text.lower():
    break  # 不可靠！Claude 不一定说这些词

# ✅ 正确做法
if response.stop_reason == "end_turn":
    break
```

### 反模式 2：用固定迭代次数作为主要停止机制

```python
# ❌ 错误做法
for i in range(10):  # 以迭代次数为主要停止条件
    response = call_claude(messages)
    # ... 处理工具调用

# ✅ 正确做法（迭代上限作为安全保护，不是主要逻辑）
MAX_ITER = 20  # 安全保护上限
for i in range(MAX_ITER):
    response = call_claude(messages)
    if response.stop_reason == "end_turn":
        break  # stop_reason 才是主要停止条件
    # 处理 tool_use...
```

### 反模式 3：检查助手文本内容作为完成指示器

```python
# ❌ 错误做法
last_block = response.content[-1]
if last_block.type == "text":  # 有文本 = 完成？错误！
    break

# ✅ 正确做法
if response.stop_reason == "end_turn":
    break
```

---

## 💡 Claude Code 提示词（用于生成学习代码）

将以下提示词粘贴到 Claude Code 中，生成对应的学习示例：

```
请用 Python + Anthropic SDK 实现一个完整的 Agentic Loop 示例，要求：

1. 创建一个客户订单查询 Agent，包含两个工具：
   - lookup_order(order_id): 返回订单状态（模拟数据）
   - get_customer(customer_id): 返回客户信息（模拟数据）

2. 实现标准 Agentic Loop：
   - 检查 stop_reason 决定是否继续
   - 将工具结果正确追加到 conversation history
   - 使用 end_turn 作为终止条件

3. 展示三个反模式的对比：
   - 错误：解析自然语言判断终止
   - 错误：固定迭代次数
   - 正确：基于 stop_reason

4. 添加详细中文注释解释每个步骤

文件名：agentic_loop_demo.py
```

---

## 🔍 真实生产案例：Westpac FTH 支付 Agent

结合你在 Westpac NZ 的 FTH 支付系统经验，以下是一个贴近真实场景的类比：

```
FTH 支付处理 ≈ Agentic Loop

传统代码（预配置决策树）：
  发送 IMT → 等待响应 → 处理 AM04 → 重试
  （固定顺序，由代码决定每步）

Agent 方式（模型驱动）：
  Claude 分析支付状态 → 决定调用 check_balance / retry_payment / escalate
  （Claude 基于上下文推理下一步）

关键区别：
  - 你的 CustomFthPaymentResponseProcessor 是"预配置"模式
  - Agent 的工具调用是"模型驱动"模式
  - 两者的 stop 条件都应该是"业务完成"而非"迭代次数"
```

---

## 📝 重点要点（必须深刻理解）

### 要点 1：Claude 不执行工具，你的代码执行工具

```
[你的代码] → 调用 Claude API
[Claude]   → 返回 tool_use 请求（说"我想调用 X 工具，参数是 Y"）
[你的代码] → 实际执行工具，获得结果
[你的代码] → 将结果追加到 messages
[你的代码] → 再次调用 Claude API（带更新后的 messages）
```

这个"委托执行"模式是理解 agentic loop 的核心。

### 要点 2：对话历史是 Claude 的"记忆"

每次循环，完整的 messages 列表（包含所有历史工具结果）必须传给 Claude，因为 Claude API 是**无状态的（stateless）**。

### 要点 3：stop_reason 是唯一可靠的终止信号

只有 `"end_turn"` 表示 Claude 认为任务完成。其他任何启发式方法（文本内容、迭代次数）都是不可靠的。

### 要点 4：工具结果格式必须正确

```python
# 工具结果必须作为 role="user" 的消息追加
# tool_use_id 必须与 Claude 请求的 tool id 匹配
{"role": "user", "content": [
    {"type": "tool_result", 
     "tool_use_id": "toolu_01XFDUDYJgAACTvnnFRFM72P",  # 必须匹配
     "content": "工具执行结果"}
]}
```

---

## 🎯 完整代码示例

```python
import anthropic
import json

client = anthropic.Anthropic()

# ============================================================
# 工具定义（Tool Definitions）
# ============================================================
tools = [
    {
        "name": "lookup_order",
        "description": "通过订单 ID 查询订单详情。输入：order_id（字符串）。"
                       "返回：订单状态、金额、配送信息。"
                       "使用场景：客户询问特定订单时。",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号，格式如 #12345"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "get_customer",
        "description": "通过客户 ID 查询客户信息并验证身份。"
                       "必须在处理任何订单操作前先调用此工具。",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"}
            },
            "required": ["customer_id"]
        }
    }
]

# ============================================================
# 模拟工具执行（Tool Execution - your code runs these）
# ============================================================
def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "lookup_order":
        return json.dumps({
            "order_id": tool_input["order_id"],
            "status": "shipped",
            "amount": 99.99,
            "eta": "2024-03-20"
        })
    elif tool_name == "get_customer":
        return json.dumps({
            "customer_id": tool_input["customer_id"],
            "name": "Arthur Chen",
            "verified": True
        })
    return json.dumps({"error": "unknown tool"})

# ============================================================
# 标准 Agentic Loop（Standard Agentic Loop）
# ============================================================
def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    MAX_ITERATIONS = 20  # 安全保护上限，非主要停止条件
    
    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- 迭代 {iteration + 1} ---")
        
        # Step 1: 调用 Claude
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        print(f"stop_reason: {response.stop_reason}")
        
        # Step 2: 检查 stop_reason（关键判断）
        if response.stop_reason == "end_turn":
            # ✅ 正确终止条件：Claude 认为任务完成
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "任务完成"
            )
            return final_text
        
        if response.stop_reason != "tool_use":
            # 处理意外的 stop_reason
            return f"意外的 stop_reason: {response.stop_reason}"
        
        # Step 3: 将 Claude 响应追加到对话历史
        messages.append({"role": "assistant", "content": response.content})
        
        # Step 4: 执行所有请求的工具（Claude 可能请求多个）
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"执行工具: {block.name}({block.input})")
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # 必须匹配 Claude 的请求 ID
                    "content": result
                })
        
        # Step 5: 将工具结果追加到对话历史（作为 user 消息）
        messages.append({"role": "user", "content": tool_results})
        
        # 循环继续 → 带更新的 messages 再次调用 Claude
    
    return "已达最大迭代次数限制"


# ============================================================
# 运行示例
# ============================================================
if __name__ == "__main__":
    result = run_agent("请帮我查询客户 C001 的订单 #12345 状态")
    print(f"\n最终结果：{result}")
```

---

## ⚠️ 重点 & 难点

### 难点 1：多个工具调用在同一次响应中

Claude 可能在一次响应的 `content` 列表中包含**多个** `tool_use` 块。必须遍历所有块，执行每一个工具，并将所有结果打包成一个 `user` 消息：

```python
# ✅ 正确：处理多个工具调用
tool_results = []
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result
        })
# 一次追加所有结果
messages.append({"role": "user", "content": tool_results})
```

### 难点 2：`tool_use_id` 必须精确匹配

每个工具结果的 `tool_use_id` 必须与 Claude 响应中 `tool_use` 块的 `id` 完全一致，否则 API 会报错。

### 难点 3：同一响应中可能混有 text 和 tool_use

```python
# Claude 的 content 可能是：
[
    {"type": "text", "text": "好的，我来查询订单信息..."},
    {"type": "tool_use", "id": "toolu_abc", "name": "lookup_order", "input": {...}}
]
# 必须完整追加整个 content 列表，不能只追加 tool_use 部分
messages.append({"role": "assistant", "content": response.content})
```

---

## 📋 例题练习（考试模拟）

### 例题 1（难度：★★☆）

你实现了一个 Agentic Loop，代码如下：

```python
while True:
    response = client.messages.create(...)
    text = extract_text(response)
    if "我已完成所有操作" in text:
        break
    execute_tools(response)
```

这段代码存在什么问题？最佳修改方案是什么？

**A)** 没有最大迭代次数限制，应该加 `for i in range(10)`

**B)** 使用自然语言解析来判断终止，应该使用 `response.stop_reason == "end_turn"`

**C)** 应该在每次调用前清空 messages

**D)** 工具执行顺序不正确，应该反转

> **正确答案：B**
> 解析：自然语言信号不可靠，`stop_reason` 才是官方终止信号。选项 A 会将迭代次数变为主要停止机制，这是反模式。

---

### 例题 2（难度：★★★）

在一个客户支持 Agent 中，Claude 在单次响应中同时请求调用 `get_customer` 和 `lookup_order` 两个工具。下面哪种处理方式是正确的？

**A)** 只执行第一个工具请求，下一次迭代再执行第二个

**B)** 同时执行两个工具，将两个结果分别作为两条独立的 `user` 消息追加

**C)** 同时执行两个工具，将两个结果合并到一条 `user` 消息中作为列表追加

**D)** 先询问用户确认后再执行工具

> **正确答案：C**
> 解析：所有工具结果必须在一个 `user` 消息中以 `tool_result` 列表形式追加。分成两条消息会破坏对话格式。

---

### 例题 3（难度：★★★）

生产日志显示你的 Agent 在某些情况下会无限循环。最可能的根本原因是什么？

**A)** tools 列表定义中有语法错误

**B)** 工具执行结果没有被追加到 conversation history，导致 Claude 每次都重新请求相同的工具

**C)** max_tokens 设置太小

**D)** 应该使用 `claude-haiku` 而非 `claude-opus`

> **正确答案：B**
> 解析：如果工具结果没有正确追加到 messages，Claude 下一次迭代时看不到执行结果，会继续请求相同工具，造成无限循环。这是最常见的 agentic loop 实现错误。

---

## 🔗 学习资源

- Anthropic 官方文档：Tool Use Overview
- 相关考试场景：Scenario 1（Customer Support），Scenario 4（Developer Productivity）
- 关联知识点：Domain 1.2（多 Agent 编排）、Domain 1.4（工作流强制执行）

---

*文档版本：v1.0 | 对应考试 Domain 1 Task Statement 1.1*
