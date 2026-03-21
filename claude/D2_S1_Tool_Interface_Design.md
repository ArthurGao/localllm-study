# Domain 2 · 3.1 — 设计有效工具接口：描述与边界
> Task Statement 2.1 | 考试高频考点 ⭐⭐⭐

---

## 📌 核心概念速查

| 中文 | 英文关键词 |
|------|-----------|
| 工具描述 | Tool Description |
| 工具选择可靠性 | Tool Selection Reliability |
| 功能重叠 | Functional Overlap |
| 路由错误 | Misrouting |
| 边界说明 | Boundary Explanation |
| 关键字敏感指令 | Keyword-sensitive Instruction |

---

## 🧠 核心知识点

### 1. 工具描述是 LLM 工具选择的主要机制

```
工具描述质量 → 直接决定工具选择准确率

最小化描述（坏）：
  "Retrieves customer information"
  "Retrieves order details"
  → 模型无法区分，随机选择

完整描述（好）：
  工具描述 = 目的 + 输入格式 + 输出 + 使用场景 + 边界（何时不用此工具）
```

---

### 2. 最小化描述 vs 完整描述对比

```python
# ❌ 最小化描述（导致路由错误）
tools_bad = [
    {
        "name": "get_customer",
        "description": "Retrieves customer information",  # 太模糊
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}}
        }
    },
    {
        "name": "lookup_order",
        "description": "Retrieves order details",  # 太模糊
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}}
        }
    }
]

# ✅ 完整描述（准确路由）
tools_good = [
    {
        "name": "get_customer",
        "description": (
            "通过客户 ID 查询并验证客户账户信息。"
            "输入：customer_id（格式：CUST-XXXXX）。"
            "输出：姓名、账户状态、注册日期、联系方式。"
            "使用场景：客户提供其客户 ID 时，或在处理任何订单操作前需要验证身份时。"
            "注意：如果用户提供的是订单号而非客户 ID，请使用 lookup_order。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "客户 ID，格式：CUST-XXXXX"
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "lookup_order",
        "description": (
            "通过订单号查询特定订单的详情和状态。"
            "输入：order_id（格式：#XXXXX 或纯数字）。"
            "输出：订单状态、金额、商品、配送信息、时间线。"
            "使用场景：客户询问特定订单（'我的订单 #12345 在哪里？'）。"
            "注意：不用于查询客户账户信息，只用于查询特定订单。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，如 #12345 或 12345"
                }
            },
            "required": ["order_id"]
        }
    }
]
```

---

### 3. 拆分通用工具 vs 保留通用工具

```python
# ❌ 过于通用（选择歧义）
tools_generic = [
    {
        "name": "analyze_document",
        "description": "分析文档"  # 太宽泛，不知道用于何种分析
    }
]

# ✅ 拆分为专用工具（清晰边界）
tools_specific = [
    {
        "name": "extract_data_points",
        "description": (
            "从文档中提取结构化数据点（数字、日期、名称）。"
            "输出：JSON 格式的键值对。"
            "使用场景：需要提取表格数据、财务数字、统计数据时。"
        )
    },
    {
        "name": "summarize_content",
        "description": (
            "生成文档内容的简洁摘要。"
            "输出：2-3 段的自然语言摘要。"
            "使用场景：用户需要了解文档主旨，而非具体数据时。"
        )
    },
    {
        "name": "verify_claim_against_source",
        "description": (
            "验证特定声明是否与文档内容一致。"
            "输入：声明文本 + 文档内容。"
            "输出：true/false + 支持/反驳证据引用。"
            "使用场景：事实核查、内容验证。"
        )
    }
]
```

---

### 4. System Prompt 关键字对工具选择的影响

```python
# ⚠️ 危险：system prompt 中的关键字可能覆盖工具描述

# 场景：system prompt 包含 "analyze all content"
# 工具：analyze_content（用于网页）vs analyze_document（用于 PDF）
# 问题："content" 关键字可能导致模型始终选择 analyze_content

# 解决方案：
# 1. 重命名工具以消除关键字歧义
bad_name = "analyze_content"
good_name = "extract_web_results"  # 更具体，避免 "content" 歧义

# 2. 审查 system prompt 中的措辞
# 避免使用与工具名高度重叠的通用词
```

---

## 💡 Claude Code 提示词

```
演示工具描述质量对工具选择的影响：
1. 创建两组工具定义：
   - 最小化描述版（导致路由错误）
   - 完整描述版（准确路由）
2. 用相同的用户请求测试两组工具
   - "check my order #12345"（应用 lookup_order）
   - "verify customer C001"（应用 get_customer）
3. 记录每次的工具选择，统计准确率差异
4. 展示通用工具拆分为专用工具的前后对比

文件名：tool_description_quality_demo.py
```

---

## 📋 例题

### 例题 1（对应考试 Question 2）⭐⭐⭐
生产日志显示 Agent 频繁在用户询问订单（如"check my order #12345"）时调用 `get_customer` 而非 `lookup_order`。两个工具描述都很简单（"Retrieves customer information" / "Retrieves order details"），且接受相似格式的 ID。改进工具选择可靠性的最有效第一步？

**A)** 添加 5-8 个 few-shot 示例展示正确的工具选择模式

**B)** 扩展每个工具描述，包含处理的输入格式、示例查询、边缘情况和边界说明

**C)** 实现路由层：在每次对话前解析用户输入并基于检测的关键字预选工具

**D)** 将两个工具合并为单一的 `lookup_entity` 工具

> **答案：B** — 工具描述是 LLM 工具选择的主要机制，最小化描述是根本原因。B 用低成本、高杠杆的方式直接解决根因。A 增加 token 开销但不解决根本问题，C 过度工程化，D 需要更多工作量。

---

### 例题 2 ⭐⭐
以下哪种工具描述设计原则最能减少相似工具间的路由错误？

**A)** 工具名称尽量简短，描述简洁

**B)** 在描述中明确说明：此工具的使用场景 AND 何时应选择其他工具

**C)** 将所有工具合并为一个，在内部逻辑中路由

**D)** 只需要工具名称唯一，描述内容不重要

> **答案：B** — 描述中的边界说明（"何时不用此工具"）是消除功能相近工具间歧义的关键。

---

*文档版本：v1.0 | Domain 2 Task Statement 2.1*
