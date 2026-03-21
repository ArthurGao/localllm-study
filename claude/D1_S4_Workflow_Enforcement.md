# Domain 1 · 2.4 — 多步骤工作流：强制执行与交接模式
> Task Statement 1.4 | 考试高频考点 ⭐⭐⭐

---

## 📌 核心概念速查

| 中文 | 英文关键词 |
|------|-----------|
| 程序化强制 | Programmatic Enforcement |
| 前提条件门控 | Prerequisite Gate |
| Prompt 指导 | Prompt-based Guidance |
| 确定性合规 | Deterministic Compliance |
| 概率性合规 | Probabilistic Compliance |
| 结构化交接协议 | Structured Handoff Protocol |
| 人工升级 | Human Escalation |

---

## 🧠 核心知识点

### 1. 程序化强制 vs Prompt 指导（最重要区别）

```
Prompt 指导：
  "请先验证客户身份，再处理退款"
  → Claude 大部分时候会遵守，但有 ~12% 失败率
  → 不适合有财务后果的场景

程序化强制：
  代码层面阻止 process_refund 被调用，
  直到 get_customer 返回已验证的 customer_id
  → 100% 强制，无例外
```

**核心结论**：当违规有真实后果（财务、安全、合规），必须用程序化强制。

---

### 2. 前提条件门控实现

```python
class WorkflowGate:
    def __init__(self):
        self.verified_customer_id = None  # 门控状态

    def get_customer(self, customer_id: str) -> dict:
        result = {"customer_id": customer_id, "verified": True, "name": "Arthur"}
        self.verified_customer_id = customer_id  # 解锁门控
        return result

    def process_refund(self, order_id: str, amount: float) -> dict:
        # 程序化前提条件检查
        if not self.verified_customer_id:
            raise PermissionError(
                "必须先调用 get_customer 验证身份才能处理退款"
            )
        return {"status": "approved", "amount": amount}

# 工具执行器（在 agentic loop 中调用）
gate = WorkflowGate()

def execute_tool(tool_name: str, tool_input: dict):
    if tool_name == "get_customer":
        return gate.get_customer(tool_input["customer_id"])
    elif tool_name == "process_refund":
        return gate.process_refund(  # 门控在这里检查
            tool_input["order_id"],
            tool_input["amount"]
        )
```

---

### 3. 结构化交接摘要（Human Handoff）

当 Agent 需要升级给人工时，人工没有访问对话历史的权限，必须在摘要中包含所有关键信息：

```python
def build_handoff_summary(case_context: dict) -> str:
    """
    结构化交接摘要模板
    人工接手时无法访问对话历史，所有信息必须自包含
    """
    return f"""
## 升级摘要

**客户信息**
- 客户 ID：{case_context['customer_id']}
- 姓名：{case_context['customer_name']}
- 账户状态：{case_context['account_status']}

**根本原因分析**
{case_context['root_cause']}

**已尝试的解决方案**
{case_context['attempted_actions']}

**推荐操作**
{case_context['recommended_action']}

**升级原因**
{case_context['escalation_reason']}
"""
```

---

### 4. 并行调查多关注点请求

```python
# 客户同时有两个问题：退款 + 账单争议
# 正确做法：并行调查，共享上下文，最后合并

def handle_multi_concern_request(concerns: list[str], customer_id: str):
    # 先验证客户（共享上下文）
    customer = get_customer(customer_id)

    # 并行调查各问题（独立但共享客户上下文）
    results = {}
    for concern in concerns:
        results[concern] = investigate_concern(concern, customer)

    # 合并为统一响应
    return synthesize_resolution(results, customer)
```

---

## ⚠️ 重点难点

**难点**：Prompt 指令的失败率在生产环境中约 5-15%，对于财务操作这意味着真实损失。考题会用具体失败率数字（如"12% 的案例跳过了 get_customer"）来暗示需要程序化强制。

---

## 💡 Claude Code 提示词

```
用 Python 实现一个带前提条件门控的客户支持 Agent：
1. WorkflowGate 类：管理 get_customer → process_refund 的强制顺序
2. 在 agentic loop 中集成门控检查
3. 演示：没有门控时 Claude 跳过验证的场景（概率失败）
4. 演示：有门控时程序化阻止（确定性强制）
5. 生成结构化交接摘要

文件名：workflow_enforcement_demo.py
```

---

## 📋 例题

### 例题 1 ⭐⭐⭐
生产数据显示 12% 的案例中 Agent 跳过 `get_customer` 直接调用 `lookup_order`，导致账户误识别和错误退款。最有效的解决方案？

**A)** 添加程序化前提条件：阻止 `lookup_order` 和 `process_refund` 调用，直到 `get_customer` 返回已验证的客户 ID

**B)** 增强 system prompt：说明 `get_customer` 在任何订单操作前是强制性的

**C)** 添加 few-shot 示例：展示 Agent 总是先调用 `get_customer`

**D)** 实现路由分类器：分析每个请求并只启用该请求类型合适的工具子集

> **答案：A** — 当错误有财务后果时，程序化强制提供确定性保证，prompt 方法（B、C）依赖概率性 LLM 合规，D 解决工具可用性而非工具顺序问题。

---

### 例题 2 ⭐⭐
当需要升级到人工 Agent 时，下面哪项最能描述交接摘要应包含的内容？

**A)** 完整的对话历史记录

**B)** 仅包含客户 ID 和问题描述

**C)** 客户 ID、根本原因、已退款金额、推荐操作

**D)** 升级原因和情绪分析

> **答案：C** — 人工 Agent 无法访问对话历史，必须提供自包含的结构化摘要，包含所有需要继续处理的关键信息。

---

*文档版本：v1.0 | Domain 1 Task Statement 1.4*
