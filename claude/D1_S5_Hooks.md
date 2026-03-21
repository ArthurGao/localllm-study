# Domain 1 · 2.5 — Agent SDK Hooks：工具拦截与数据规范化
> Task Statement 1.5 | 考试权重中等 ⭐⭐

---

## 📌 核心概念速查

| 中文 | 英文关键词 |
|------|-----------|
| 工具使用后钩子 | `PostToolUse` hook |
| 工具调用前拦截 | Pre-call Interception Hook |
| 数据规范化 | Data Normalization |
| 确定性保证 | Deterministic Guarantee |
| 概率性合规 | Probabilistic Compliance |
| 策略违规拦截 | Policy Violation Interception |

---

## 🧠 核心知识点

### 1. Hooks 的两种类型

```
Hook 类型一：PostToolUse（工具结果变换）
  工具执行 → [Hook 拦截] → 规范化数据 → 传给模型
  
  用途：
  - 统一不同工具返回的数据格式
  - Unix 时间戳 → ISO 8601
  - 数字状态码 → 人类可读描述

Hook 类型二：Pre-call Interception（工具调用拦截）
  模型请求调用工具 → [Hook 拦截] → 检查规则 → 允许/阻止
  
  用途：
  - 阻止超出阈值的退款（>$500 → 升级到人工）
  - 执行业务规则的确定性保证
```

---

### 2. PostToolUse：数据规范化

真实场景：三个 MCP 工具返回不同格式的时间戳，模型需要一致的格式：

```python
from datetime import datetime, timezone

def post_tool_use_hook(tool_name: str, tool_result: dict) -> dict:
    """
    PostToolUse hook：规范化工具返回的数据
    在模型看到数据之前统一格式
    """
    normalized = tool_result.copy()
    
    # 规范化时间戳
    if "created_at" in normalized:
        ts = normalized["created_at"]
        if isinstance(ts, (int, float)):
            # Unix 时间戳 → ISO 8601
            normalized["created_at"] = datetime.fromtimestamp(
                ts, tz=timezone.utc
            ).isoformat()
    
    # 规范化状态码
    STATUS_MAP = {1: "active", 2: "suspended", 3: "closed", 0: "unknown"}
    if "status_code" in normalized:
        code = normalized["status_code"]
        normalized["status"] = STATUS_MAP.get(code, f"unknown({code})")
        del normalized["status_code"]
    
    return normalized
```

---

### 3. Pre-call Hook：策略违规拦截

```python
def pre_tool_call_hook(tool_name: str, tool_input: dict) -> dict:
    """
    工具调用前拦截 hook
    返回：{"action": "allow"} 或 {"action": "block", "redirect": ...}
    """
    
    if tool_name == "process_refund":
        amount = tool_input.get("amount", 0)
        
        # 业务规则：>$500 退款必须人工审批
        if amount > 500:
            return {
                "action": "block",
                "redirect": "escalate_to_human",
                "reason": f"退款金额 ${amount} 超过自动处理上限 $500",
                "suggested_action": "请将此案例升级到高级支持团队"
            }
    
    return {"action": "allow"}


# 在 agentic loop 中集成 hook
def execute_tool_with_hooks(tool_name: str, tool_input: dict):
    # 1. 调用前检查
    pre_result = pre_tool_call_hook(tool_name, tool_input)
    if pre_result["action"] == "block":
        return {
            "blocked": True,
            "reason": pre_result["reason"],
            "next_action": pre_result.get("redirect")
        }
    
    # 2. 执行工具
    raw_result = execute_tool(tool_name, tool_input)
    
    # 3. 调用后规范化
    normalized_result = post_tool_use_hook(tool_name, raw_result)
    
    return normalized_result
```

---

### 4. Hooks vs Prompt 的根本区别

| 特性 | Hooks（程序化） | Prompt 指令 |
|------|----------------|------------|
| 合规率 | 100% | ~85-95% |
| 适用场景 | 业务规则必须强制执行 | 偏好和风格指导 |
| 例子 | 退款金额上限 | 回复语气友好 |

---

## 💡 Claude Code 提示词

```
实现 Agent SDK Hooks 演示系统：
1. PostToolUse hook：规范化三种工具返回的异构格式
   - Unix 时间戳 → ISO 8601
   - 数字状态码 → 文字描述
   - 货币格式统一（cents → dollars）
2. Pre-call hook：拦截超过 $500 的退款，重定向到人工升级
3. 对比演示：有 hook vs 无 hook 时模型收到的数据差异

文件名：agent_hooks_demo.py
```

---

## 📋 例题

### 例题 1 ⭐⭐
你的 Agent 从三个不同的 MCP 工具接收数据：一个返回 Unix 时间戳，一个返回 ISO 8601，一个返回人类可读字符串。最优解决方案？

**A)** 在 system prompt 中告诉 Claude 如何解读不同时间格式

**B)** 实现 PostToolUse hook 在模型处理前统一规范化所有时间格式

**C)** 修改每个 MCP 工具的源代码统一输出格式

**D)** 让 Claude 在回复时自行转换时间格式

> **答案：B** — PostToolUse hook 是在不修改上游工具的情况下集中处理格式规范化的最优位置。

---

### 例题 2 ⭐⭐⭐
关于使用 hooks 强制执行业务规则（如退款上限）vs 在 prompt 中说明规则，哪个说法正确？

**A)** Prompt 指令和 hooks 提供相同的可靠性保证

**B)** Hooks 提供确定性保证，prompt 指令存在非零失败率

**C)** Prompt 指令更可靠，因为模型能理解业务上下文

**D)** 只在首次违规时需要 hooks，之后 Claude 会记住规则

> **答案：B** — 核心原则：程序化 hooks 是确定性的，prompt 指令是概率性的。对于有业务后果的规则必须用 hooks。

---

*文档版本：v1.0 | Domain 1 Task Statement 1.5*
