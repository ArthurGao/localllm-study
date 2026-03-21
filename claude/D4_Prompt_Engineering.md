# Domain 4 — 提示工程与结构化输出
> Task Statements 4.1–4.6 | 考试权重：20%

---

## 4.1 设计精确标准：减少误报
> ⭐⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 明确标准 | Explicit Criteria |
| 误报 | False Positive |
| 分类标准 | Categorical Criteria |
| 严重级别 | Severity Level |

### 核心知识点

```python
# ❌ 模糊指令（无效）
bad_prompt = """
Review this code. Be conservative, only report high-confidence findings.
"""
# 问题：Claude 不知道"高置信度"的具体含义

# ✅ 明确分类标准
good_prompt = """
Review this code. Report issues according to these SPECIFIC rules:

ALWAYS REPORT:
- Security vulnerabilities (SQL injection, XSS, auth bypass)
- Bugs where code behavior contradicts comments
- Null pointer exceptions with clear reproduction path

NEVER REPORT:
- Code style preferences (naming conventions, formatting)
- Performance optimizations without evidence of bottleneck
- Patterns that differ from your preferences but work correctly

SEVERITY DEFINITION:
- Critical: Can cause data breach or service outage
- High: Incorrect behavior in common user paths
- Medium: Incorrect behavior in edge cases
- Low: Code quality issues affecting maintainability
"""
```

**关键原则**：高误报类别会破坏对准确类别的信任。如果"注释准确性"类别有 60% 误报，开发者会开始忽略所有报告，包括真正的安全漏洞。

**应对策略**：暂时禁用高误报类别，修复后重新启用，而非尝试用"更保守"的模糊指令改善。

---

### 📋 例题

**例题 ⭐⭐**

CI 代码审查的误报率很高，开发者开始忽略所有 AI 反馈。你尝试在 prompt 中添加"只报告高置信度发现"，但没有改善。最有效的下一步？

**A)** 将阈值从"高置信度"改为"非常高置信度"

**B)** 写明确的分类标准，定义哪些问题报告（bugs、安全）vs 跳过（小样式问题）

**C)** 切换到更强大的模型

**D)** 减少代码审查的频率

> **答案：B** — 通用指令（"保守点"）无法改善精确度，因为 Claude 不知道你认为什么是高置信度。明确的分类标准才能直接解决根本问题。

---

## 4.2 Few-shot 提示
> ⭐⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 少样本示例 | Few-shot Examples |
| 歧义案例处理 | Ambiguous Case Handling |
| 格式一致性 | Output Format Consistency |
| 泛化能力 | Generalization to Novel Patterns |

### 核心知识点

```python
# 场景：代码审查格式不一致（详细说明后仍不一致 → 需要 few-shot）

few_shot_prompt = """
Review code changes and report issues in this exact format.

EXAMPLES:

Example 1 (Security Issue):
Input: user_id = request.params['id']
       query = f"SELECT * FROM users WHERE id = {user_id}"
Output:
{
  "location": "auth.py:45",
  "issue": "SQL injection vulnerability via unsanitized user input",
  "severity": "critical",
  "suggested_fix": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))"
}

Example 2 (Acceptable Pattern - Do NOT flag):
Input: # This function handles refunds
       def process_refund(amount, reason):
           if not reason:
               reason = "No reason provided"
Output:
{
  "finding": "none",
  "reason": "Default value for optional parameter is acceptable pattern"
}

Example 3 (Ambiguous - How to handle):
Input: timeout = 30  # timeout in milliseconds
       time.sleep(timeout)  # wait for timeout
Output:
{
  "location": "utils.py:12",
  "issue": "Comment says milliseconds but time.sleep() takes seconds",
  "severity": "high",
  "suggested_fix": "Either change to time.sleep(timeout/1000) or update comment"
}

Now review the following code:
[CODE]
"""
```

**何时使用 few-shot**：
- 详细指令后输出格式仍不一致
- 需要演示歧义情况的处理方式
- 提取任务中出现幻觉（few-shot 显著减少幻觉）

---

## 4.3 强制结构化输出：tool_use + JSON Schema
> ⭐⭐⭐ 高频考点

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 工具使用提取 | `tool_use` extraction |
| 语法错误 | Syntax Errors（JSON 格式错误）|
| 语义错误 | Semantic Errors（值逻辑错误）|
| 可空字段 | Nullable Fields |
| 枚举类型 | Enum Fields |

### tool_choice 三种模式（重要）

```python
import anthropic
client = anthropic.Anthropic()

# 定义提取工具
extraction_tool = {
    "name": "extract_invoice_data",
    "description": "从发票文档中提取结构化数据",
    "input_schema": {
        "type": "object",
        "properties": {
            "invoice_number": {
                "type": "string",
                "description": "发票编号"
            },
            "total_amount": {
                "type": "number",
                "description": "发票总金额（美元）"
            },
            "vendor_name": {
                "type": "string",
                "description": "供应商名称"
            },
            "payment_terms": {
                # ✅ nullable：文档中可能没有此信息
                "type": ["string", "null"],
                "description": "付款条款，如文档未提供则为 null"
            },
            "document_type": {
                # ✅ enum + "other" 模式：可扩展分类
                "type": "string",
                "enum": ["invoice", "receipt", "purchase_order", "other"],
                "description": "文档类型"
            },
            "document_type_detail": {
                # 当 document_type 为 "other" 时填写
                "type": ["string", "null"],
                "description": "当类型为 other 时的具体说明"
            }
        },
        "required": ["invoice_number", "total_amount", "vendor_name"]
        # payment_terms 不在 required 中 → 允许为 null
    }
}

# 模式1：auto（可能返回文本而非调用工具）
response_auto = client.messages.create(
    model="claude-opus-4-5",
    tools=[extraction_tool],
    tool_choice={"type": "auto"},
    messages=[{"role": "user", "content": "从以下发票中提取数据：..."}]
)

# 模式2：any（保证调用工具）← 文档类型未知时推荐
response_any = client.messages.create(
    model="claude-opus-4-5",
    tools=[extraction_tool, other_tool],  # 多个工具
    tool_choice={"type": "any"},  # 必须调用某个工具
    messages=[...]
)

# 模式3：强制特定工具（提取必须先于丰富步骤）
response_forced = client.messages.create(
    model="claude-opus-4-5",
    tools=[extraction_tool, enrichment_tool],
    tool_choice={"type": "tool", "name": "extract_invoice_data"},  # 必须先提取
    messages=[...]
)


# 提取工具结果
def get_extraction_result(response) -> dict:
    for block in response.content:
        if block.type == "tool_use":
            return block.input  # 这就是提取的结构化数据
    return {}
```

**重要区别：语法错误 vs 语义错误**：

```python
# tool_use 消除语法错误（JSON 格式问题）
# 但无法阻止语义错误！

# 语义错误示例（tool_use 也无法防止）：
extraction_result = {
    "line_items": [
        {"description": "Service A", "amount": 100},
        {"description": "Service B", "amount": 150}
    ],
    "total_amount": 200  # ← 语义错误！100+150=250，不是200
}
# 解决语义错误：需要应用层验证（见 4.4）
```

---

### 📋 例题

**例题 ⭐⭐⭐**

你的提取系统要处理三种文档类型（发票、收据、采购订单），每种有不同的提取 schema。有时文档类型未知。如何保证总是返回结构化输出？

**A)** 使用 `tool_choice: "auto"` 让模型决定是否调用提取工具

**B)** 为每种文档类型定义一个提取工具，设置 `tool_choice: "any"` 保证调用某个工具

**C)** 在 prompt 中要求模型始终返回 JSON

**D)** 使用正则表达式从模型响应中提取 JSON

> **答案：B** — `tool_choice: "any"` 保证模型调用某个工具（而非返回文本），在文档类型未知时让模型选择最合适的 schema。A 可能返回纯文本，C/D 无法保证 JSON 格式正确。

---

## 4.4 验证、重试与反馈循环
> ⭐⭐

### 核心知识点

```python
import json

def extract_with_retry(document: str, max_retries: int = 2) -> dict:
    """
    带验证重试的提取流程
    """
    
    for attempt in range(max_retries + 1):
        # 提取
        result = run_extraction(document)
        
        # 验证
        errors = validate_extraction(result, document)
        
        if not errors:
            return result  # 成功
        
        if attempt < max_retries:
            # 重试：包含原始文档 + 失败提取 + 具体错误
            retry_prompt = f"""
原始文档：
{document}

上次提取结果（有错误）：
{json.dumps(result, ensure_ascii=False, indent=2)}

具体错误：
{chr(10).join(errors)}

请修正以上错误并重新提取。
"""
            # 用包含错误信息的 prompt 重试
            result = run_extraction(retry_prompt)
    
    return result  # 返回最后一次结果


def validate_extraction(result: dict, source_doc: str) -> list[str]:
    """验证语义正确性"""
    errors = []
    
    # 检查数值一致性
    if "line_items" in result and "total_amount" in result:
        calculated = sum(item["amount"] for item in result["line_items"])
        stated = result["total_amount"]
        if abs(calculated - stated) > 0.01:
            errors.append(
                f"数学错误：行项目合计 {calculated}，但总金额字段为 {stated}"
            )
    
    return errors


# 何时重试有效 vs 无效
# ✅ 有效：格式错误（日期格式不对、金额单位不对）
# ✅ 有效：结构错误（字段放错位置）
# ❌ 无效：信息在原始文档中根本不存在（重试也找不到）
```

**`detected_pattern` 字段**：用于分析误报模式

```python
# 在审查发现中添加 detected_pattern 字段
finding_schema = {
    "issue": "具体问题描述",
    "severity": "high|medium|low",
    "detected_pattern": "null_check_missing",  # 记录触发此发现的代码构造
    "suggested_fix": "..."
}
# 当开发者频繁忽略某类发现时，通过 detected_pattern 分析是否误报模式
```

---

## 4.5 批量处理策略
> ⭐⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| Message Batches API | Message Batches API |
| 自定义 ID | `custom_id` |
| SLA 计算 | SLA Calculation |
| 延迟容忍 | Latency-tolerant |

### 核心知识点

```
Message Batches API 特性：
  ✅ 50% 成本节省
  ✅ 最多 24 小时处理窗口
  ❌ 无保证延迟 SLA
  ❌ 不支持单请求内多轮工具调用

适用场景（延迟容忍）：
  ✅ 隔夜报告
  ✅ 每周审计
  ✅ 夜间测试生成
  ✅ 大量文档批量提取

不适用场景（需要实时）：
  ❌ 阻塞式合并前检查（开发者等待结果）
  ❌ 实时客户支持
  ❌ 需要工具调用的交互工作流
```

```python
import anthropic
client = anthropic.Anthropic()

# 提交批量请求
batch_requests = []
for i, document in enumerate(documents):
    batch_requests.append({
        "custom_id": f"doc-{i:04d}",  # 用于关联请求和响应
        "params": {
            "model": "claude-opus-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": f"提取以下文档的数据：{document}"}]
        }
    })

# 提交
batch = client.beta.messages.batches.create(requests=batch_requests)

# 检查结果（轮询）
import time
while batch.processing_status != "ended":
    time.sleep(60)  # 等待1分钟
    batch = client.beta.messages.batches.retrieve(batch.id)

# 处理结果（通过 custom_id 关联）
for result in client.beta.messages.batches.results(batch.id):
    doc_id = result.custom_id
    if result.result.type == "succeeded":
        process_success(doc_id, result.result.message)
    else:
        # 只重新提交失败的文档
        handle_failure(doc_id, result.result.error)
```

**SLA 计算示例**：

```
要求：30小时 SLA
批量 API 最长：24小时处理
安全提交窗口：30 - 24 = 6小时

→ 每6小时提交一次批量作业
→ 确保最坏情况下（24小时处理）仍满足30小时 SLA
```

---

## 4.6 多实例与多轮审查架构
> ⭐⭐

### 核心知识点

```
自我审查的局限性：
  同一会话中生成和审查：
  → Claude 记得自己的设计决策
  → 不太可能质疑自己的选择
  → 细微 bug 容易被自我确认偏见忽略

独立审查实例（更有效）：
  单独的 Claude 调用（无生成时的推理上下文）
  → 更可能发现细微问题
  → 类似于人工代码审查由不同人完成
```

```python
# 多实例审查模式
def review_generated_code(original_code: str, requirements: str) -> dict:
    
    # 步骤1：生成代码（实例 A）
    generation_response = client.messages.create(
        model="claude-opus-4-5",
        messages=[{
            "role": "user",
            "content": f"实现以下需求：{requirements}"
        }]
    )
    generated_code = extract_code(generation_response)
    
    # 步骤2：独立审查（实例 B，全新会话，无生成上下文）
    review_response = client.messages.create(
        model="claude-opus-4-5",
        # 注意：这是全新的 API 调用，没有步骤1的上下文
        messages=[{
            "role": "user",
            "content": f"""
请审查以下代码：

需求：{requirements}

代码：
{generated_code}

检查：安全漏洞、逻辑错误、边缘情况处理
"""
        }]
    )
    
    return {"code": generated_code, "review": extract_review(review_response)}
```

---

### 📋 例题

**例题 ⭐⭐（对应考试 Question 12）**

一个 PR 修改了 14 个文件，单次全量审查结果不一致：部分文件深度反馈，其他表浅；明显 bug 被遗漏；在一个文件标记的模式在另一个文件相同代码却被通过。如何重构审查？

**A)** 拆分为聚焦的 pass：每文件独立本地分析，然后单独的集成 pass 检查跨文件数据流

**B)** 要求开发者将大 PR 拆分为 3-4 个文件的提交

**C)** 切换到有更大上下文窗口的模型

**D)** 运行三个独立 pass，只标记出现至少两次的问题

> **答案：A** — 直接解决注意力稀释问题。C 误解了更大上下文窗口不能解决注意力质量问题。D 实际上会压制只偶尔被捕获的真实 bug 的检测。

---

*文档版本：v1.0 | Domain 4 Task Statements 4.1–4.6*
