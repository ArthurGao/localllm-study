# Domain 5 — 上下文管理与可靠性
> Task Statements 5.1–5.6 | 考试权重：15%

---

## 5.1 跨长交互保留关键信息
> ⭐⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 渐进式摘要风险 | Progressive Summarization Risk |
| 迷失在中间效应 | Lost-in-the-middle Effect |
| 案例事实块 | Case Facts Block |
| 工具输出裁剪 | Tool Output Trimming |
| 位置感知排序 | Position-aware Ordering |

### 核心知识点

**迷失在中间效应**：

```
模型对输入位置的注意力分布：
  ████████████████ ← 开头（高注意力）
  ░░░░░░░░░░░░░░░░ ← 中间（低注意力）⚠️ 关键发现可能被遗漏
  ████████████████ ← 结尾（高注意力）

应对策略：
  1. 关键摘要放在聚合输入的开头
  2. 用明确节标题组织详细结果
  3. 关键事实块始终在 prompt 中保持最新
```

**案例事实块模式**：

```python
def build_customer_support_prompt(
    case_facts: dict,        # 持久的关键事实
    conversation_summary: str,  # 压缩的对话历史
    current_message: str
) -> str:
    """
    案例事实块放在 prompt 开头，不随摘要压缩而丢失
    """
    
    return f"""
## 案例事实（始终更新）
客户 ID: {case_facts['customer_id']}
订单号: {case_facts['order_id']}
退款金额: ${case_facts['refund_amount']}
问题类型: {case_facts['issue_type']}
账户状态: {case_facts['account_status']}
SLA 截止: {case_facts['sla_deadline']}

## 对话摘要
{conversation_summary}

## 当前消息
{current_message}
"""
    # 关键数值（金额、日期）在案例事实块中，不会因摘要而丢失
```

**工具输出裁剪**：

```python
# 问题：订单查询返回 40+ 字段，只有 5 个相关
raw_order_result = {
    "order_id": "12345",
    "status": "shipped",
    "amount": 99.99,
    "customer_id": "C001",
    "return_eligible": True,
    # ... 35 个不相关字段（shipping_partner_code, internal_tracking_id, ...）
}

# ✅ 在追加到上下文前裁剪
def trim_order_for_context(raw_result: dict) -> dict:
    # 只保留退款处理相关字段
    return {
        "order_id": raw_result["order_id"],
        "status": raw_result["status"],
        "amount": raw_result["amount"],
        "return_eligible": raw_result["return_eligible"],
        "purchase_date": raw_result["purchase_date"]
    }
    # 节省 ~80% 的上下文 token
```

---

### 📋 例题

**例题 ⭐⭐**

一个多轮客户支持 Agent 在对话后期错误地引用了错误的退款金额。根本原因最可能是什么？

**A)** 模型在长对话中能力下降

**B)** 渐进式摘要将具体的退款金额压缩为模糊描述，导致关键数值丢失

**C)** 需要更大的上下文窗口

**D)** 系统提示太长，占用了太多上下文

> **答案：B** — 渐进式摘要的风险是将数值（金额、日期、订单号）压缩为模糊描述。解决方案是将这些事实提取到不被摘要的"案例事实块"中。

---

## 5.2 升级与歧义解决模式
> ⭐⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 升级触发器 | Escalation Trigger |
| 策略缺口 | Policy Gap |
| 情绪升级 | Sentiment-based Escalation |
| 置信度升级 | Confidence-based Escalation |
| 歧义解决 | Ambiguity Resolution |

### 正确的升级触发器

```python
# ✅ 应该触发升级的情况
ESCALATION_TRIGGERS = [
    "customer_explicitly_requests_human",    # 客户明确要求人工
    "policy_gap_or_exception",               # 策略对此情况沉默/不明确
    "unable_to_make_meaningful_progress",    # 无法取得有效进展
]

# ❌ 不可靠的升级触发器（考试陷阱）
UNRELIABLE_TRIGGERS = [
    "negative_sentiment_detected",           # 情绪负面 ≠ 需要升级
    "low_confidence_score",                  # 自我报告置信度不可靠
    "case_seems_complex",                    # 复杂性是主观判断
]
```

**策略缺口升级场景**：

```python
# 场景：客户要求竞争对手价格匹配
# 策略文件只涵盖：自家平台历史价格匹配

def should_escalate(request_type: str, policy: dict) -> bool:
    # ✅ 策略明确覆盖 → 可自主处理
    if request_type in policy["covered_cases"]:
        return False
    
    # ✅ 策略明确拒绝 → 可自主拒绝并解释
    if request_type in policy["denied_cases"]:
        return False
    
    # ✅ 策略沉默（缺口）→ 必须升级！不能猜测
    if request_type not in policy["covered_cases"] and \
       request_type not in policy["denied_cases"]:
        return True  # 升级！策略缺口
```

**多匹配客户处理**：

```python
# ❌ 错误：启发式选择（选最近的或最相关的）
def bad_resolve_customer(name: str, results: list) -> dict:
    return results[0]  # 随机选第一个 → 可能选错账户！

# ✅ 正确：请求额外标识符
def good_resolve_customer(name: str, results: list) -> str:
    if len(results) > 1:
        return f"""
我们找到了 {len(results)} 个名为 {name} 的账户。
为了确保我们处理正确的账户，请提供以下任一信息：
- 注册邮箱
- 账户 ID
- 最近一次交易的金额和日期
"""
```

---

### 📋 例题

**例题 ⭐⭐（对应考试 Question 3）**

Agent 的首次联系解决率为 55%（目标 80%）。日志显示它将简单的损坏替换（有照片证据）升级给人工，但尝试自主处理需要策略例外的复杂情况。改善升级校准的最有效方法？

**A)** 在 system prompt 中添加明确的升级标准，包含展示何时升级 vs 自主解决的 few-shot 示例

**B)** 让 Agent 在每次响应前自我报告置信度（1-10），置信度低于阈值自动路由到人工

**C)** 部署一个在主 Agent 开始处理前预测哪些请求需要升级的独立分类器模型

**D)** 实现情绪分析，负面情绪超过阈值时自动升级

> **答案：A** — 明确的升级标准配合 few-shot 示例直接解决根本原因：决策边界不清晰。B 因 LLM 自我报告置信度校准差而失败，C 在 prompt 优化未尝试前过度工程化，D 解决完全不同的问题（情绪≠案例复杂度）。

---

## 5.3 跨多 Agent 系统的错误传播
> ⭐⭐

### 核心知识点

```python
# ✅ 结构化错误传播（让 Coordinator 能做智能决策）
def search_subagent_error_response(
    error_type: str,
    attempted_query: str,
    partial_results: list
) -> dict:
    
    return {
        "isError": True,
        "failureType": error_type,           # "timeout" | "not_found" | "permission"
        "attemptedQuery": attempted_query,    # Coordinator 可以修改查询重试
        "partialResults": partial_results,   # 即使失败也返回已获得的部分数据
        "alternativeApproaches": [           # 建议 Coordinator 的恢复策略
            "尝试使用不同的搜索词",
            "切换到备用数据源",
            "降低结果精确度要求"
        ]
    }


# 区分访问失败 vs 有效空结果
def handle_search_result(result: dict) -> str:
    if result.get("isError") and result.get("failureType") == "timeout":
        return "ACCESS_FAILURE"    # Coordinator 需要决定是否重试
    
    if not result.get("isError") and len(result.get("results", [])) == 0:
        return "VALID_EMPTY"       # 成功查询，确实没有匹配结果
    
    return "SUCCESS"


# 覆盖缺口注释
synthesis_output = {
    "report": "...",
    "coverage_annotations": {
        "well_supported_findings": ["AI 对视觉艺术的影响"],
        "gap_areas": ["音乐行业数据不可用（搜索超时）"],
        "confidence_level": "medium"  # 因为有覆盖缺口
    }
}
```

---

## 5.4 大型代码库探索的上下文管理
> ⭐⭐

### 核心知识点

```python
# 上下文退化的信号
# Agent 开始说"基于典型的 Spring Boot 模式..."（而非"基于你的代码中..."）
# Agent 引用之前分析过的函数时细节有误

# 对策1：草稿文件持久化关键发现
def maintain_scratchpad(discovery: dict):
    """将关键发现写入草稿文件，对抗上下文退化"""
    with open(".claude_scratchpad.md", "a") as f:
        f.write(f"""
## 发现 [{datetime.now()}]
- 入口点：{discovery['entry_point']}
- 关键类：{discovery['key_classes']}
- 依赖关系：{discovery['dependencies']}
""")
    # 后续问题："参考 .claude_scratchpad.md 中的发现..."

# 对策2：使用 /compact 压缩上下文
# 当上下文充满冗长的探索输出时，使用 /compact 压缩历史

# 对策3：崩溃恢复 - 结构化状态导出
agent_state = {
    "analyzed_files": ["src/auth/login.py", "src/payment/"],
    "pending_tasks": ["检查 RefundService 集成点"],
    "key_findings": {
        "race_condition": "src/payment/processor.py:127",
        "missing_index": "customers.email"
    }
}
# 每个 Agent 将状态导出到已知位置
# Coordinator 恢复时加载 manifest
```

---

## 5.5 人工审查工作流与置信度校准
> ⭐⭐

### 核心知识点

```
聚合准确率的陷阱：
  整体准确率 97% ≠ 所有字段都准确
  
  可能的情况：
  - 常见发票类型：99.8% 准确
  - 手写发票：65% 准确   ← 被整体数字掩盖！
  - vendor_name 字段：98% 准确
  - payment_terms 字段：71% 准确  ← 被整体数字掩盖！

应对策略：
  1. 按文档类型分层分析准确率
  2. 按字段分层分析准确率
  3. 在减少人工审查前验证所有细分的一致性能
```

```python
# 字段级置信度分数 + 路由
extraction_result = {
    "invoice_number": {"value": "INV-2024-001", "confidence": 0.99},
    "total_amount":   {"value": 1500.00, "confidence": 0.98},
    "vendor_name":    {"value": "Acme Corp", "confidence": 0.87},
    "payment_terms":  {"value": "Net 30", "confidence": 0.62}  # 低置信度！
}

def route_for_review(result: dict, threshold: float = 0.85) -> str:
    low_confidence_fields = [
        field for field, data in result.items()
        if data["confidence"] < threshold
    ]
    
    if low_confidence_fields:
        return f"人工审查（低置信度字段：{low_confidence_fields}）"
    else:
        return "自动通过"


# 分层随机采样（即使高置信度也要定期采样）
import random

def should_sample_for_review(confidence: float) -> bool:
    """高置信度提取的随机采样，持续监控错误率"""
    if confidence < 0.85:
        return True   # 总是审查
    elif confidence < 0.95:
        return random.random() < 0.10  # 10% 采样率
    else:
        return random.random() < 0.02  # 2% 采样率（检测新错误模式）
```

---

## 5.6 多源合成中的信息溯源
> ⭐⭐

### 核心知识点

```python
# ✅ 结构化声明-来源映射（Claim-Source Mapping）
def build_claim_source_mapping(findings: list) -> list:
    """
    每个发现必须包含来源归属，通过合成步骤保留
    """
    return [
        {
            "claim": "生成式 AI 使设计工作流速度提升 40%",
            "evidence_excerpt": "研究参与者报告...",
            "source_url": "https://example.com/study",
            "source_name": "MIT 设计研究院 2024",
            "publication_date": "2024-01",  # 防止时间差异被误解为矛盾
            "credibility": "peer_reviewed"
        }
    ]


# 处理冲突数据（不能随机选一个！）
conflicting_stats = {
    "claim": "AI 在创意工作者中的采用率",
    "value_a": "47%",
    "source_a": "Adobe Survey 2024（n=5000，全球）",
    "value_b": "23%",
    "source_b": "Forrester Report 2024（n=500，北美企业）",
    "conflict_note": "差异可能源于样本范围（全球 vs 北美企业）和样本量的不同",
    "resolution": "NOT_RESOLVED"  # 保留冲突，让 Coordinator 决定
}
# ❌ 不能：随机选 47% 或 23%
# ✅ 应该：在报告中保留两个值，标注冲突原因
```

---

### 📋 例题

**例题 ⭐⭐**

多 Agent 研究系统中，最终报告没有任何来源引用，尽管 Search Agent 确实找到了有来源的文章。根本原因最可能是什么？

**A)** Synthesis Agent 的 system prompt 没有强调引用来源的重要性

**B)** Search Agent 的输出格式不包含来源元数据

**C)** Coordinator 在向 Synthesis Agent 传递上下文时只传递了内容，丢失了来源元数据（URL、文档名、日期）

**D)** 应该使用更强的模型

> **答案：C** — 如果 Coordinator 传递给 Synthesis Agent 的 prompt 中只有内容而没有元数据，Synthesis Agent 就没有可引用的来源数据。B 也部分正确，但 C 更直接指向 Coordinator 在传递上下文时丢失元数据的行为。

---

*文档版本：v1.0 | Domain 5 Task Statements 5.1–5.6*
