# Domain 1 · 2.7 — 会话状态管理：恢复与 Fork
> Task Statement 1.7 | 考试权重较低 ⭐

---

## 📌 核心概念速查

| 中文 | 英文关键词 |
|------|-----------|
| 命名会话恢复 | Named Session Resumption |
| 会话 Fork | `fork_session` |
| 过期工具结果 | Stale Tool Results |
| 结构化摘要注入 | Structured Summary Injection |
| 分支探索 | Divergent Approach Exploration |

---

## 🧠 核心知识点

### 1. 三种会话管理模式

```
模式一：--resume <session-name>
  继续一个命名的已有会话
  适用：上下文大部分仍有效
  命令：claude --resume my-investigation "继续分析认证模块"

模式二：fork_session
  从共享基线创建独立分支
  适用：需要探索多个互斥方向
  场景：比较两种重构方案

模式三：新会话 + 注入摘要
  丢弃旧会话，开启新会话并注入结构化摘要
  适用：旧工具结果已过期（代码已大幅修改）
  最可靠的方式
```

---

### 2. 恢复 vs 新会话的决策树

```
代码自上次会话以来是否修改？
    │
    ├─ 小幅修改（1-2个文件）
    │   → 使用 --resume + 告知修改的具体文件
    │
    ├─ 大幅修改（多文件重构）
    │   → 新会话 + 注入结构化摘要
    │   原因：旧工具结果（文件内容、函数签名）已过期
    │
    └─ 无修改
        → 直接 --resume 继续
```

---

### 3. fork_session 使用场景

```python
# 场景：比较两种测试策略，从同一代码分析出发

# 共享基线分析（两个分支共用）
codebase_analysis = """
分析结果：
- 主要入口点：src/payment/processor.py
- 关键业务逻辑：refund_handler, verify_customer
- 现有测试覆盖率：23%
- 高风险未测试路径：...
"""

# Fork A：单元测试策略
fork_a_prompt = f"""
【代码库分析基线】
{codebase_analysis}

基于以上分析，制定【单元测试优先】策略：
- 为每个函数写独立单元测试
- 使用 mock 隔离外部依赖
- 目标：覆盖率 80%
"""

# Fork B：集成测试策略（从同一基线出发）
fork_b_prompt = f"""
【代码库分析基线】
{codebase_analysis}

基于以上分析，制定【集成测试优先】策略：
- 测试完整业务流程
- 使用真实数据库（test env）
- 目标：关键路径 100% 覆盖
"""

# 两个 fork 独立运行，最后比较结论
```

---

### 4. 恢复会话时告知文件变更

```python
# ❌ 错误：直接继续，不告知变更
# claude --resume auth-investigation "继续分析"
# 问题：模型会引用已过期的文件内容

# ✅ 正确：告知具体变更
resume_prompt = """
继续上次的认证模块调查。

【重要更新：自上次会话以来的变更】
- src/auth/login.py：已重构 validate_token() 函数（逻辑变更）
- src/auth/session.py：新增 refresh_session() 方法
- 其他文件：无变更

请基于以上更新，重新分析这两个文件，然后继续调查。
"""
```

---

### 5. 新会话 + 结构化摘要（最可靠方式）

```python
# 当旧工具结果大量过期时，注入摘要比恢复更可靠

structured_summary = """
【上次调查摘要 - 注入新会话】

已发现：
1. 支付流程中存在竞态条件（src/payment/processor.py:127）
2. get_customer 查询缺少索引（customers 表 email 列）
3. 重试逻辑对幂等 API 不安全

待调查：
- RefundService 与 AuditLog 的集成点
- 分布式锁的实现是否正确

优先级：先解决竞态条件（生产 bug），再优化查询性能
"""

# 新会话直接从摘要出发，避免依赖可能过期的工具调用历史
```

---

## 💡 Claude Code 提示词

```
实现会话管理演示：
1. 模拟 --resume 场景：告知文件变更后继续调查
2. 模拟 fork_session：从同一代码分析基线探索两种架构方案
3. 对比：resume with stale results vs 新会话 + 摘要注入
4. 展示结构化摘要模板

文件名：session_management_demo.py
```

---

## 📋 例题

### 例题 1 ⭐
你正在调查一个认证 bug。上次会话结束后，团队对 `src/auth/validator.py` 进行了重大重构。现在需要继续调查。最佳方案？

**A)** 直接 `--resume` 继续，模型会自动感知文件变更

**B)** `--resume` 继续，并在 prompt 中明确告知 validator.py 已发生变更及变更内容

**C)** 必须开启新会话，因为旧会话的工具结果已过期

**D)** 使用 `fork_session` 创建一个新分支

> **答案：B** — 恢复时需要告知 Agent 已变更的具体文件，以便进行针对性的重新分析，而不是要求完整的重新探索。

---

### 例题 2 ⭐
`fork_session` 最适合哪种场景？

**A)** 在不同的工作时段继续同一调查

**B)** 代码修改后更新 Agent 对新文件的认知

**C)** 从同一代码库分析基线出发，并行探索两种互斥的重构方案

**D)** 当上下文窗口接近上限时压缩历史

> **答案：C** — fork_session 专为从共享分析基线创建独立分支而设计，避免在比较不同方向时互相干扰。

---

*文档版本：v1.0 | Domain 1 Task Statement 1.7*
