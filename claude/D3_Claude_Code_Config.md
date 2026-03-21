# Domain 3 — Claude Code 配置与工作流
> Task Statements 3.1–3.6 | 考试权重：20%

---

## 3.1 CLAUDE.md 配置层次结构
> ⭐⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 配置层次 | Configuration Hierarchy |
| 用户级配置 | User-level (`~/.claude/CLAUDE.md`) |
| 项目级配置 | Project-level (`.claude/CLAUDE.md` / root `CLAUDE.md`) |
| 目录级配置 | Directory-level (subdirectory `CLAUDE.md`) |
| 导入语法 | `@import` syntax |
| 规则目录 | `.claude/rules/` |

### 三级层次结构

```
~/.claude/CLAUDE.md          ← 用户级（只对本人有效，不入版本控制）
    ↓ 叠加
.claude/CLAUDE.md            ← 项目级（团队共享，入版本控制）
    ↓ 叠加
src/api/CLAUDE.md            ← 目录级（仅对该目录生效）
```

**关键记忆点**：
- 用户级 = 个人偏好，不共享
- 项目级 = 团队标准，必须共享
- 目录级 = 子系统特定规范

### @import 模块化

```markdown
# .claude/CLAUDE.md（项目根）

# 通用标准（所有包适用）
@import .claude/standards/code-style.md
@import .claude/standards/security.md

# 支付模块特定标准（只在相关包中导入）
# 在 src/payment/CLAUDE.md 中：
@import .claude/standards/pci-dss-compliance.md
@import .claude/standards/iso20022-format.md
```

### .claude/rules/ 目录

```
.claude/
├── CLAUDE.md           ← 通用规则
└── rules/
    ├── testing.md      ← 测试规范
    ├── api-conventions.md  ← API 规范
    └── deployment.md   ← 部署规范
```

每个 rules 文件通过 YAML frontmatter 指定路径作用域（见 3.3）

---

### 📋 例题

**例题 ⭐⭐⭐**

新团队成员反馈 Claude Code 没有应用项目的代码规范。检查后发现规范在 `~/.claude/CLAUDE.md` 中。如何修复？

**A)** 让每个成员手动复制 `~/.claude/CLAUDE.md` 内容

**B)** 将规范迁移到项目根目录的 `CLAUDE.md` 或 `.claude/CLAUDE.md` 并提交到版本控制

**C)** 在 `.mcp.json` 中引用 `~/.claude/CLAUDE.md`

**D)** 用户级配置自动同步到团队成员

> **答案：B** — 用户级配置不共享。团队规范必须在项目级配置中，通过版本控制分发给所有成员。

---

## 3.2 自定义 Slash 命令与 Skills
> ⭐⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 项目作用域命令 | Project-scoped commands (`.claude/commands/`) |
| 用户作用域命令 | User-scoped commands (`~/.claude/commands/`) |
| Skill 定义文件 | `SKILL.md` |
| Fork 上下文 | `context: fork` |
| 允许工具列表 | `allowed-tools` |
| 参数提示 | `argument-hint` |

### Slash 命令 vs Skills 区别

```
/custom-command（Slash 命令）
  → 在 .claude/commands/ 中定义
  → 直接执行 prompt 模板
  → 适合：团队标准化工作流（代码审查、PR 描述）

Skill（技能）
  → 在 .claude/skills/ 中定义，带 SKILL.md frontmatter
  → 支持更多配置选项（fork、allowed-tools）
  → 适合：复杂的任务特定工作流
```

### Skills SKILL.md frontmatter

```markdown
---
context: fork          # 在隔离子 Agent 中运行，不污染主对话
allowed-tools:         # 限制 skill 执行期间可用的工具
  - Write
  - Edit
argument-hint: "输入要分析的模块名称"  # 无参数调用时的提示
---

# Skill：代码库分析

分析 $ARGUMENTS 模块的依赖关系和架构...
```

**`context: fork` 的作用**：

```
没有 fork：
  主对话 → 调用 Skill → Skill 产生大量探索输出 → 污染主对话上下文

有 context: fork：
  主对话 → 调用 Skill → [独立子 Agent 运行 Skill] → 只返回摘要 → 主对话继续
```

### 文件位置

```
.claude/
├── commands/
│   ├── review.md        ← /review 命令（团队共享）
│   └── pr-description.md
└── skills/
    ├── analyze-module/
    │   └── SKILL.md     ← analyze-module skill（团队共享）
    └── generate-tests/
        └── SKILL.md

~/.claude/
├── commands/            ← 个人命令（不共享）
└── skills/              ← 个人 skill 变体（不共享）
```

---

### 📋 例题

**例题 ⭐⭐⭐**

你创建了一个 `/review` slash 命令，用于运行团队的标准代码审查清单。要让所有开发者克隆仓库后即可使用，应该放在哪里？

**A)** `.claude/commands/`（项目仓库中）

**B)** `~/.claude/commands/`（每个开发者的主目录）

**C)** 根目录的 `CLAUDE.md` 文件中

**D)** `.claude/config.json` 的 commands 数组中

> **答案：A** — 项目作用域命令存储在 `.claude/commands/`，通过版本控制共享，开发者克隆后自动可用。B 是个人命令，C 用于指令而非命令定义，D 不存在。

---

## 3.3 路径特定规则：条件加载
> ⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 路径作用域规则 | Path-scoped Rules |
| YAML frontmatter | YAML frontmatter `paths` field |
| Glob 模式 | Glob Pattern |
| 条件加载 | Conditional Loading |

### 实现示例

```markdown
---
# .claude/rules/testing.md
paths:
  - "**/*.test.tsx"
  - "**/*.spec.ts"
  - "src/__tests__/**"
---

# 测试文件规范

所有测试必须遵循以下规范：
- 使用 React Testing Library（不用 Enzyme）
- 测试描述使用中文
- 每个组件至少覆盖：渲染、交互、边缘情况
```

```markdown
---
# .claude/rules/terraform.md
paths:
  - "terraform/**/*"
  - "infra/**/*.tf"
---

# Terraform 规范

- 所有资源必须有 tags 块
- 使用 local.common_tags 合并通用标签
```

### 路径规则 vs 目录级 CLAUDE.md

```
问题场景：测试文件分散在整个代码库
  src/components/Button.test.tsx
  src/hooks/useAuth.test.ts
  src/utils/format.test.ts
  ...（数十个目录）

❌ 目录级 CLAUDE.md：需要在每个目录都放一个文件
✅ 路径规则 + glob：一个文件，用 **/*.test.tsx 匹配所有测试文件
```

---

## 3.4 Plan 模式 vs 直接执行
> ⭐⭐⭐ 高频考点

### 决策矩阵

```
任务特征                          → 推荐模式
─────────────────────────────────────────────
架构决策、多个有效方案              → Plan 模式
影响 45+ 文件                     → Plan 模式
微服务重构、库迁移                  → Plan 模式
不同基础设施需求的方案比较           → Plan 模式

单文件 bug 修复（有清晰堆栈跟踪）    → 直接执行
添加一个验证检查                    → 直接执行
明确范围的变更                      → 直接执行
```

### Explore 子 Agent

```python
# 多阶段任务中使用 Explore 防止上下文窗口耗尽

# ❌ 直接探索（占用主会话上下文）
def migrate_library_direct():
    explore_all_usages()    # 产生大量输出，填满上下文
    create_migration_plan() # 可能上下文不足

# ✅ Explore 子 Agent（隔离探索输出）
def migrate_library_with_explore():
    # Plan 阶段：用 Explore 子 Agent 进行探索
    summary = explore_subagent(
        task="分析所有 lodash 使用情况并返回摘要"
        # Explore 产生大量输出，但只返回摘要给主会话
    )
    
    # 执行阶段：基于摘要直接执行迁移
    execute_migration(plan=summary)
```

---

### 📋 例题

**例题 ⭐⭐⭐（对应考试 Question 5）**

你被分配重构团队的单体应用为微服务，涉及数十个文件的变更，需要决定服务边界和模块依赖。应采用哪种方法？

**A)** 进入 plan 模式探索代码库、理解依赖关系，设计实现方案后再做变更

**B)** 开始直接执行，增量变更，让实现揭示自然的服务边界

**C)** 用直接执行配合详细的前期指令说明每个服务如何构建

**D)** 从直接执行开始，遇到意外复杂性再切换到 plan 模式

> **答案：A** — Plan 模式适合大规模变更、多个有效方案和架构决策。B 在依赖关系后期发现时会有代价高昂的返工。D 忽略了复杂性已经在需求中明确说明，不是"可能出现"的事情。

---

## 3.5 迭代精化技术
> ⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 输入/输出示例 | Input/Output Examples |
| 测试驱动迭代 | Test-driven Iteration |
| 访谈模式 | Interview Pattern |
| 相互影响的问题 | Interacting Problems |
| 独立问题 | Independent Problems |

### 核心技巧

```python
# 技巧一：输入/输出示例（散文描述不一致时）
# ❌ 模糊描述
prompt_bad = "将日期格式标准化"

# ✅ 具体示例
prompt_good = """
请标准化日期格式。示例：
  输入: "Mar 15, 2024"  → 输出: "2024-03-15"
  输入: "15/03/2024"    → 输出: "2024-03-15"
  输入: "2024.03.15"    → 输出: "2024-03-15"
  边缘情况: null/空字符串 → 输出: null
"""


# 技巧二：访谈模式（在实现前发现未知考虑因素）
interview_prompt = """
我需要实现一个缓存系统。
在开始实现之前，请先问我几个关键问题，
帮助发现我可能没有考虑到的设计决策。
"""
# Claude 会问：缓存失效策略？最大容量？并发访问？分布式？


# 技巧三：相互影响 vs 独立问题的处理
# 相互影响（一次提供所有问题）：
message_interacting = """
请同时修复以下三个相关问题：
1. refund_amount 验证逻辑
2. audit_log 记录时机（必须在验证后）
3. error_message 格式（依赖验证结果）
"""

# 独立问题（逐一解决）：
message_independent_1 = "修复 button 组件的样式问题"
message_independent_2 = "添加 API 文档注释"  # 与上面无关，分开处理
```

---

## 3.6 Claude Code 集成到 CI/CD
> ⭐⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 非交互模式 | Non-interactive Mode |
| `-p` / `--print` 标志 | `-p` / `--print` flag |
| 结构化输出标志 | `--output-format json` + `--json-schema` |
| 会话上下文隔离 | Session Context Isolation |

### CI 集成关键命令

```bash
# ❌ 错误：不加 -p，在 CI 中会无限等待交互输入
claude "Analyze this PR for security issues"

# ✅ 正确：-p 标志，非交互模式，处理完成后退出
claude -p "Analyze this PR for security issues"

# 结构化输出（用于自动发布 PR 评论）
claude -p "Review this code" \
  --output-format json \
  --json-schema ./review-schema.json
```

### 会话隔离的重要性

```
生成代码的 Claude 实例：
  → 记得自己的设计决策
  → 不太可能质疑自己的选择
  → 审查自己代码时效果差

独立审查 Claude 实例：
  → 没有生成时的推理上下文
  → 更可能发现细微问题
  → 更有效的代码审查
```

### 避免重复 PR 评论

```bash
# 重新运行审查时，包含之前的发现
claude -p "
Code changes: [diff]

Previous review findings (already posted):
[previous findings]

IMPORTANT: Only report NEW issues or STILL-UNADDRESSED issues.
Do not repeat issues that were already fixed.
"
```

---

### 📋 例题

**例题 1 ⭐⭐⭐（对应考试 Question 10）**

CI 脚本运行 `claude "Analyze this PR for security issues"` 但 job 无限挂起。日志显示 Claude Code 在等待交互输入。正确方案？

**A)** 添加 `-p` 标志：`claude -p "Analyze this PR for security issues"`

**B)** 设置环境变量 `CLAUDE_HEADLESS=true`

**C)** 重定向 stdin：`claude "..." < /dev/null`

**D)** 添加 `--batch` 标志

> **答案：A** — `-p`（`--print`）是运行 Claude Code 非交互模式的官方方式。B、D 引用不存在的功能，C 是 Unix 变通方法，不能正确处理 Claude Code 的命令语法。

---

**例题 2 ⭐⭐（对应考试 Question 11）**

你的团队想降低 API 成本。当前两个工作流都用实时调用：(1) 阻塞式合并前检查（开发者等待结果）；(2) 隔夜技术债务报告。经理建议都切换到 Message Batches API（50% 成本节省）。如何评估？

**A)** 只将技术债务报告切换到批量处理；合并前检查保留实时调用

**B)** 两个工作流都切换到批量处理，用状态轮询检查完成情况

**C)** 两个工作流都保留实时调用，避免批量结果排序问题

**D)** 都切换到批量，加超时回退到实时调用

> **答案：A** — Message Batches API 有最多 24 小时处理时间，无保证延迟 SLA，不适合阻塞式合并前检查。但非常适合隔夜批处理作业。

---

*文档版本：v1.0 | Domain 3 Task Statements 3.1–3.6*
