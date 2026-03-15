# Claude Code 自定义 CLI 深度指南
## Custom Commands · Skills · Hooks · Shell Integration

> 本文档配合 Claude Code 生成学习代码。每节末尾附有精准的代码生成提示词。

---

## 目录

1. [核心概念：三种扩展方式](#1-核心概念三种扩展方式)
2. [Custom Slash Commands 详解](#2-custom-slash-commands-详解)
3. [Skills 详解（推荐新方式）](#3-skills-详解推荐新方式)
4. [Hooks 详解](#4-hooks-详解)
5. [Shell Alias 集成](#5-shell-alias-集成)
6. [CLAUDE.md 项目配置](#6-claudemd-项目配置)
7. [完整项目实战案例](#7-完整项目实战案例)

---

## 1. 核心概念：三种扩展方式

```
Claude Code 扩展层次
┌────────────────────────────────────────────────────┐
│                                                    │
│  Shell Alias        终端快捷方式，绕过交互模式       │
│  alias ccommit="claude -p '/commit'"               │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Claude Code 会话内                          │  │
│  │                                              │  │
│  │  Slash Commands   /command  手动触发         │  │
│  │  Skills           /skill    手动 + 自动触发  │  │
│  │  Hooks            自动触发（工具调用前后）     │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

### 三种方式对比

| 特性 | Slash Commands | Skills | Hooks |
|------|---------------|--------|-------|
| 触发方式 | 手动输入 `/name` | 手动 `/name` + 自动识别 | 自动（工具事件驱动）|
| 配置文件 | `.md` 文件 | `SKILL.md` 文件 | `settings.json` |
| 携带上下文 | ✅ 支持 shell 命令注入 | ✅ 支持 shell 命令注入 | ✅ 可访问工具输入输出 |
| 支持参数 | ✅ `$ARGUMENTS` / `$1 $2` | ✅ 同左 | ❌ 事件驱动 |
| 推荐场景 | 快速命令 | 复杂工作流 | 质量守门员 |
| 推荐程度 | 旧格式（仍支持）| ⭐ 推荐 | ⭐ 推荐 |

### 文件目录结构总览

```
项目根目录/
├── CLAUDE.md                        ← 项目级 AI 说明书
└── .claude/
    ├── settings.json                ← 权限 + Hooks 配置
    ├── commands/                    ← Slash Commands（旧格式）
    │   ├── review.md                → /review
    │   └── gen-test.md              → /gen-test
    └── skills/                      ← Skills（推荐新格式）
        ├── code-review/
        │   └── SKILL.md             → /code-review
        └── gen-test/
            └── SKILL.md             → /gen-test

~/.claude/                           ← 个人全局（跨所有项目）
├── commands/
│   └── daily-standup.md
└── skills/
    └── my-workflow/
        └── SKILL.md
```

---

## 2. Custom Slash Commands 详解

### 2.1 最简单的 Command

创建 `.claude/commands/hello.md`：

```markdown
Say hello to the user and list the top 3 files in the current directory.
```

使用：在 Claude Code 中输入 `/hello`

---

### 2.2 带参数的 Command

`$ARGUMENTS` 接收命令后的所有文字。

创建 `.claude/commands/explain.md`：

```markdown
Explain the following code or concept in simple terms, with a real-world analogy:

$ARGUMENTS
```

使用：
```
/explain the difference between process and thread
/explain src/payment/FthPaymentProcessor.java
```

**多参数：用 `$1` `$2` 占位**

创建 `.claude/commands/gen-test.md`：

```markdown
---
argument-hint: [class-name] [test-type]
description: Generate tests for a Java class
---

Generate $2 tests for the class $1.
Focus on edge cases and error handling.
Follow existing test patterns in the project.
```

使用：
```
/gen-test FthPaymentProcessor unit
/gen-test PaymentResponseHandler integration
```

---

### 2.3 Shell 命令注入：`!` 语法

`!` 反引号语法在 prompt 发送给 Claude 之前先执行 shell 命令，把输出插入 prompt。

**这是 Custom Command 最强大的特性。**

```
!`shell command`  →  命令输出替换这个占位符
```

创建 `.claude/commands/review-diff.md`：

```markdown
---
description: Review the current git diff
allowed-tools: Read, Bash(git *)
---

Review the following code changes:

## Git Diff
!`git diff HEAD`

## Changed Files
!`git diff --name-only HEAD`

## Recent Commits
!`git log --oneline -5`

Focus on:
1. Logic correctness
2. Error handling completeness
3. Missing test coverage
4. Security issues
```

使用：`/review-diff` — Claude 自动获取当前 diff 内容进行审查

---

### 2.4 Frontmatter 配置项

```markdown
---
description:   命令描述（显示在 /help 中，也帮助 Claude 决定何时自动使用）
argument-hint: [arg1] [arg2]    参数提示（显示在补全中）
allowed-tools: Read, Bash(git *), Write(*.md)   授权工具
model:         claude-opus-4-6  指定使用的模型
agent:         Explore          使用内置子 Agent
---
```

**allowed-tools 语法：**

```
Read             允许读取任何文件
Write(*.java)    只允许写入 .java 文件
Bash(git *)      只允许执行 git 开头的命令
Bash(npm test)   只允许执行 npm test
```

---

### 2.5 File Reference：引用项目文件

```markdown
---
description: Review payment code against our standards
---

Review the payment processing code.

Reference our coding standards:
@CLAUDE.md

Reference the file to review:
@$ARGUMENTS
```

---

> **💡 Claude Code 提示词 — 创建 Slash Commands 套件：**
> ```
> 帮我在当前项目中创建以下 Custom Slash Commands，放在 .claude/commands/ 目录：
>
> 1. review.md → /review [file]
>    - 读取 $ARGUMENTS 指定的文件
>    - 注入 !`git diff HEAD -- $ARGUMENTS` 获取最新改动
>    - 要求 Claude 检查：逻辑正确性、错误处理、测试覆盖
>    - allowed-tools: Read, Bash(git *)
>
> 2. commit.md → /commit
>    - 注入 !`git diff --cached` 获取暂存区内容
>    - 注入 !`git log --oneline -3` 获取最近提交
>    - 要求 Claude 生成符合 Conventional Commits 规范的提交信息
>    - 自动执行 git commit（不需要用户二次确认）
>    - allowed-tools: Bash(git *)
>
> 3. gen-test.md → /gen-test [ClassName] [unit|integration]
>    - 读取对应的源文件
>    - 注入 !`find . -name "*Test*" -type f | head -5` 找到现有测试样例
>    - 生成对应类型的测试代码
>    - allowed-tools: Read, Write(src/test/*)
>
> 4. explain.md → /explain [topic]
>    - 用简单语言解释 $ARGUMENTS
>    - 要求给出代码示例和实际使用场景
>
> 每个文件都要有完整的 frontmatter（description, argument-hint, allowed-tools）
> 创建后运行 /help 验证命令出现在列表中
> ```

---

## 3. Skills 详解（推荐新方式）

Skills 是 Commands 的升级版，支持**自动触发**（Claude 根据对话内容判断何时激活），并支持附带脚本和资源文件。

### 3.1 Skills 目录结构

```
.claude/skills/
└── my-skill/              ← skill 名称（同时是 /my-skill 命令）
    ├── SKILL.md           ← 必须，包含 frontmatter + 指令
    ├── template.java      ← 可选，辅助文件
    └── examples/          ← 可选，示例目录
        └── example1.java
```

### 3.2 SKILL.md 结构

```markdown
---
name:        skill-name          ← /slash-command 名称
description: 描述（⭐ 极为重要，Claude 靠这个决定何时自动激活）
allowed-tools: Read, Write, Bash(git *)
model:       claude-opus-4-6     ← 可选，指定模型
agent:       Explore             ← 可选，使用子 Agent
---

# Skill 指令正文（Markdown 格式）

当这个 skill 被激活时，Claude 会遵循以下指令...
```

### 3.3 description 是核心

description 决定两件事：
1. 显示在 `/help` 列表里的说明
2. **Claude 在对话中自动判断是否激活这个 skill**

```markdown
# ❌ 太模糊，Claude 不知道何时触发
description: Review code

# ✅ 清晰的场景描述，Claude 能精准自动触发
description: >
  Review Java code for correctness, error handling, and test coverage.
  Use when the user asks to review code, check a file, or before committing.
  Also use when the user says "LGTM?" or "is this good?"
```

### 3.4 完整 Skill 示例：代码审查

创建 `.claude/skills/code-review/SKILL.md`：

```markdown
---
name: code-review
description: >
  Perform a thorough code review. Use when the user asks to review code,
  check for bugs, improve quality, or when they paste code and ask "is
  this okay?" or "what do you think?". Also use before git commits.
allowed-tools: Read, Bash(git diff:*), Bash(git log:*)
---

# Code Review Skill

当执行代码审查时，按以下结构输出报告：

## 审查维度

### 1. 逻辑正确性
- 边界条件是否处理？
- 空指针 / null 检查？
- 并发安全？

### 2. 错误处理
- 异常是否被正确捕获和处理？
- 错误信息是否足够清晰？
- 是否有重试机制（对于网络/IO操作）？

### 3. 测试覆盖
- 是否有对应的单元测试？
- 关键路径是否覆盖？

### 4. 安全性
- 是否有 SQL 注入、XSS 等风险？
- 敏感数据是否被正确处理？

## 输出格式

```
## 代码审查报告

### ✅ 优点
- ...

### ⚠️  需要改进
- [严重] ...
- [建议] ...

### 🔧 修改建议
[附具体修改代码]
```
```

### 3.5 Shell 命令注入在 Skills 中

Skills 同样支持 `!` 反引号语法：

```markdown
---
name: pr-summary
description: Summarize a pull request. Use when asked to summarize PR changes.
allowed-tools: Bash(gh *)
---

# PR Summary

## PR 信息
!`gh pr view --json title,body,author`

## 变更文件
!`gh pr diff --name-only`

## Diff 内容
!`gh pr diff`

请总结这个 PR 的主要变更，并指出潜在风险。
```

### 3.6 多参数 + 复杂逻辑 Skill

创建 `.claude/skills/gen-payment-test/SKILL.md`：

```markdown
---
name: gen-payment-test
description: >
  Generate payment system tests. Use when asked to write tests for
  payment processing, MQ consumers, or Finacle API integration code.
argument-hint: [ClassName] [unit|integration|contract]
allowed-tools: Read, Write(src/test/*), Bash(find *), Bash(mvn test:*)
---

# Payment Test Generator

## 目标类
读取文件：src/main/java/**/$1.java

## 现有测试样例（参考风格）
!`find src/test -name "*Test*.java" | head -3`

## 项目测试依赖
!`grep -A5 "<dependencies>" pom.xml | grep -E "junit|mockito|wiremock"`

## 生成要求

根据 $2 类型生成测试：

**unit**: 使用 Mockito mock 所有依赖，覆盖正常流程 + 异常路径
**integration**: 使用 WireMock mock 外部 API，测试完整流程
**contract**: 生成 Pact 消费者契约测试

## 测试场景必须包含
1. 正常成功路径
2. AM04 余额不足错误处理
3. AC01 账户无效错误处理
4. 网络超时重试逻辑
5. 幂等性验证（相同 EndToEndId 重复提交）
```

---

> **💡 Claude Code 提示词 — 创建 Skills 套件：**
> ```
> 帮我创建以下 Skills，放在 .claude/skills/ 目录：
>
> 1. .claude/skills/explain-code/SKILL.md
>    name: explain-code
>    description: 让 Claude 遇到"怎么用/解释/这是什么意思"等问题时自动触发
>    功能：
>    - 先用一个生活中的比喻解释概念
>    - 用 ASCII 图展示结构或流程
>    - 逐步解释代码执行过程
>    - 指出一个常见误区（gotcha）
>
> 2. .claude/skills/refactor/SKILL.md
>    name: refactor
>    description: 遇到重构、优化、clean code 相关请求时触发
>    功能：
>    - 注入 !`git diff HEAD` 了解最近变动
>    - 检查：命名清晰度、单一职责、重复代码、复杂度
>    - 给出重构前后对比
>    - allowed-tools: Read, Write, Bash(git diff:*)
>
> 3. .claude/skills/debug/SKILL.md
>    name: debug
>    description: 遇到 bug、错误、为什么不工作等问题时触发
>    功能：
>    - 注入 !`git log --oneline -5` 了解最近改动
>    - 分析错误原因（从症状到根因）
>    - 给出修复方案 + 验证方法
>    - allowed-tools: Read, Bash(git log:*)
>
> 每个 SKILL.md 都要有 YAML frontmatter + 详细的中文指令内容
> 创建后用自然语言触发测试（不用 /slash 命令）：
>   "帮我解释一下这段代码"  → 应该触发 explain-code
>   "这段代码有什么问题"    → 应该触发 debug
> ```

---

## 4. Hooks 详解

Hooks 是**事件驱动的自动化**。在 Claude 执行工具前后自动运行 shell 脚本，用于质量检查、格式化、通知等。

### 4.1 Hook 事件类型

```
事件触发时机：

PreToolUse    Claude 准备调用工具之前
              → 可用于：权限检查、危险操作拦截

PostToolUse   Claude 调用工具之后
              → 可用于：格式化、lint、测试、通知

Notification  Claude 需要用户注意时（等待输入等）
              → 可用于：发送桌面通知

Stop          Claude 完成任务停止时
              → 可用于：汇总报告、清理工作
```

### 4.2 配置文件：`.claude/settings.json`

```json
{
  "permissions": {
    "allow": ["Read", "Write", "Bash(git *)"],
    "deny": ["Bash(rm -rf *)"]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write(*.py)",
        "hooks": [
          {
            "type": "command",
            "command": "black $file && echo 'Formatted: $file'"
          }
        ]
      }
    ]
  }
}
```

### 4.3 matcher 语法

```json
// 匹配写入任何 Java 文件
"matcher": "Write(*.java)"

// 匹配写入 src/main 下的文件
"matcher": "Write(src/main/**)"

// 匹配所有 Bash 工具调用
"matcher": "Bash(*)"

// 匹配特定 git 命令
"matcher": "Bash(git commit*)"

// 匹配任何工具
"matcher": "*"
```

### 4.4 实用 Hook 配置示例

#### Java 项目 Hook（格式化 + 编译检查）

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write(*.java)",
        "hooks": [
          {
            "type": "command",
            "command": "cd $(git rev-parse --show-toplevel) && mvn spotless:apply -q && echo '✅ Java formatted'"
          }
        ]
      },
      {
        "matcher": "Write(src/main/**/*.java)",
        "hooks": [
          {
            "type": "command",
            "command": "cd $(git rev-parse --show-toplevel) && mvn compile -q 2>&1 | tail -5"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash(git push*)",
        "hooks": [
          {
            "type": "command",
            "command": "cd $(git rev-parse --show-toplevel) && mvn test -q && echo '✅ Tests passed, safe to push'"
          }
        ]
      }
    ]
  }
}
```

#### Python 项目 Hook（black + mypy）

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write(*.py)",
        "hooks": [
          {
            "type": "command",
            "command": "black $file && echo '✅ Formatted'"
          },
          {
            "type": "command",
            "command": "mypy $file --ignore-missing-imports 2>&1 | tail -3"
          }
        ]
      }
    ]
  }
}
```

#### 桌面通知 Hook（Claude 等待输入时提醒你）

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Claude needs your attention\" with title \"Claude Code\"'"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Task completed!\" with title \"Claude Code\"'"
          }
        ]
      }
    ]
  }
}
```

#### 安全拦截 Hook（防止危险操作）

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash(rm *)",
        "hooks": [
          {
            "type": "command",
            "command": "echo '⚠️  rm command intercepted. Use trash instead.' && exit 1"
          }
        ]
      },
      {
        "matcher": "Write(.env*)",
        "hooks": [
          {
            "type": "command",
            "command": "echo '🚫 Writing to .env files is blocked for safety.' && exit 1"
          }
        ]
      }
    ]
  }
}
```

### 4.5 Hook 环境变量

Hook 脚本可以访问以下环境变量：

```bash
$file          # 被写入/读取的文件路径（Write/Read 工具）
$CLAUDE_FILE   # 同上
$command       # 执行的命令内容（Bash 工具）
$tool_name     # 工具名称（Write / Read / Bash 等）
```

---

> **💡 Claude Code 提示词 — 配置 Hooks：**
> ```
> 帮我在当前 Python 项目中创建 .claude/settings.json，配置以下 Hooks：
>
> PostToolUse Hooks：
> 1. 写入 *.py 时：自动运行 black 格式化 + ruff lint 检查
>    只打印警告，不阻断（exit 0）
>
> 2. 写入 src/**/*.py 时：运行 mypy 类型检查
>    只打印前 10 行错误，不阻断
>
> 3. 写入 tests/**/*.py 时：自动运行 pytest 执行该测试文件
>    如果测试失败，打印失败信息
>
> PreToolUse Hooks：
> 4. 执行 git push 之前：运行完整测试套件 pytest
>    如果失败则阻断（exit 1）并提示 "Tests must pass before push"
>
> 5. 写入 .env* 文件时：阻断并提示不允许修改环境配置
>
> Stop Hook：
> 6. Claude 完成任务时：打印 "✅ Claude Code task completed at [时间]"
>
> permissions 同时配置：
> - allow: Read, Write(不包括 .env), Bash(git *), Bash(pytest *), Bash(black *), Bash(ruff *)
> - deny: Write(.env), Write(.env.*)
>
> 创建后测试：让 Claude 写一个有格式问题的 Python 文件，验证 Hook 自动触发
> ```

---

## 5. Shell Alias 集成

用 `claude -p` 的非交互模式，把 Claude Code 命令嵌入日常 shell workflow。

### 5.1 基础用法：`-p` 非交互模式

```bash
# -p: print mode，发送 prompt 后直接退出，不开交互会话
claude -p "Explain this error: $(cat error.log)"
claude -p "/commit"
claude -p "/review src/main/Payment.java"
```

### 5.2 推荐 Alias 配置

加入 `~/.zshrc` 或 `~/.bashrc`：

```bash
# ── 基础 Alias ─────────────────────────────────────────
# 快速提交（自动生成 commit message）
alias ccommit='claude -p "/commit"'

# 快速 lint
alias clint='claude -p "/lint"'

# 快速审查当前 diff
alias creview='claude -p "/review-diff"'

# ── 带参数的 Alias ─────────────────────────────────────
# 解释文件
cexplain() {
  claude -p "Explain this file in simple terms: $(cat $1)"
}

# 审查指定文件
creview_file() {
  claude -p "/review $1"
}

# 生成测试
cgentest() {
  claude -p "/gen-test $1 ${2:-unit}"
}

# ── Pipeline 集成 ─────────────────────────────────────
# 解释错误日志
canalyze_error() {
  cat $1 | claude -p "Analyze this error log and suggest fixes:"
}

# 总结 git 变更
cgit_summary() {
  git diff HEAD | claude -p "Summarize these code changes in 3 bullet points:"
}

# ── 实用组合 ──────────────────────────────────────────
# 提交前完整检查（lint + review + commit）
csmart_commit() {
  echo "Running lint..." && claude -p "/lint" && \
  echo "Reviewing diff..." && claude -p "/review-diff" && \
  echo "Generating commit..." && claude -p "/commit"
}
```

### 5.3 在 CI/CD 中使用

```bash
#!/bin/bash
# ci-review.sh：在 CI 流水线中自动 code review

# 获取 PR diff
PR_DIFF=$(git diff origin/main...HEAD)

# 调用 Claude Code 审查
REVIEW_RESULT=$(echo "$PR_DIFF" | claude -p "
Review this PR diff for:
1. Security vulnerabilities
2. Logic errors
3. Missing error handling
Output as JSON: {\"passed\": bool, \"issues\": []}
")

# 解析结果
PASSED=$(echo $REVIEW_RESULT | python3 -c "import sys,json; print(json.load(sys.stdin)['passed'])")

if [ "$PASSED" = "False" ]; then
  echo "❌ Code review failed"
  echo $REVIEW_RESULT
  exit 1
fi

echo "✅ Code review passed"
```

---

> **💡 Claude Code 提示词 — Shell 集成配置：**
> ```
> 帮我创建一个完整的 Claude Code Shell 集成配置：
>
> 1. 创建 ~/.claude_aliases 文件，包含以下 alias 和函数：
>
>    基础命令：
>    - ccommit   → claude -p "/commit"
>    - creview   → claude -p "/review-diff"
>    - clint     → claude -p "/lint"
>    - chelp     → claude -p "/help"（展示所有可用命令）
>
>    带参数函数：
>    - cexplain <file>       解释文件内容
>    - creview_file <file>   审查指定文件
>    - cfix <error_msg>      根据错误信息给出修复方案
>    - ctest <class>         为指定类生成测试
>
>    组合函数：
>    - csmart_push()
>      步骤：1. 运行 lint  2. 运行测试  3. 审查 diff  4. 生成 commit message  5. git push
>      每步失败则停止
>
> 2. 在文件末尾加说明：如何把 source ~/.claude_aliases 加入 .zshrc/.bashrc
>
> 3. 创建一个 test_aliases.sh 脚本，验证每个 alias 能正确调用
> ```

---

## 6. CLAUDE.md 项目配置

`CLAUDE.md` 是项目的 AI 说明书。每次 Claude Code 在项目中启动时自动读取，给 Claude 提供持久上下文。

### 6.1 CLAUDE.md 的作用

```
没有 CLAUDE.md：
  Claude 每次都要重新了解项目结构、构建方式、约定...

有 CLAUDE.md：
  Claude 启动即知道：
    ✅ 项目是什么、技术栈是什么
    ✅ 怎么构建、运行、测试
    ✅ 代码规范和约定
    ✅ 哪些文件/目录要注意
    ✅ 常见问题和解决方案
```

### 6.2 CLAUDE.md 模板

```markdown
# 项目名称

## 项目概述
[一段话描述这个项目是什么，解决什么问题]

## 技术栈
- 语言：Java 17
- 框架：Spring Boot 3.x, Apache Camel 4.x
- 消息队列：IBM MQ
- 数据库：Oracle 19c
- 测试：JUnit 5, Mockito, WireMock

## 快速开始

### 构建
```bash
mvn clean compile
```

### 运行测试
```bash
mvn test                    # 所有测试
mvn test -Dtest=ClassName   # 单个类
```

### 本地启动
```bash
mvn spring-boot:run -Dspring.profiles.active=local
```

## 项目结构

```
src/
├── main/java/
│   ├── routes/          # Apache Camel 路由
│   ├── processors/      # 消息处理器
│   ├── services/        # 业务逻辑
│   └── config/          # 配置类
└── test/java/
    ├── unit/
    └── integration/
```

## 代码规范

- 所有 public 方法必须有 Javadoc
- 异常必须被记录到 log（不允许 catch 后静默）
- 新增 API 必须有对应的 WireMock 测试
- 支付金额使用 `BigDecimal`，禁止 `double`

## 重要文件

- `src/main/resources/application.yml`：主配置
- `src/main/resources/routes/`：Camel 路由 XML
- `CHANGELOG.md`：版本变更记录

## 常见问题

### IBM MQ 连接失败
检查 `application-local.yml` 中的 MQ 配置，确保本地 MQ 服务已启动。

### 测试数据库问题
集成测试使用 H2 内存数据库，配置在 `src/test/resources/application-test.yml`。

## 禁止事项

- ❌ 不要修改 `src/main/resources/mq-config.xml`（手动维护）
- ❌ 不要使用 `System.out.println`（使用 `log.info()`）
- ❌ 不要在 main 分支直接提交
```

### 6.3 层级 CLAUDE.md

可以在子目录放置额外的 CLAUDE.md，针对特定模块提供更精细的上下文：

```
项目根/
├── CLAUDE.md              ← 全局说明
└── src/
    ├── main/
    │   └── CLAUDE.md      ← 主代码特定说明
    └── test/
        └── CLAUDE.md      ← 测试特定说明（测试规范、Mock 策略等）
```

---

> **💡 Claude Code 提示词 — 生成 CLAUDE.md：**
> ```
> 分析当前项目，自动生成一个完整的 CLAUDE.md 文件：
>
> 1. 扫描项目结构（目录树、主要文件）
> 2. 检测技术栈（package.json / pom.xml / requirements.txt / go.mod）
> 3. 提取构建和测试命令
> 4. 分析现有代码推断代码规范（命名、注释风格）
> 5. 识别重要配置文件和入口文件
>
> 生成的 CLAUDE.md 包含以下章节：
> - 项目概述（1-2 句话）
> - 技术栈（带版本号）
> - 快速开始（构建/运行/测试命令）
> - 项目结构（带注释的目录树）
> - 代码规范（从代码中推断）
> - 重要文件（值得特别关注的文件）
> - 常见问题（如果能从 README 或代码注释推断）
> - 禁止事项（关键的不能做的事）
>
> 同时生成 src/test/CLAUDE.md，专门描述测试约定
> ```

---

## 7. 完整项目实战案例

### 案例：FTH 支付系统 Claude Code 配置

**目标：** 为 Westpac FTH 项目配置完整的 Claude Code 工作流自动化

#### 最终目录结构

```
fth-payment-system/
├── CLAUDE.md                          ← FTH 项目 AI 说明书
└── .claude/
    ├── settings.json                  ← 权限 + Hooks
    ├── commands/                      ← 快速命令
    │   ├── review-payment.md          → /review-payment
    │   └── check-iso20022.md          → /check-iso20022
    └── skills/
        ├── payment-test-gen/          → /payment-test-gen
        │   └── SKILL.md
        ├── camel-route-review/        → /camel-route-review
        │   └── SKILL.md
        └── fth-debug/                 → /fth-debug
            └── SKILL.md
```

#### CLAUDE.md（FTH 专用）

```markdown
# FTH - Funds Transfer Hub

## 项目概述
Westpac NZ 的国际汇款处理系统，处理 IMT/TTP 支付流程。
通过 IBM MQ 接收支付指令，调用 Finacle API 执行，通过 ISO 20022 消息格式通信。

## 技术栈
- Java 17, Spring Boot 3.x
- Apache Camel 4.x（消息路由）
- IBM MQ（消息队列）
- Oracle 19c（持久化）
- Finacle API（核心银行系统）
- ISO 20022（支付消息标准）
- WireMock（API Mock 测试）
- JUnit 5 + Mockito

## 关键业务规则
- 所有支付金额使用 BigDecimal，精度 2 位小数
- EndToEndId 是幂等键，相同 ID 的请求只处理一次
- AM04（余额不足）最多重试 3 次，间隔 5 分钟
- AC01（无效账户）不重试，直接返回失败
- 所有 Finacle API 调用必须记录 correlationId

## 目录结构
src/main/java/
├── routes/         Apache Camel 路由（消息流转）
├── processors/     消息处理器（核心业务逻辑）
├── services/       服务层（Finacle API 调用）
├── model/          ISO 20022 数据模型
└── config/         配置（MQ, DB, Finacle）

## 构建与测试
mvn clean test              # 全量测试
mvn test -Dtest=*Payment*   # 支付相关测试
mvn spring-boot:run -Dspring.profiles.active=local

## 禁止事项
- ❌ 不要修改 EndToEndId 的生成逻辑（幂等性依赖）
- ❌ 支付金额禁止使用 double 或 float
- ❌ Finacle API 调用必须在 service 层，不能在 processor 直接调用
```

#### `/review-payment` 命令

`.claude/commands/review-payment.md`:

```markdown
---
description: Review payment processing code for FTH-specific requirements
argument-hint: [file-path]
allowed-tools: Read, Bash(git diff:*), Bash(grep *)
---

Review the payment code in $ARGUMENTS for FTH-specific requirements:

## 代码内容
@$ARGUMENTS

## 最近变更
!`git diff HEAD -- $ARGUMENTS`

## 审查清单

### 幂等性
- [ ] EndToEndId 是否作为幂等键使用？
- [ ] 重复请求是否有检查和处理？

### 错误处理
- [ ] AM04 错误是否有重试逻辑（最多3次）？
- [ ] AC01 错误是否直接拒绝（不重试）？
- [ ] 每个 Finacle 调用是否有 try-catch？

### ISO 20022 合规
- [ ] 金额字段是否使用 BigDecimal？
- [ ] MsgId / EndToEndId / PmtInfId 是否正确填充？

### 日志
- [ ] correlationId 是否在每条日志中携带？
- [ ] 是否有足够的 INFO/ERROR 日志？
```

#### `payment-test-gen` Skill

`.claude/skills/payment-test-gen/SKILL.md`:

```markdown
---
name: payment-test-gen
description: >
  Generate payment system tests for FTH. Use when asked to write tests
  for payment processors, MQ consumers, Finacle API integration, or
  ISO 20022 message handling. Also use when user says "add tests" or
  "write unit tests" for payment-related classes.
argument-hint: [ClassName] [unit|integration]
allowed-tools: Read, Write(src/test/**), Bash(find src/test *)
---

# FTH Payment Test Generator

## 读取目标类
!`find src/main -name "$1.java" 2>/dev/null | head -1`

## 参考现有测试风格
!`find src/test -name "*Test.java" | head -3`

## 生成 $2 测试

根据 FTH 项目规范生成测试：

### 测试结构
- 使用 @ExtendWith(MockitoExtension.class)
- Mock 所有外部依赖（FincaleService, MQTemplate, Repository）
- 每个测试方法名格式：should_[预期结果]_when_[条件]

### 必须覆盖的场景
1. **正常路径**：完整支付流程成功
2. **AM04 重试**：余额不足，验证重试 3 次后失败
3. **AC01 拒绝**：无效账户，验证不重试直接失败
4. **幂等性**：相同 EndToEndId 第二次调用不执行实际支付
5. **超时处理**：Finacle API 超时时的行为
6. **日志验证**：关键步骤的 correlationId 日志存在

### WireMock（集成测试）
如果是 integration 类型，使用 WireMock mock Finacle API：
- 成功响应：HTTP 200 + 正确 JSON
- AM04 响应：HTTP 200 + 错误 body
- 超时：WireMock delay
```

#### `settings.json`（Hooks 配置）

`.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Write(src/**)",
      "Bash(git *)",
      "Bash(mvn *)",
      "Bash(grep *)",
      "Bash(find *)"
    ],
    "deny": [
      "Write(src/main/resources/mq-config.xml)",
      "Bash(git push --force*)"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write(*.java)",
        "hooks": [
          {
            "type": "command",
            "command": "cd $(git rev-parse --show-toplevel) && mvn spotless:apply -pl . -q 2>&1 && echo '✅ Java formatted'"
          }
        ]
      },
      {
        "matcher": "Write(src/main/**/*.java)",
        "hooks": [
          {
            "type": "command",
            "command": "cd $(git rev-parse --show-toplevel) && mvn compile -q 2>&1 | tail -5 && echo '✅ Compiled'"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash(git push*)",
        "hooks": [
          {
            "type": "command",
            "command": "cd $(git rev-parse --show-toplevel) && mvn test -q && echo '✅ All tests passed'"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"\\n✅ Claude Code completed at $(date '+%H:%M:%S')\""
          }
        ]
      }
    ]
  }
}
```

---

> **💡 Claude Code 综合提示词 — 完整 FTH 配置：**
> ```
> 帮我为当前 Java/Maven 项目创建完整的 Claude Code 工作流配置：
>
> 1. CLAUDE.md
>    - 分析项目结构，自动填充技术栈
>    - 包含：构建命令、测试命令、目录说明、代码规范
>    - 特别注明：关键业务规则（从现有代码注释推断）
>
> 2. .claude/settings.json
>    - permissions: 允许读写 src/**，允许 git/mvn 命令，禁止强制推送
>    - PostToolUse hooks:
>      a. 写入 *.java 时自动格式化（如果有 spotless 插件）
>      b. 写入 src/main/*.java 时自动编译检查
>    - PreToolUse hooks:
>      a. git push 前自动运行 mvn test
>    - Stop hook: 完成时打印时间戳
>
> 3. .claude/commands/ 目录（3个命令）：
>    a. review.md → /review [file]
>       - 注入 git diff
>       - 检查：空指针、异常处理、日志完整性
>    b. commit.md → /commit
>       - 注入 git diff --cached
>       - 生成 Conventional Commits 格式 message，自动执行
>    c. gen-test.md → /gen-test [Class] [unit|integration]
>       - 读取源文件，参考现有测试，生成测试代码
>
> 4. .claude/skills/ 目录（2个 skill）：
>    a. skills/debug/SKILL.md
>       - 自动触发：遇到错误、异常、"为什么不工作"
>       - 注入 git log --oneline -5
>       - 分析根因 + 修复方案
>    b. skills/explain/SKILL.md
>       - 自动触发：遇到"解释/explain/这是什么"
>       - 比喻 + ASCII 图 + 逐步解释 + gotcha
>
> 创建完所有文件后：
> 1. 运行 /help 展示所有可用命令
> 2. 测试 /review 命令（审查一个现有文件）
> 3. 写一个有格式问题的 Java 文件，验证 Hook 触发
> ```

---

## 附录：常用命令速查

### 内置 Slash Commands

| 命令 | 作用 |
|------|------|
| `/help` | 显示所有可用命令（含自定义）|
| `/clear` | 清除对话历史 |
| `/compact` | 压缩对话历史（节省 token）|
| `/model` | 切换 LLM 模型 |
| `/context` | 查看当前 context 使用情况 |
| `/permissions` | 查看和配置工具权限 |
| `/hooks` | 通过交互菜单配置 Hooks |
| `/init` | 初始化项目（生成 CLAUDE.md）|

### CLI Flags 速查

```bash
claude                      # 进入交互模式
claude -p "prompt"          # 非交互，执行后退出
claude -p "/command arg"    # 执行自定义命令后退出
claude --model opus         # 指定模型
claude --no-stream          # 不流式输出（脚本用）
claude --verbose            # 详细日志
```

### 文件路径速查

```
作用域        路径
─────────────────────────────────────────────────
项目命令      .claude/commands/*.md
项目 Skills   .claude/skills/*/SKILL.md
项目配置      .claude/settings.json
项目 AI 说明  CLAUDE.md
子目录 AI 说明 任意子目录/CLAUDE.md

个人全局命令  ~/.claude/commands/*.md
个人全局 Skills ~/.claude/skills/*/SKILL.md
```
