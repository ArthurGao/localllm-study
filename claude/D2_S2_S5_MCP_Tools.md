# Domain 2 · 3.2–3.5 — MCP 工具设计：错误处理、工具分配、服务器配置、内置工具
> Task Statements 2.2–2.5

---

## 3.2 MCP 工具结构化错误响应
> ⭐⭐⭐ 高频考点

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 错误标志 | `isError` flag |
| 错误分类 | `errorCategory` |
| 可重试标志 | `isRetryable` |
| 瞬时错误 | Transient Error（超时、服务不可用）|
| 验证错误 | Validation Error（无效输入）|
| 业务错误 | Business Error（策略违规）|
| 权限错误 | Permission Error |

### 核心知识点

**四种错误类型与处理策略**：

```python
# ✅ 结构化错误响应模板
def build_error_response(error_type: str, details: str) -> dict:
    
    error_configs = {
        "transient": {
            "errorCategory": "transient",
            "isRetryable": True,
            "userMessage": "服务暂时不可用，正在重试...",
            "isError": True
        },
        "validation": {
            "errorCategory": "validation",
            "isRetryable": False,
            "userMessage": f"输入格式无效：{details}",
            "isError": True
        },
        "business": {
            "errorCategory": "business",
            "isRetryable": False,
            "retriable": False,
            "userMessage": f"操作不符合业务规则：{details}",
            "isError": True
        },
        "permission": {
            "errorCategory": "permission",
            "isRetryable": False,
            "userMessage": "权限不足，请联系管理员",
            "isError": True
        }
    }
    
    return error_configs.get(error_type, error_configs["transient"])


# ❌ 错误：统一错误响应阻止 Agent 做出恢复决策
def bad_error_response():
    return {"error": "Operation failed"}  # 模型不知道是否应该重试


# ✅ 正确：结构化错误让 Agent 做出正确决策
def good_error_response_transient():
    return {
        "isError": True,
        "errorCategory": "transient",
        "isRetryable": True,
        "attemptedAction": "lookup_order #12345",
        "partialResults": None,
        "suggestedAlternative": "稍后重试或使用备用数据源"
    }

def good_error_response_business():
    return {
        "isError": True,
        "errorCategory": "business",
        "isRetryable": False,  # 不要重试！
        "retriable": False,
        "reason": "退款金额 $750 超过自动处理上限 $500",
        "userMessage": "此退款金额需要人工审批",
        "suggestedAction": "escalate_to_human"
    }
```

**区分访问失败 vs 有效空结果**：

```python
# 关键区别：两种"没有结果"的不同含义

# 有效空结果（成功查询，无匹配）
valid_empty = {
    "isError": False,
    "results": [],
    "message": "该时间段内没有符合条件的订单"  # 这是正确答案，不是错误
}

# 访问失败（系统错误，需要重试决策）
access_failure = {
    "isError": True,
    "errorCategory": "transient",
    "isRetryable": True,
    "message": "数据库连接超时"  # 这需要重试
}
```

---

### 📋 例题

**例题 ⭐⭐⭐**

你的 MCP 工具在退款金额超过 $500 时返回 `{"error": "Operation failed"}`。Agent 看到此错误后不断重试。根本原因和解决方案？

**A)** 在 system prompt 中告诉 Agent 不要对退款错误重试

**B)** 返回结构化错误，包含 `errorCategory: "business"`、`isRetryable: false` 和说明超出上限的用户友好消息

**C)** 实现指数退避的自动重试逻辑

**D)** 将退款工具拆分为小额和大额两个版本

> **答案：B** — 统一错误响应阻止 Agent 区分"应该重试"和"不应该重试"的情况。结构化错误让 Agent 能做出正确的恢复决策。

---

## 3.3 跨 Agent 分配工具与 tool_choice
> ⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 工具过多 | Tool Overload（18个 vs 4-5个）|
| 工具专一性 | Tool Specialization |
| 作用域工具访问 | Scoped Tool Access |
| 自动选择 | `tool_choice: "auto"` |
| 强制调用工具 | `tool_choice: "any"` |
| 强制特定工具 | `tool_choice: {"type": "tool", "name": "..."}` |

### 核心知识点

**工具分配原则**：

```python
# ❌ 错误：所有工具给所有 Agent（18个工具）
all_tools = [web_search, fetch_url, read_doc, extract_data, 
             summarize, verify_fact, write_report, ...18 个]
# 后果：合成 Agent 可能误用 web_search，搜索 Agent 可能误用 write_report

# ✅ 正确：每个 Agent 只获得其角色所需的工具
search_agent_tools   = [web_search, fetch_url]           # 2个
analysis_agent_tools = [read_document, extract_data]     # 2个
synthesis_agent_tools = [verify_fact, write_report]      # 2个
# 加跨角色工具（高频需求）：verify_fact 给合成 Agent


# tool_choice 三种模式
import anthropic
client = anthropic.Anthropic()

# 模式1：auto（默认）- 模型决定是否调用工具
response = client.messages.create(
    model="claude-opus-4-5",
    tools=tools,
    tool_choice={"type": "auto"},  # 可能返回纯文本
    messages=[...]
)

# 模式2：any - 必须调用工具，但自选哪个
response = client.messages.create(
    model="claude-opus-4-5",
    tools=tools,
    tool_choice={"type": "any"},  # 保证调用工具，防止返回对话文本
    messages=[...]
)

# 模式3：强制特定工具 - 必须调用指定工具
response = client.messages.create(
    model="claude-opus-4-5",
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_metadata"},  # 必须调用此工具
    messages=[...]
)
```

**使用场景总结**：

| `tool_choice` | 使用场景 |
|--------------|---------|
| `"auto"` | 常规对话，模型自行判断 |
| `"any"` | 需要结构化输出时，防止模型返回纯文本 |
| `{"type":"tool","name":"X"}` | 工作流第一步必须运行特定工具 |

---

### 📋 例题

**例题 ⭐⭐**

你需要确保数据提取流程总是先调用 `extract_metadata` 工具（作为后续丰富步骤的输入），然后再进行其他操作。如何强制执行？

**A)** 在 system prompt 中说明 `extract_metadata` 必须首先被调用

**B)** 设置 `tool_choice: {"type": "tool", "name": "extract_metadata"}`，然后在后续 turn 中处理下一步

**C)** 将 `extract_metadata` 设为唯一可用工具

**D)** 给工具添加 priority 字段

> **答案：B** — 强制工具选择确保特定工具在第一步被调用，后续步骤在下一个 turn 中处理。

---

## 3.4 MCP 服务器集成：Claude Code 与 Agent 工作流
> ⭐⭐⭐

### 📌 核心概念

| 中文 | 英文关键词 |
|------|-----------|
| 项目级配置 | Project-level (`.mcp.json`) |
| 用户级配置 | User-level (`~/.claude.json`) |
| 环境变量扩展 | Environment Variable Expansion |
| MCP 资源 | MCP Resources |
| 内容目录 | Content Catalog |

### 核心知识点

**配置文件层级**：

```json
// .mcp.json（项目级，提交到版本控制，团队共享）
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"  // 环境变量扩展，不提交密钥！
      }
    },
    "jira": {
      "command": "npx",
      "args": ["@company/mcp-jira"],
      "env": {
        "JIRA_API_KEY": "${JIRA_API_KEY}"
      }
    }
  }
}
```

```json
// ~/.claude.json（用户级，个人/实验性，不进入版本控制）
{
  "mcpServers": {
    "my-experimental-tool": {
      "command": "node",
      "args": ["/home/arthur/dev/my-mcp-server/index.js"]
    }
  }
}
```

**MCP Resources vs Tools**：

```
MCP Tools（工具）：执行操作，有副作用
  → create_issue(), send_message(), query_database()

MCP Resources（资源）：暴露内容目录，只读
  → 列出所有可用 issue 的摘要目录
  → 数据库 schema 文档
  → API 文档层次结构

使用 Resources 的好处：
  Agent 可以"浏览"可用内容，减少探索性工具调用
  例：不需要调用 list_issues() 20 次，直接读取 issues 资源目录
```

---

### 📋 例题

**例题 ⭐⭐⭐**

新加入的团队成员反映在他们的机器上 Claude Code 没有应用项目的编码规范。他们在 `~/.claude/CLAUDE.md` 中找到了这些规范。问题在哪里？

**A)** 用户级配置（`~/.claude/`）不通过版本控制共享，新成员不会自动获得

**B)** CLAUDE.md 不支持编码规范，应该用 `.eslintrc`

**C)** 需要在 `.mcp.json` 中注册 CLAUDE.md

**D)** 用户级配置优先级高于项目级，会覆盖项目配置

> **答案：A** — 核心原则：用户级配置（`~/.claude/`）只对该用户生效，不通过版本控制共享。团队共享的配置必须放在项目级（`.claude/CLAUDE.md` 或根目录 `CLAUDE.md`）。

---

## 3.5 内置工具选择：Read/Write/Edit/Bash/Grep/Glob
> ⭐⭐

### 📌 工具选择矩阵

| 工具 | 用途 | 输入 | 何时选择 |
|------|------|------|---------|
| `Grep` | 内容搜索 | 模式 + 路径 | 找函数调用者、错误消息、import |
| `Glob` | 路径模式匹配 | glob pattern | 按名称/扩展名找文件 |
| `Read` | 读取文件内容 | 文件路径 | 需要看完整文件内容 |
| `Write` | 写入完整文件 | 路径 + 内容 | 创建新文件或完整替换 |
| `Edit` | 精确替换文本 | 唯一锚点文本 | 修改文件中特定部分 |
| `Bash` | 执行命令 | shell 命令 | 运行脚本、测试 |

### 核心规则

```python
# Edit 失败时的降级策略
def modify_file(path: str, old_text: str, new_text: str):
    try:
        # 首选：Edit（精确，只修改目标部分）
        edit_file(path, old_text, new_text)
    except NonUniqueMatchError:
        # 降级：Read + Write（当 old_text 在文件中出现多次时）
        content = read_file(path)
        new_content = content.replace(old_text, new_text, 1)  # 只替换第一次
        write_file(path, new_content)


# 增量构建代码库理解（不要一次读取所有文件）
def understand_codebase(entry_point: str):
    # Step 1：Grep 找入口点
    grep_results = grep(pattern="def main", path=".")
    
    # Step 2：Read 跟踪 import
    entry_file = read(entry_point)
    imports = extract_imports(entry_file)
    
    # Step 3：按需 Read 相关文件（不是所有文件）
    for import_path in imports:
        read(import_path)  # 只读需要的
```

---

### 📋 例题

**例题 ⭐⭐**

你需要在一个大型代码库中找到所有调用 `process_payment()` 函数的地方。最合适的工具？

**A)** `Read` 逐一读取所有文件并手动搜索

**B)** `Glob` 找到所有 `.py` 文件，然后 `Read` 每个文件

**C)** `Grep` 在代码库中搜索 `process_payment` 模式

**D)** `Bash` 运行 `find . -name "*.py"`

> **答案：C** — Grep 专门用于内容搜索（在文件内容中查找模式），这正是跨代码库查找函数调用者的最优工具。

---

*文档版本：v1.0 | Domain 2 Task Statements 2.2–2.5*
