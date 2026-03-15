# 提示链（Prompt Chaining）完整指南

> 把复杂任务拆成多个步骤，每步输出作为下步输入，像流水线一样串联。
>
> 本文包含原理 + 6种模式 + 完整可运行代码，供 Claude Code 生成示例。

---

## 目录

1. [核心概念](#1-核心概念)
2. [基础提示链](#2-基础提示链)
3. [六种提示链模式](#3-六种提示链模式)
4. [状态管理与上下文传递](#4-状态管理与上下文传递)
5. [错误处理与重试](#5-错误处理与重试)
6. [与 RAG 结合](#6-与-rag-结合)
7. [实战项目](#7-实战项目)

---

## 1. 核心概念

### 1.1 什么是提示链

```
单步 Prompt：
  输入 ──→ [LLM] ──→ 输出

提示链（Sequential Chain）：
  输入 ──→ [LLM Step 1] ──→ 中间结果1
                                  ↓
           [LLM Step 2] ──→ 中间结果2
                                  ↓
           [LLM Step 3] ──→ 最终输出
```

### 1.2 提示链 vs 单一 Prompt

| 维度 | 单一 Prompt | 提示链 |
|------|------------|--------|
| 任务复杂度 | 简单任务 | 多步骤复杂任务 |
| 可控性 | 低（黑盒） | 高（每步可检查） |
| 调试难度 | 难（出错不知哪步） | 易（逐步定位） |
| 中间结果 | 无 | 每步都可保存和审核 |
| Token 效率 | 一次性消耗 | 按需消耗 |
| 适合场景 | 问答、翻译、摘要 | 分析、生成、多阶段处理 |

### 1.3 核心原则

```
原则1：单一职责（Single Responsibility）
  每个 prompt 只做一件事，不要贪心

原则2：输出即输入（Output as Input）
  每步的输出格式要为下步设计

原则3：结构化中间结果
  用 JSON 传递中间数据，避免解析歧义

原则4：失败隔离（Fail Isolation）
  某步失败时能定位到具体步骤，不影响其他步骤
```

---

## 2. 基础提示链

### 2.1 最简单的提示链（3步）

```python
# basic_chain.py
# Claude Code 提示词：
# 创建一个3步提示链：文档摘要 → 关键点提取 → 行动建议
# 使用 Ollama qwen3:8b，每步打印中间结果

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=4096)
parser = StrOutputParser()

# 原始文档（输入）
document = """
FTH 系统在 2024 年 Q3 共处理支付交易 120 万笔，其中失败交易 8,400 笔（0.7%）。
失败原因分布：AM04（余额不足）占 45%，AC01（账号错误）占 30%，
超时错误占 15%，其他占 10%。
超时问题在高峰期（9:00-11:00）尤为突出，平均响应时间达到 28 秒，
接近 30 秒阈值。运维团队已识别 Finacle API 连接池配置为主要瓶颈。
"""

# ===== Step 1：提取事实 =====
step1_prompt = ChatPromptTemplate.from_template("""
从以下文档中提取所有数字事实和关键数据点。
只输出事实列表，每行一条，格式：「事实内容」。

文档：
{document}
""")

step1_chain = step1_prompt | llm | parser
facts = step1_chain.invoke({"document": document})

print("=" * 50)
print("Step 1 输出（关键事实）：")
print(facts)

# ===== Step 2：问题分析 =====
step2_prompt = ChatPromptTemplate.from_template("""
基于以下事实列表，分析存在的主要问题和风险。
输出格式：
问题1：[问题描述] | 严重程度：[高/中/低] | 影响范围：[描述]
问题2：...

事实列表：
{facts}
""")

step2_chain = step2_prompt | llm | parser
problems = step2_chain.invoke({"facts": facts})

print("\n" + "=" * 50)
print("Step 2 输出（问题分析）：")
print(problems)

# ===== Step 3：生成行动建议 =====
step3_prompt = ChatPromptTemplate.from_template("""
基于以下问题分析，生成具体可执行的改进建议。
每条建议包含：优先级、负责团队、预期效果。

问题分析：
{problems}
""")

step3_chain = step3_prompt | llm | parser
actions = step3_chain.invoke({"problems": problems})

print("\n" + "=" * 50)
print("Step 3 输出（行动建议）：")
print(actions)
```

### 2.2 用 LangChain LCEL 串联（更优雅）

```python
# lcel_chain.py
# Claude Code 提示词：
# 用 LangChain LCEL（|运算符）把3个步骤串成一条链
# 展示 RunnablePassthrough 如何在链中传递原始输入

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=4096)
parser = StrOutputParser()

# 定义各步骤 Prompt
summarize_prompt = ChatPromptTemplate.from_template(
    "用3句话摘要以下文档：\n\n{document}"
)
keywords_prompt = ChatPromptTemplate.from_template(
    "从以下摘要中提取5个关键词（逗号分隔）：\n\n{summary}"
)
title_prompt = ChatPromptTemplate.from_template(
    "根据以下关键词，生成一个吸引人的标题（不超过15字）：\n\n{keywords}"
)

# 方式1：手动串联（显式传递）
def run_manual_chain(document: str) -> dict:
    summary  = (summarize_prompt | llm | parser).invoke({"document": document})
    keywords = (keywords_prompt  | llm | parser).invoke({"summary": summary})
    title    = (title_prompt     | llm | parser).invoke({"keywords": keywords})
    return {"summary": summary, "keywords": keywords, "title": title}


# 方式2：LCEL 串联（保留所有中间结果）
full_chain = (
    RunnablePassthrough.assign(
        summary=summarize_prompt | llm | parser
    )
    | RunnablePassthrough.assign(
        keywords=lambda x: (keywords_prompt | llm | parser).invoke(
            {"summary": x["summary"]}
        )
    )
    | RunnablePassthrough.assign(
        title=lambda x: (title_prompt | llm | parser).invoke(
            {"keywords": x["keywords"]}
        )
    )
)

# 运行
document = "你的文档内容..."
result = full_chain.invoke({"document": document})
print(f"标题：{result['title']}")
print(f"关键词：{result['keywords']}")
print(f"摘要：{result['summary']}")
```

---

## 3. 六种提示链模式

### 模式一：顺序链（Sequential Chain）

最基础，步骤A → 步骤B → 步骤C，线性执行。

```python
# pattern_sequential.py
# Claude Code 提示词：
# 实现一个文档质检顺序链：
# 步骤1 语法检查 → 步骤2 事实核查 → 步骤3 风格优化 → 步骤4 最终输出
# 每步输出 JSON，包含 issues（问题列表）和 revised_text（修订文本）

import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=4096)

STEPS = [
    {
        "name": "语法检查",
        "prompt": """检查以下文本的语法错误并修正。
输出纯 JSON（不要 markdown）：
{{"issues": ["问题1", "问题2"], "revised_text": "修正后的文本"}}

文本：{text}"""
    },
    {
        "name": "逻辑优化",
        "prompt": """优化以下文本的逻辑结构，使表达更清晰。
输出纯 JSON：
{{"issues": ["逻辑问题1"], "revised_text": "优化后的文本"}}

文本：{text}"""
    },
    {
        "name": "风格润色",
        "prompt": """润色以下文本的表达风格，使其更专业。
输出纯 JSON：
{{"issues": ["风格建议1"], "revised_text": "润色后的文本"}}

文本：{text}"""
    },
]

def run_sequential_qa(original_text: str) -> dict:
    current_text = original_text
    all_issues = []
    history = []

    for step in STEPS:
        prompt = step["prompt"].format(text=current_text)
        response = llm.invoke(prompt).content

        try:
            # 提取 JSON（去掉可能的 markdown 包裹）
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            result = json.loads(clean.strip())
        except json.JSONDecodeError:
            result = {"issues": [], "revised_text": current_text}

        history.append({
            "step": step["name"],
            "issues_found": result.get("issues", []),
            "text_before": current_text[:100] + "...",
        })
        all_issues.extend(result.get("issues", []))
        current_text = result.get("revised_text", current_text)
        print(f"  ✅ {step['name']} 完成，发现 {len(result.get('issues',[]))} 个问题")

    return {
        "original": original_text,
        "final":    current_text,
        "all_issues": all_issues,
        "history":  history,
    }

result = run_sequential_qa("这个系统在处理交易时候会出现错误，因为配置有问题所以性能不好。")
print(f"\n原文：{result['original']}")
print(f"优化后：{result['final']}")
print(f"共发现 {len(result['all_issues'])} 个问题")
```

---

### 模式二：分支链（Branching Chain）

根据中间结果选择不同的处理路径。

```python
# pattern_branching.py
# Claude Code 提示词：
# 实现一个客服工单分类分支链：
# Step 1 分类（技术问题/账单问题/其他）
# Step 2 根据分类走不同处理路径
# Step 3 汇总生成统一格式的回复

import json
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=4096)

def classify_ticket(ticket: str) -> str:
    """Step 1：分类"""
    prompt = f"""将以下客服工单分类，只输出一个词：TECHNICAL / BILLING / OTHER

工单内容：{ticket}

分类："""
    return llm.invoke(prompt).content.strip().upper()


def handle_technical(ticket: str) -> str:
    """技术问题处理路径"""
    prompt = f"""你是技术支持专家。
分析以下技术问题，提供故障排查步骤（3步以内）。

问题：{ticket}

排查步骤："""
    return llm.invoke(prompt).content


def handle_billing(ticket: str) -> str:
    """账单问题处理路径"""
    prompt = f"""你是账单支持专员。
处理以下账单问题，说明需要核实的信息和处理流程。

问题：{ticket}

处理方案："""
    return llm.invoke(prompt).content


def handle_other(ticket: str) -> str:
    """其他问题处理路径"""
    prompt = f"""你是客服助理。
礼貌地回复以下问题，如无法解决则说明转接流程。

问题：{ticket}

回复："""
    return llm.invoke(prompt).content


def format_response(ticket: str, category: str, handling: str) -> str:
    """Step 3：生成统一格式回复"""
    prompt = f"""将以下处理方案格式化为标准客服回复邮件。

原始工单：{ticket}
问题类型：{category}
处理方案：{handling}

请生成一封专业的回复邮件（包含称呼、正文、结语）："""
    return llm.invoke(prompt).content


def process_ticket(ticket: str) -> dict:
    """完整分支链"""
    print(f"📩 工单：{ticket[:50]}...")

    # Step 1：分类
    category = classify_ticket(ticket)
    print(f"  🏷️  分类结果：{category}")

    # Step 2：分支处理
    handlers = {
        "TECHNICAL": handle_technical,
        "BILLING":   handle_billing,
        "OTHER":     handle_other,
    }
    handler = handlers.get(category, handle_other)
    handling = handler(ticket)

    # Step 3：统一格式化
    response = format_response(ticket, category, handling)

    return {"category": category, "handling": handling, "final_response": response}


# 测试
tickets = [
    "我的账户登录时一直报错 500，已经试了3次了",
    "上个月的账单多收了我 50 元，请帮我核查",
    "你们的服务什么时候支持英文界面？",
]
for t in tickets:
    result = process_ticket(t)
    print(f"\n最终回复（前100字）：{result['final_response'][:100]}...")
    print("-" * 60)
```

---

### 模式三：并行链（Parallel Chain）

多个步骤同时执行，最后汇总结果，节省时间。

```python
# pattern_parallel.py
# Claude Code 提示词：
# 实现并行分析链：对同一篇文章同时进行
# （情感分析 / 关键实体提取 / 可读性评分）
# 三个任务并行执行，最后合并结果
# 使用 asyncio 实现真正的并行

import asyncio
import json
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# 异步版本
from langchain_ollama import OllamaLLM

async def analyze_sentiment(text: str, llm) -> dict:
    """情感分析"""
    prompt = f"""分析以下文本的情感倾向。
输出纯 JSON：{{"sentiment": "POSITIVE/NEGATIVE/NEUTRAL", "confidence": 0.0-1.0, "reason": "理由"}}

文本：{text}"""
    response = await llm.ainvoke(prompt)
    try:
        clean = response.content if hasattr(response, 'content') else str(response)
        start, end = clean.find('{'), clean.rfind('}') + 1
        return json.loads(clean[start:end])
    except:
        return {"sentiment": "UNKNOWN", "confidence": 0.0, "reason": "解析失败"}


async def extract_entities(text: str, llm) -> dict:
    """实体提取"""
    prompt = f"""从以下文本提取命名实体。
输出纯 JSON：{{"persons": [], "organizations": [], "locations": [], "dates": []}}

文本：{text}"""
    response = await llm.ainvoke(prompt)
    try:
        clean = response.content if hasattr(response, 'content') else str(response)
        start, end = clean.find('{'), clean.rfind('}') + 1
        return json.loads(clean[start:end])
    except:
        return {"persons": [], "organizations": [], "locations": [], "dates": []}


async def score_readability(text: str, llm) -> dict:
    """可读性评分"""
    prompt = f"""评估以下文本的可读性。
输出纯 JSON：{{"score": 1-10, "level": "简单/中等/复杂", "suggestions": ["建议1"]}}

文本：{text}"""
    response = await llm.ainvoke(prompt)
    try:
        clean = response.content if hasattr(response, 'content') else str(response)
        start, end = clean.find('{'), clean.rfind('}') + 1
        return json.loads(clean[start:end])
    except:
        return {"score": 5, "level": "中等", "suggestions": []}


async def parallel_analyze(text: str) -> dict:
    """并行执行三个分析任务"""
    llm = ChatOllama(model="qwen3:8b", temperature=0.0, num_ctx=4096)

    import time
    start = time.time()

    # 三个任务同时执行
    sentiment, entities, readability = await asyncio.gather(
        analyze_sentiment(text, llm),
        extract_entities(text, llm),
        score_readability(text, llm),
    )

    elapsed = time.time() - start
    print(f"  ⏱  并行执行耗时：{elapsed:.1f}s（串行约需 {elapsed*2.5:.1f}s）")

    # 合并所有结果
    return {
        "text_preview": text[:80] + "...",
        "sentiment":    sentiment,
        "entities":     entities,
        "readability":  readability,
        "analysis_time": f"{elapsed:.1f}s",
    }


# 运行
text = """
阿里巴巴集团于2024年3月在杭州举行年度技术峰会，
CEO 张勇宣布将在未来两年内投入100亿元用于 AI 基础设施建设。
此次会议吸引了来自北京、上海、深圳的3000余名开发者参与。
"""

result = asyncio.run(parallel_analyze(text))
print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

### 模式四：迭代精炼链（Iterative Refinement Chain）

对同一个输出反复优化，直到满足质量标准。

```python
# pattern_iterative.py
# Claude Code 提示词：
# 实现迭代精炼链：
# 生成初稿 → 自我评估（打分）→ 如果分数 < 阈值则继续优化
# 最多迭代5次，用 Ollama qwen3:8b
# 每次迭代打印分数和改进点

import json
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b", temperature=0.3, num_ctx=4096)

def generate_draft(task: str) -> str:
    """生成初稿"""
    return llm.invoke(f"请完成以下写作任务：\n\n{task}").content


def evaluate_and_score(text: str, criteria: str) -> dict:
    """自我评估"""
    prompt = f"""请评估以下文本的质量。

评估标准：{criteria}

文本：
{text}

输出纯 JSON（不要 markdown）：
{{
  "score": 1到10的整数,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["缺点1", "缺点2"],
  "specific_improvements": ["具体改进点1", "具体改进点2"]
}}"""
    
    response = llm.invoke(prompt).content
    try:
        start, end = response.find('{'), response.rfind('}') + 1
        return json.loads(response[start:end])
    except:
        return {"score": 5, "strengths": [], "weaknesses": [], "specific_improvements": []}


def refine(text: str, improvements: list) -> str:
    """根据改进点优化文本"""
    improvements_text = "\n".join(f"- {i}" for i in improvements)
    prompt = f"""请根据以下改进建议，重写优化这段文本。
保留原有优点，重点解决指出的问题。

改进建议：
{improvements_text}

原始文本：
{text}

优化后的文本："""
    return llm.invoke(prompt).content


def iterative_refine(
    task: str,
    criteria: str,
    target_score: int = 8,
    max_iterations: int = 5
) -> dict:
    """迭代精炼主函数"""
    print(f"📝 任务：{task[:60]}...")
    print(f"🎯 目标分数：{target_score}/10，最多迭代 {max_iterations} 次\n")

    current_text = generate_draft(task)
    history = []

    for i in range(max_iterations):
        # 评估当前版本
        evaluation = evaluate_and_score(current_text, criteria)
        score = evaluation.get("score", 0)

        history.append({
            "iteration": i,
            "score": score,
            "weaknesses": evaluation.get("weaknesses", []),
            "text_preview": current_text[:80] + "...",
        })

        print(f"迭代 {i} | 分数：{score}/10 | 问题：{evaluation.get('weaknesses', [])}")

        if score >= target_score:
            print(f"\n✅ 达到目标分数 {score}/10，停止迭代")
            break

        if i < max_iterations - 1:
            improvements = evaluation.get("specific_improvements", [])
            current_text = refine(current_text, improvements)
    else:
        print(f"\n⚠️  达到最大迭代次数，最终分数：{history[-1]['score']}/10")

    return {
        "final_text":  current_text,
        "final_score": history[-1]["score"],
        "iterations":  len(history),
        "history":     history,
    }


# 测试
result = iterative_refine(
    task="写一段介绍 ISO 20022 支付标准的技术文档段落（200字）",
    criteria="技术准确性、逻辑清晰、适合开发者阅读、包含具体例子",
    target_score=8,
    max_iterations=4,
)
print(f"\n📄 最终文本：\n{result['final_text']}")
print(f"\n📊 经过 {result['iterations']} 次迭代，最终得分：{result['final_score']}/10")
```

---

### 模式五：路由链（Router Chain）

根据输入内容，动态选择最合适的子链处理。

```python
# pattern_router.py
# Claude Code 提示词：
# 实现一个智能路由链：
# 路由器判断问题类型（代码/数学/写作/问答）
# 不同类型用不同的 system prompt 和参数处理
# 展示路由决策过程

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 不同任务用不同温度
code_llm    = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=8192)
math_llm    = ChatOllama(model="qwen3:8b", temperature=0.0, num_ctx=4096)
writing_llm = ChatOllama(model="qwen3:8b", temperature=0.8, num_ctx=4096)
qa_llm      = ChatOllama(model="qwen3:8b", temperature=0.2, num_ctx=4096)

# 各类型专属 Prompt
ROUTES = {
    "CODE": {
        "llm": code_llm,
        "prompt": ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的软件工程师。提供清晰、可运行的代码，附上注释解释关键逻辑。"),
            ("human",  "{question}")
        ]),
    },
    "MATH": {
        "llm": math_llm,
        "prompt": ChatPromptTemplate.from_messages([
            ("system", "你是一个数学专家。逐步展示计算过程，确保每步都有说明。"),
            ("human",  "{question}")
        ]),
    },
    "WRITING": {
        "llm": writing_llm,
        "prompt": ChatPromptTemplate.from_messages([
            ("system", "你是一个专业写作专家。注重文字的表达力和结构。"),
            ("human",  "{question}")
        ]),
    },
    "QA": {
        "llm": qa_llm,
        "prompt": ChatPromptTemplate.from_messages([
            ("system", "你是一个知识渊博的助手。给出简洁准确的回答。"),
            ("human",  "{question}")
        ]),
    },
}

def route(question: str) -> str:
    """路由器：判断问题类型"""
    prompt = f"""判断以下问题属于哪个类别，只输出一个词：
CODE（编程/代码相关）
MATH（数学/计算相关）  
WRITING（写作/文章相关）
QA（其他知识问答）

问题：{question}

类别："""
    result = qa_llm.invoke(prompt).content.strip().upper()
    # 提取有效类别（防止模型输出多余内容）
    for category in ["CODE", "MATH", "WRITING", "QA"]:
        if category in result:
            return category
    return "QA"  # 默认


def smart_router_chain(question: str) -> dict:
    """智能路由链"""
    # Step 1：路由决策
    category = route(question)
    print(f"  🔀 路由决策：{category}")

    # Step 2：选择对应的链并执行
    route_config = ROUTES[category]
    chain = route_config["prompt"] | route_config["llm"] | StrOutputParser()
    answer = chain.invoke({"question": question})

    return {"question": question, "category": category, "answer": answer}


# 测试
questions = [
    "用 Python 实现一个二分查找算法",
    "计算 (3 + 5) × 2 的结果，并解释运算顺序",
    "写一段介绍秋天的散文",
    "Python 和 JavaScript 哪个更适合初学者？",
]
for q in questions:
    result = smart_router_chain(q)
    print(f"\n问题：{q}")
    print(f"类型：{result['category']}")
    print(f"回答（前100字）：{result['answer'][:100]}...")
    print("-" * 60)
```

---

### 模式六：反思链（Reflection Chain）

让 LLM 生成答案后，再用另一个 LLM（或同一个）审查和批评。

```python
# pattern_reflection.py
# Claude Code 提示词：
# 实现反思链：
# Actor LLM 生成答案
# Critic LLM 找出错误和不足
# Actor LLM 根据批评修正
# 可配置反思轮数（默认2轮）

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# Actor 和 Critic 可以是不同模型，这里都用 qwen3:8b
actor_llm  = ChatOllama(model="qwen3:8b", temperature=0.3, num_ctx=4096)
critic_llm = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=4096)  # Critic 低温，更严格

def actor_generate(task: str, previous_critique: str = "") -> str:
    """Actor：生成或修正答案"""
    if previous_critique:
        prompt = f"""任务：{task}

批评意见：
{previous_critique}

请根据上述批评意见，修正并改进你的答案："""
    else:
        prompt = f"请完成以下任务：\n\n{task}"
    
    return actor_llm.invoke(prompt).content


def critic_review(task: str, answer: str) -> dict:
    """Critic：批评和评估"""
    prompt = f"""你是一个严格的质量审查员。
请批评性地审查以下任务的答案。

原始任务：{task}
待审查答案：{answer}

请指出：
1. 事实性错误（如有）
2. 逻辑漏洞（如有）
3. 遗漏的重要内容
4. 表达不清晰的地方
5. 综合评分（1-10）

格式：
错误：[列出错误，没有则写"无"]
遗漏：[列出遗漏，没有则写"无"]
改进：[具体改进建议]
评分：[数字]/10"""
    
    review = critic_llm.invoke(prompt).content
    
    # 提取评分
    score = 5  # 默认分
    for line in review.split('\n'):
        if '评分' in line and '/' in line:
            try:
                score = int(line.split('：')[-1].split('/')[0].strip())
            except:
                pass
    
    return {"review": review, "score": score}


def reflection_chain(
    task: str,
    max_rounds: int = 2,
    stop_score: int = 8
) -> dict:
    """反思链主函数"""
    print(f"🎭 任务：{task[:60]}...")
    print(f"📋 最多反思 {max_rounds} 轮，目标分数 {stop_score}/10\n")

    history = []
    critique = ""

    for round_num in range(max_rounds + 1):  # +1 因为第0轮是初稿
        # Actor 生成
        label = "初稿" if round_num == 0 else f"第{round_num}次修正"
        answer = actor_generate(task, critique)
        print(f"📝 {label}（前80字）：{answer[:80]}...")

        # Critic 评审
        review_result = critic_review(task, answer)
        score = review_result["score"]
        print(f"🔍 Critic 评分：{score}/10")

        history.append({
            "round":  round_num,
            "label":  label,
            "answer": answer,
            "score":  score,
            "review": review_result["review"],
        })

        if score >= stop_score:
            print(f"\n✅ 达到目标分数 {score}/10，停止反思")
            break

        if round_num < max_rounds:
            critique = review_result["review"]
            print(f"💬 继续反思...\n")
        else:
            print(f"\n⚠️  达到最大轮数，最终分数：{score}/10")

    return {
        "task":        task,
        "final_answer": history[-1]["answer"],
        "final_score":  history[-1]["score"],
        "rounds":       len(history),
        "history":      history,
    }


# 测试
result = reflection_chain(
    task="解释什么是 CAP 定理，并举一个分布式系统的实际例子",
    max_rounds=2,
    stop_score=8,
)
print(f"\n最终答案：\n{result['final_answer']}")
```

---

## 4. 状态管理与上下文传递

### 4.1 用字典传递状态（最推荐）

```python
# state_management.py
# Claude Code 提示词：
# 实现带完整状态追踪的提示链
# state 字典贯穿所有步骤，每步追加自己的结果
# 最后输出完整的执行报告

from langchain_ollama import ChatOllama
from datetime import datetime
import time

llm = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=4096)

def create_state(input_data: dict) -> dict:
    """初始化状态对象"""
    return {
        "input": input_data,
        "steps": [],          # 记录每步执行情况
        "outputs": {},        # 每步的输出结果
        "errors": [],         # 错误记录
        "start_time": time.time(),
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "model": "qwen3:8b",
        }
    }

def run_step(state: dict, step_name: str, prompt_fn, input_key: str, output_key: str) -> dict:
    """
    执行单个步骤并更新状态
    
    Args:
        state:      当前状态字典
        step_name:  步骤名称
        prompt_fn:  接受 state 返回 prompt 字符串的函数
        input_key:  从 state["outputs"] 中读取哪个键作为输入
        output_key: 把输出写入 state["outputs"] 的哪个键
    """
    step_start = time.time()
    
    try:
        prompt = prompt_fn(state)
        output = llm.invoke(prompt).content.strip()
        
        state["outputs"][output_key] = output
        state["steps"].append({
            "name":     step_name,
            "status":   "success",
            "duration": round(time.time() - step_start, 2),
            "input_key":  input_key,
            "output_key": output_key,
        })
        print(f"  ✅ {step_name} ({state['steps'][-1]['duration']}s)")
        
    except Exception as e:
        state["errors"].append({"step": step_name, "error": str(e)})
        state["steps"].append({
            "name":   step_name,
            "status": "error",
            "error":  str(e),
        })
        print(f"  ❌ {step_name} 失败：{e}")
    
    return state


def generate_report(state: dict) -> str:
    """从状态生成执行报告"""
    total_time = round(time.time() - state["start_time"], 2)
    success_steps = [s for s in state["steps"] if s["status"] == "success"]
    
    report = f"""
=== 执行报告 ===
总耗时：{total_time}s
步骤：{len(success_steps)}/{len(state['steps'])} 成功
错误：{len(state['errors'])} 个

步骤详情：
"""
    for step in state["steps"]:
        icon = "✅" if step["status"] == "success" else "❌"
        duration = step.get("duration", "N/A")
        report += f"  {icon} {step['name']} ({duration}s)\n"
    
    return report


# 使用示例
document = "FTH 系统 Q3 报告：交易量120万，失败率0.7%，超时问题在高峰期明显..."

state = create_state({"document": document})

state = run_step(
    state, "文档摘要", 
    lambda s: f"用2句话摘要：{s['input']['document']}",
    "document", "summary"
)
state = run_step(
    state, "关键词提取",
    lambda s: f"提取5个关键词（逗号分隔）：{s['outputs']['summary']}",
    "summary", "keywords"
)
state = run_step(
    state, "问题识别",
    lambda s: f"从摘要中识别主要问题（列表）：{s['outputs']['summary']}",
    "summary", "problems"
)

print(generate_report(state))
print("\n最终输出：")
for key, value in state["outputs"].items():
    print(f"  {key}: {value[:80]}...")
```

---

## 5. 错误处理与重试

```python
# error_handling.py
# Claude Code 提示词：
# 为提示链添加完整的错误处理：
# 1. JSON 解析失败时自动重试（最多3次）
# 2. LLM 超时时退回到简化版 prompt
# 3. 关键步骤失败时有 fallback 输出

import json
import time
from functools import wraps
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=4096)

def retry_on_failure(max_retries=3, delay=1.0, fallback=None):
    """装饰器：失败时自动重试，超过次数返回 fallback"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {func.__name__} 第{attempt+1}次失败，{delay}s后重试...")
                        time.sleep(delay)
            
            print(f"  ❌ {func.__name__} 失败{max_retries}次，使用 fallback")
            if fallback is not None:
                return fallback
            raise last_error
        return wrapper
    return decorator


def parse_json_robust(text: str, expected_keys: list = None) -> dict:
    """
    健壮的 JSON 解析：
    1. 先尝试直接解析
    2. 尝试提取 {} 之间的内容
    3. 尝试修复常见格式错误
    """
    # 方法1：直接解析
    try:
        return json.loads(text)
    except:
        pass
    
    # 方法2：提取 JSON 块
    try:
        start = text.find('{')
        end   = text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    
    # 方法3：清理后解析
    try:
        clean = text.replace("'", '"')          # 单引号改双引号
        clean = clean.replace("True", "true")   # Python bool → JSON bool
        clean = clean.replace("False", "false")
        start = clean.find('{')
        end   = clean.rfind('}') + 1
        return json.loads(clean[start:end])
    except:
        pass
    
    # 全部失败：返回空结构
    if expected_keys:
        return {k: None for k in expected_keys}
    return {}


@retry_on_failure(max_retries=3, fallback={"category": "UNKNOWN", "confidence": 0.0})
def classify_with_retry(text: str) -> dict:
    """带重试的分类步骤"""
    prompt = f"""对以下文本分类，输出纯 JSON（不要 markdown）：
{{"category": "POSITIVE/NEGATIVE/NEUTRAL", "confidence": 0.0-1.0}}

文本：{text}"""
    
    response = llm.invoke(prompt).content
    result = parse_json_robust(response, ["category", "confidence"])
    
    # 验证结果
    if not result.get("category"):
        raise ValueError("分类结果为空")
    
    return result


# 使用
texts = [
    "这个产品非常好用，强烈推荐！",
    "服务很差，完全不推荐。",
    "一般般，没什么特别的。",
]
for text in texts:
    result = classify_with_retry(text)
    print(f"  {text[:20]} → {result['category']} ({result['confidence']:.2f})")
```

---

## 6. 与 RAG 结合

```python
# rag_chain.py
# Claude Code 提示词：
# 实现 RAG + 提示链的组合：
# Step 1 查询改写（让检索更准）
# Step 2 RAG 检索
# Step 3 答案生成
# Step 4 答案验证（检查是否有幻觉）
# 使用 ChromaDB + nomic-embed-text + qwen3:8b

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=8192)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

def step1_rewrite_query(question: str) -> list[str]:
    """Step 1：把用户问题改写为更适合检索的查询"""
    prompt = f"""将以下问题改写为2个不同角度的搜索查询（每行一个，直接输出）：
{question}"""
    response = llm.invoke(prompt).content
    queries = [line.strip() for line in response.strip().split('\n') if line.strip()]
    return [question] + queries[:2]  # 原问题 + 2个改写版


def step2_retrieve(queries: list[str], k=3) -> list:
    """Step 2：多查询检索，去重合并"""
    all_docs, seen = [], set()
    for q in queries:
        for doc in vectorstore.similarity_search(q, k=k):
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                all_docs.append(doc)
    return all_docs


def step3_generate_answer(question: str, docs: list) -> str:
    """Step 3：基于检索结果生成答案"""
    context = "\n\n---\n\n".join([
        f"[来源: {d.metadata.get('source','未知')}]\n{d.page_content}"
        for d in docs
    ])
    prompt = f"""严格基于以下文档内容回答问题。
文档中没有的信息请说"文档未提及"，不要编造。

文档：
{context}

问题：{question}

回答："""
    return llm.invoke(prompt).content


def step4_verify_faithfulness(question: str, answer: str, docs: list) -> dict:
    """Step 4：验证答案忠实度（幻觉检测）"""
    context = "\n".join([d.page_content[:200] for d in docs])
    prompt = f"""判断以下答案是否完全基于提供的文档内容，
有无包含文档中不存在的信息（幻觉）。

文档上下文（摘要）：
{context}

问题：{question}
答案：{answer}

输出纯 JSON：
{{"is_faithful": true/false, "hallucinated_parts": ["幻觉内容1"], "confidence": 0.0-1.0}}"""
    
    response = llm.invoke(prompt).content
    try:
        start, end = response.find('{'), response.rfind('}') + 1
        return json.loads(response[start:end])
    except:
        return {"is_faithful": True, "hallucinated_parts": [], "confidence": 0.5}


def rag_chain_with_verification(question: str) -> dict:
    """完整 RAG + 提示链"""
    print(f"\n🔍 问题：{question}")

    queries = step1_rewrite_query(question)
    print(f"  Step 1 改写查询：{len(queries)} 个")

    docs = step2_retrieve(queries)
    print(f"  Step 2 检索文档：{len(docs)} 个")

    answer = step3_generate_answer(question, docs)
    print(f"  Step 3 生成答案：{answer[:60]}...")

    verification = step4_verify_faithfulness(question, answer, docs)
    faithful = verification.get("is_faithful", True)
    print(f"  Step 4 忠实度验证：{'✅ 通过' if faithful else '❌ 发现幻觉'}")

    if not faithful:
        hallucinations = verification.get("hallucinated_parts", [])
        print(f"  ⚠️  幻觉内容：{hallucinations}")

    return {
        "question":     question,
        "answer":       answer,
        "is_faithful":  faithful,
        "sources":      [d.metadata.get("source", "未知") for d in docs],
        "verification": verification,
    }
```

---

## 7. 实战项目

### 7.1 完整项目：文档分析报告生成器

```python
# project_report_generator.py
# Claude Code 提示词：
# 创建一个完整的文档分析报告生成器，整合以下提示链：
# 1. 文档预处理（清理、分段）
# 2. 多维分析（并行：主题/情感/实体/风险）
# 3. 综合评估（汇总分析结果）
# 4. 报告生成（格式化为 Markdown 报告）
# 5. 质量验证（检查报告完整性）
#
# 支持命令行参数：python report_generator.py --file doc.txt --model qwen3:8b
# 输出：report.md 文件

import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path
from langchain_ollama import ChatOllama

def parse_args():
    parser = argparse.ArgumentParser(description="文档分析报告生成器")
    parser.add_argument("--file",  required=True, help="输入文档路径")
    parser.add_argument("--model", default="qwen3:8b", help="LLM 模型名")
    parser.add_argument("--output", default="report.md", help="输出报告路径")
    return parser.parse_args()


async def analyze_topic(text: str, llm) -> str:
    prompt = f"识别以下文档的主要主题和子主题（列表格式）：\n\n{text[:2000]}"
    r = await llm.ainvoke(prompt)
    return r.content


async def analyze_sentiment(text: str, llm) -> str:
    prompt = f"分析以下文档的整体情感倾向和关键情感段落：\n\n{text[:2000]}"
    r = await llm.ainvoke(prompt)
    return r.content


async def extract_entities(text: str, llm) -> str:
    prompt = f"提取以下文档中的关键实体（人名/组织/地点/日期/数字）：\n\n{text[:2000]}"
    r = await llm.ainvoke(prompt)
    return r.content


async def identify_risks(text: str, llm) -> str:
    prompt = f"识别以下文档中提到的风险、问题或挑战：\n\n{text[:2000]}"
    r = await llm.ainvoke(prompt)
    return r.content


async def generate_report(document: str, model: str) -> str:
    llm = ChatOllama(model=model, temperature=0.2, num_ctx=8192)

    print("📊 Step 1：并行多维分析...")
    topic, sentiment, entities, risks = await asyncio.gather(
        analyze_topic(document, llm),
        analyze_sentiment(document, llm),
        extract_entities(document, llm),
        identify_risks(document, llm),
    )

    print("📝 Step 2：综合评估...")
    synthesis_prompt = f"""基于以下分析结果，生成综合评估摘要（200字以内）：

主题：{topic}
情感：{sentiment}
实体：{entities}
风险：{risks}

综合评估："""
    synthesis = llm.invoke(synthesis_prompt).content

    print("📄 Step 3：生成最终报告...")
    report_prompt = f"""将以下分析结果格式化为专业的 Markdown 分析报告。

综合摘要：{synthesis}
主题分析：{topic}
情感分析：{sentiment}
关键实体：{entities}
风险识别：{risks}

请生成包含以下章节的 Markdown 报告：
# 文档分析报告
## 执行摘要
## 主题分析
## 情感分析
## 关键实体
## 风险与挑战
## 结论与建议
"""
    report = llm.invoke(report_prompt).content

    # 添加元数据头部
    header = f"""---
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
使用模型：{model}
文档长度：{len(document)} 字符
---

"""
    return header + report


async def main():
    args = parse_args()

    # 读取文档
    doc_path = Path(args.file)
    if not doc_path.exists():
        print(f"❌ 文件不存在：{args.file}")
        return

    document = doc_path.read_text(encoding="utf-8")
    print(f"📂 已加载文档：{args.file}（{len(document)} 字符）")

    # 生成报告
    report = await generate_report(document, args.model)

    # 保存报告
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"\n✅ 报告已保存：{args.output}")
    print(f"📊 报告长度：{len(report)} 字符")


if __name__ == "__main__":
    asyncio.run(main())
```

### 7.2 Claude Code 综合提示词

```
在 prompt_chaining/ 目录创建完整的提示链示例项目：

1. basic_chain.py          → 3步顺序链（文档分析）
2. pattern_branching.py    → 分支链（客服工单分类路由）
3. pattern_parallel.py     → 并行链（asyncio 同时执行3个分析）
4. pattern_iterative.py    → 迭代精炼链（自动评分，达标停止）
5. pattern_reflection.py   → 反思链（Actor-Critic 互相批评）
6. pattern_router.py       → 路由链（按问题类型选择不同处理）
7. state_management.py     → 状态管理（完整执行追踪）
8. error_handling.py       → 错误处理（重试 + fallback + JSON 解析）
9. rag_chain.py            → RAG + 提示链（检索+生成+验证）
10. report_generator.py    → 综合项目（并行分析 + 报告生成）

技术要求：
  所有示例使用 Ollama qwen3:8b
  每个文件顶部有 Claude Code 提示词说明
  每个示例有测试数据可以直接运行
  关键步骤有 print 输出，方便看执行过程
  JSON 输出都有 fallback 处理
  异步代码使用 asyncio.run() 入口
```

---

## 总结

```
六种模式速查：

顺序链（Sequential）
  用途：线性多步骤处理（清洗→分析→格式化）
  特点：前一步输出是后一步输入

分支链（Branching）
  用途：根据内容走不同路径（分类→专项处理）
  特点：有条件判断，路径动态选择

并行链（Parallel）
  用途：独立任务同时执行（多维分析）
  特点：asyncio.gather，节省时间

迭代精炼链（Iterative）
  用途：需要多次优化达到质量标准
  特点：循环+评分，满足条件停止

反思链（Reflection）
  用途：需要高质量输出，自我批评改进
  特点：Actor 生成，Critic 批评，循环修正

路由链（Router）
  用途：统一入口，按类型分发到专项处理
  特点：先分类再处理，不同类型不同参数

选择原则：
  单方向多步骤    → 顺序链
  需要判断分叉    → 分支链 / 路由链
  步骤互相独立    → 并行链
  需要达到质量标准 → 迭代精炼链
  需要高质量审查  → 反思链
```

---

*文档版本：v1.0 | 2026年3月 | 适用于 Claude Code 代码生成*
