# RAGAS 评估体系深度指南
## Faithfulness · Context Recall · Context Precision · Answer Relevancy · LLM-as-Judge

> 阶段：Week 4-5 | 前置：基础 RAG + 高级 RAG 技术已实现
>
> 核心原则：没有评估，就没有改进。这章把 RAG 从"感觉能用"变成"数据可证"。

---

## 目录

1. [为什么需要自动化评估](#1-为什么需要自动化评估)
2. [RAGAS 核心指标详解](#2-ragas-核心指标详解)
3. [LLM-as-Judge：用 LLM 评估 LLM](#3-llm-as-judge用-llm-评估-llm)
4. [RAGAS 完整实现](#4-ragas-完整实现)
5. [构建评估数据集](#5-构建评估数据集)
6. [评估结果分析与诊断](#6-评估结果分析与诊断)
7. [持续评估体系（CI/CD）](#7-持续评估体系cicd)
8. [完整项目实战](#8-完整项目实战)

---

## 1. 为什么需要自动化评估

### 1.1 手工评估的问题

大多数人评估 RAG 的方式是：跑几个问题，看答案"感觉不错"，就认为系统好了。这有几个根本缺陷：

```
主观性（Subjectivity）：
  "这个答案不错" → 不同人有不同标准
  同一个人今天和明天的判断也可能不同

不可扩展（Not Scalable）：
  10个问题可以手工看
  1000个问题呢？换了 embedding 模型重新跑？
  调整了 chunk_size 要全部重看？

无法量化改进（No Quantified Improvement）：
  "加了 Reranking 感觉好了一些" 
  → 好了多少？Context Precision 从 0.62 → 0.81 这才是有效信息

无法定位问题（Can't Diagnose）：
  "答案质量差" 
  → 是检索问题？还是生成问题？还是 chunk 太小？
  → 没有分项指标就无法定位
```

### 1.2 RAGAS 的位置

RAGAS（Retrieval Augmented Generation Assessment）是目前最主流的 RAG 评估框架，2023年由论文提出，现已成为行业标准：

```
RAG 系统
    │
    ├── 检索阶段 (Retrieval)
    │       ├── Context Precision   ← 检索到的块有多少是有用的？
    │       └── Context Recall      ← 有用的信息有多少被检索到了？
    │
    └── 生成阶段 (Generation)
            ├── Faithfulness        ← 答案有没有脱离上下文？（幻觉检测）
            ├── Answer Relevancy    ← 答案有没有回答问题？
            └── Answer Correctness  ← 答案和标准答案有多接近？（需 ground truth）
```

---

## 2. RAGAS 核心指标详解

### 2.1 Faithfulness（忠实度）

**检测目标：幻觉（Hallucination）**

**定义：** 答案中的每一个陈述（Claim），是否都能在检索到的上下文中找到支撑？

```
分数计算：
  Faithfulness = 有上下文支撑的陈述数 / 答案中的总陈述数

示例：
  上下文：
    "AM04 错误码表示账户余额不足，处理方式是通知客户充值。"
  
  答案（LLM 生成）：
    "AM04 错误码表示余额不足（✅ 有支撑），
     客户应充值后重试（✅ 有支撑），
     也可以尝试分期付款（❌ 无支撑，LLM 自己编的！）。"
  
  Faithfulness = 2 / 3 = 0.67  ← 存在幻觉！
**目标值：> 0.8**。低于 0.6 说明系统幻觉严重，用户不能信任答案。

**为什么这个指标最重要：** 用户对 RAG 系统的核心期望是"基于文档回答"，一旦 LLM 开始编造，整个系统的可信度崩塌。

```
影响 Faithfulness 的因素：
  LLM temperature 过高   → 模型发挥过度
  num_ctx 不足           → 上下文被截断，模型无法参考
  Prompt 约束不够强      → 没有明确告诉模型"不要编造"
  检索质量差             → 上下文和问题不相关，模型只能猜
```

---

### 2.2 Context Recall（上下文召回率）

**检测目标：检索有没有漏掉关键信息**

**定义：** 标准答案（Ground Truth）中包含的信息，有多少比例出现在了检索到的上下文中？

```
分数计算：
  Context Recall = 上下文中能支撑 ground truth 的句子数 / ground truth 总句子数

示例：
  Ground Truth（标准答案）：
    句子1："AM04 表示余额不足。"
    句子2："处理方式是通知客户充值。"
    句子3："系统会自动生成拒绝报文 pacs.002。"
  
  检索到的上下文包含：
    ✅ 句子1 的信息
    ✅ 句子2 的信息
    ❌ 句子3 的信息（这个 chunk 没被检索到！）
  
  Context Recall = 2 / 3 = 0.67
```

**目标值：> 0.8**。低于 0.6 说明检索遗漏了关键信息，答案会不完整。

**⚠️ 需要 Ground Truth：** Context Recall 是唯一必须提供标准答案的指标。

```
Context Recall 低的诊断：
  → 增大 k 值（检索更多候选）
  → 使用 Hybrid Search（BM25 + 向量）
  → 使用 Multi-Query 扩大覆盖
  → 换更好的 Embedding 模型
  → 减小 chunk_size（让语义更聚焦）
```

---

### 2.3 Context Precision（上下文精确率）

**检测目标：检索结果有多少噪音**

**定义：** 检索到的文档块中，有多少是真正对回答问题有用的？排名越靠前的有用文档，分数越高。

```
分数计算（带位置权重）：
  检索结果：[Doc1(有用), Doc2(无用), Doc3(有用), Doc4(无用), Doc5(有用)]
  
  位置加权精确率：
    @1: 1/1 = 1.0     (Doc1 有用，位置1)
    @2: 1/2 = 0.5     (Doc2 无用，位置2)
    @3: 2/3 = 0.67    (Doc3 有用，位置3)
    @4: 2/4 = 0.5     (Doc4 无用，位置4)
    @5: 3/5 = 0.6     (Doc5 有用，位置5)
  
  Context Precision = (1.0 + 0.67 + 0.6) / 3 = 0.76
  （只对有用文档的位置求平均，所以 Doc2/Doc4 不算）
```

**目标值：> 0.7**。越高说明检索结果越"干净"，LLM 干扰越少。

**与 Context Recall 的关系：**

```
                    高 Precision    低 Precision
高 Recall    ✅ 理想状态         ⚠️ 召回全但噪音多
低 Recall    ⚠️ 准但不全         ❌ 最差，又少又不准
```

```
Context Precision 低的诊断：
  → 减小 k 值（少检索一些候选）
  → 加 Reranker（精排过滤噪音）
  → 增大 chunk_size（语义更完整）
  → 提高 score_threshold（过滤低相关度文档）
```

---

### 2.4 Answer Relevancy（答案相关性）

**检测目标：答案有没有回答问题（而不是跑题）**

**定义：** 通过 Embedding 反向测量——让 LLM 根据答案"猜"原始问题，看猜出来的问题和真实问题的相似度。

```
计算步骤（不需要 Ground Truth）：
  1. 原始问题：  "AM04 错误码是什么意思？"
  2. 生成的答案："AM04 表示余额不足，处理方式是充值..."
  3. 让 LLM 根据答案生成 N 个可能的问题：
       - "什么错误码表示余额不足？"
       - "AM04 的含义是什么？"
       - "如何处理余额不足的错误？"
  4. Embed 这些问题和原始问题
  5. 计算相似度平均值 → Answer Relevancy 分数

直觉理解：
  如果答案很好地回答了问题，那反推出的问题应该和原始问题相似
  如果答案跑题了，反推出的问题就会和原始问题不同
```

**目标值：> 0.7**。

```
Answer Relevancy 低的诊断：
  → Prompt 约束不够（没有要求"直接回答问题"）
  → 上下文太多太杂，模型在回答时"迷失"
  → LLM temperature 过高，回答发散
```

---

### 2.5 Answer Correctness（答案正确性）

**检测目标：和标准答案的一致程度**

**定义：** 综合语义相似度（Semantic Similarity）和事实重叠（Factual Overlap）评分。

```
Answer Correctness = α × Factual_Correctness + (1-α) × Semantic_Similarity

默认 α = 0.75（偏重事实）
```

**目标值：> 0.7**。这是最直接的"答案质量"指标，但依赖 Ground Truth。

---

### 2.6 指标全览速查

| 指标 | 检测什么 | 需要 GT | 目标值 | 最影响因素 |
|------|---------|---------|--------|-----------|
| **Faithfulness** | 幻觉程度 | ❌ | > 0.8 | LLM温度、Prompt约束 |
| **Context Recall** | 检索完整性 | ✅ | > 0.8 | k值、Embedding质量 |
| **Context Precision** | 检索干净度 | ❌ | > 0.7 | Reranker、score_threshold |
| **Answer Relevancy** | 答案是否跑题 | ❌ | > 0.7 | Prompt设计 |
| **Answer Correctness** | 与标准答案一致性 | ✅ | > 0.7 | 整体RAG质量 |

> **GT = Ground Truth（标准答案）**。没有标准答案时，优先看前三个。

---

## 3. LLM-as-Judge：用 LLM 评估 LLM

### 3.1 为什么用 LLM 当裁判

RAGAS 的评估本质上是语义理解任务——判断"这个陈述是否有上下文支撑"，这需要自然语言理解能力，而 LLM 恰好擅长这个：

```
传统评估方式：
  BLEU / ROUGE → 字符串重叠率 → 不理解语义
  "猫坐在垫子上" vs "宠物躺在地毯上" → ROUGE=0（完全不同词）
  但人类判断：语义很相似！

LLM-as-Judge：
  把两段文本发给 LLM，问"这两段意思是否相同？"
  → 能理解语义等价、同义词、逻辑蕴含关系
  → 更接近人类判断
```

### 3.2 LLM-as-Judge 的 Prompt 设计

RAGAS 内部用的 Faithfulness 评估 Prompt 类似这样：

```python
FAITHFULNESS_JUDGE_PROMPT = """
你是一个严格的评估裁判。

给定以下上下文和一个陈述，判断该陈述是否完全基于上下文支撑。

上下文：
{context}

陈述：
{statement}

判断标准：
- 如果陈述中的所有信息都能从上下文中直接推断，回答 YES
- 如果陈述包含任何上下文中没有的信息，回答 NO
- 不要考虑陈述是否正确，只看是否有上下文支撑

只回答 YES 或 NO，不要解释。
"""
```

### 3.3 自定义 LLM Judge

你可以自己实现 LLM-as-Judge，用于 RAGAS 不支持的自定义评估维度：

```python
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class JudgeResult:
    score: float          # 0.0 ~ 1.0
    reasoning: str        # 评分理由
    verdict: str          # YES/NO 或 具体评级

class LLMJudge:
    """
    可复用的 LLM-as-Judge 评估器
    支持多种评估维度
    """
    
    def __init__(self, model="qwen3:8b", use_groq=False):
        if use_groq:
            self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
        else:
            self.llm = ChatOllama(model=model, temperature=0.0)  # 评估用 0 温度，保证一致性
    
    def evaluate_faithfulness(self, context: str, answer: str) -> JudgeResult:
        """评估答案忠实度（幻觉检测）"""
        prompt = ChatPromptTemplate.from_template("""
你是一个严格的 AI 评估专家。

请评估以下答案是否完全基于给定的上下文，没有任何无中生有的内容。

上下文（Context）：
{context}

生成的答案（Answer）：
{answer}

评估步骤：
1. 列出答案中的每个关键陈述
2. 对每个陈述判断：上下文中是否有直接支撑？
3. 计算有支撑的陈述比例

请以 JSON 格式输出：
{{
  "claims": [
    {{"claim": "陈述内容", "supported": true/false, "evidence": "上下文中的支撑句子或null"}}
  ],
  "faithfulness_score": 0.0到1.0之间的数字,
  "verdict": "FAITHFUL 或 HALLUCINATED",
  "summary": "一句话总结"
}}
""")
        
        response = self.llm.invoke(
            prompt.format_messages(context=context[:3000], answer=answer)
        )
        
        try:
            # 提取 JSON（LLM 可能在 JSON 前后加了额外文字）
            text = response.content
            start = text.find('{')
            end = text.rfind('}') + 1
            result = json.loads(text[start:end])
            
            return JudgeResult(
                score=result.get("faithfulness_score", 0.0),
                reasoning=result.get("summary", ""),
                verdict=result.get("verdict", "UNKNOWN")
            )
        except json.JSONDecodeError:
            # JSON 解析失败时的回退
            score = 1.0 if "FAITHFUL" in response.content.upper() else 0.0
            return JudgeResult(score=score, reasoning=response.content, verdict="PARSE_ERROR")
    
    def evaluate_answer_quality(
        self, 
        question: str, 
        answer: str, 
        context: str,
        criteria: str = "accuracy, completeness, clarity"
    ) -> JudgeResult:
        """通用答案质量评估（自定义标准）"""
        prompt = ChatPromptTemplate.from_template("""
你是一个专业的 AI 系统评估专家。

问题：{question}
上下文：{context}
生成的答案：{answer}

请根据以下标准（{criteria}）对答案质量进行评分（1-5分）：

评分标准：
5分：完全正确、完整、清晰，完美回答了问题
4分：基本正确，有轻微不足
3分：部分回答了问题，但有明显缺失或不准确
2分：回答偏题或有严重错误
1分：完全没有回答问题或完全错误

请以 JSON 格式输出：
{{
  "score_1_to_5": 整数,
  "score_normalized": 0.0到1.0,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["缺点1", "缺点2"],
  "improvement_suggestion": "改进建议"
}}
""")
        
        response = self.llm.invoke(prompt.format_messages(
            question=question, context=context[:2000], 
            answer=answer, criteria=criteria
        ))
        
        try:
            text = response.content
            start, end = text.find('{'), text.rfind('}') + 1
            result = json.loads(text[start:end])
            return JudgeResult(
                score=result.get("score_normalized", 0.0),
                reasoning=result.get("improvement_suggestion", ""),
                verdict=str(result.get("score_1_to_5", 0))
            )
        except:
            return JudgeResult(score=0.5, reasoning=response.content, verdict="PARSE_ERROR")
    
    def compare_answers(
        self, 
        question: str, 
        answer_a: str, 
        answer_b: str,
        context: str = ""
    ) -> dict:
        """A/B 对比两个答案（用于优化前后对比）"""
        prompt = ChatPromptTemplate.from_template("""
请对比以下两个答案，判断哪个更好。

问题：{question}
{context_section}

答案 A：{answer_a}

答案 B：{answer_b}

请以 JSON 格式输出：
{{
  "winner": "A 或 B 或 TIE",
  "winner_score": 0.0到1.0（胜者优势程度）,
  "a_strengths": ["A的优点"],
  "b_strengths": ["B的优点"],
  "reasoning": "判断理由"
}}
""")
        
        context_section = f"参考上下文：{context[:1500]}" if context else ""
        response = self.llm.invoke(prompt.format_messages(
            question=question, answer_a=answer_a, 
            answer_b=answer_b, context_section=context_section
        ))
        
        try:
            text = response.content
            start, end = text.find('{'), text.rfind('}') + 1
            return json.loads(text[start:end])
        except:
            return {"winner": "UNKNOWN", "reasoning": response.content}


# 使用示例
judge = LLMJudge(use_groq=True)  # 用 Groq 免费 API 当裁判，速度快

result = judge.evaluate_faithfulness(
    context="AM04 错误码表示账户余额不足，处理方式是通知客户充值。",
    answer="AM04 表示余额不足，客户应充值后重试，也可以尝试分期付款。"
)
print(f"Faithfulness 分数：{result.score:.2f}")
print(f"评估结论：{result.verdict}")
print(f"理由：{result.reasoning}")
```

### 3.4 LLM Judge 的偏差与局限

使用 LLM-as-Judge 时需要了解其已知偏差：

```
位置偏差（Position Bias）：
  LLM 倾向于认为第一个选项更好
  → 解决：随机化 A/B 顺序，各跑两次取平均

冗长偏差（Verbosity Bias）：
  LLM 倾向于认为更长的答案更好
  → 解决：在 prompt 中明确"简洁也是优点"

自我偏好（Self-Preference）：
  用 GPT-4 评估时，倾向于给 GPT-4 的答案打高分
  → 解决：用不同家的 LLM 做裁判，或用本地模型（无品牌偏向）

一致性问题（Inconsistency）：
  同一个问题多次评估可能分数不同
  → 解决：temperature=0，评估同一样本多次取平均
```

---

## 4. RAGAS 完整实现

### 4.1 安装与配置

```bash
pip install ragas datasets langchain-community --break-system-packages
```

### 4.2 使用本地 Ollama 做评估（零成本）

RAGAS 默认用 OpenAI API 做 Judge，但可以换成本地 Ollama，完全免费：

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from datasets import Dataset

# ====== 配置 Judge LLM 和 Embedding ======

# 选项1：本地 Ollama（完全免费，速度慢）
judge_llm = LangchainLLMWrapper(
    ChatOllama(model="qwen3:8b", num_ctx=8192, temperature=0.0)
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)

# 选项2：Groq（免费API，速度快，推荐用于学习）
from langchain_groq import ChatGroq
judge_llm = LangchainLLMWrapper(
    ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
)
# Embedding 还是用本地
judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)

# ====== 配置各指标使用的 LLM/Embedding ======
faithfulness.llm = judge_llm
answer_relevancy.llm = judge_llm
answer_relevancy.embeddings = judge_embeddings
context_precision.llm = judge_llm
context_recall.llm = judge_llm
answer_correctness.llm = judge_llm
answer_correctness.embeddings = judge_embeddings
```

### 4.3 准备评估数据集格式

```python
# RAGAS 需要的数据格式
eval_data = [
    {
        "question":    "AM04 错误码是什么意思？",
        "answer":      rag_chain.invoke("AM04 错误码是什么意思？"),  # RAG 系统生成的答案
        "contexts":    [doc.page_content for doc in retriever.invoke("AM04 错误码")],  # 检索到的文档块列表
        "ground_truth": "AM04 错误码表示账户余额不足（Insufficient Funds），系统会拒绝交易并返回 pacs.002 报文。",  # 标准答案（Context Recall/Answer Correctness 必需）
    },
    {
        "question":    "FTH 系统的超时机制是怎样的？",
        "answer":      rag_chain.invoke("FTH 系统的超时机制是怎样的？"),
        "contexts":    [doc.page_content for doc in retriever.invoke("FTH 超时机制")],
        "ground_truth": "FTH 系统设置了 30 秒超时阈值，超时后自动触发回滚并生成告警。",
    },
    # ... 更多测试案例（推荐 20-50 条）
]

dataset = Dataset.from_list(eval_data)
```

### 4.4 运行评估

```python
import time

print("🚀 开始 RAGAS 评估...")
start = time.time()

results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ],
    # 不再需要传 llm/embeddings，已在上方全局配置
    raise_exceptions=False,  # 遇到错误继续，不中断
)

elapsed = time.time() - start
print(f"✅ 评估完成，耗时：{elapsed:.1f}s")
print("\n📊 评估结果：")
print(results)

# 保存为 DataFrame 方便分析
df = results.to_pandas()
df.to_csv("ragas_results.csv", index=False)
print(f"\n💾 结果已保存至 ragas_results.csv")
```

**输出示例：**
```
📊 评估结果：
{'faithfulness': 0.8234, 
 'answer_relevancy': 0.7891, 
 'context_precision': 0.7456, 
 'context_recall': 0.7123, 
 'answer_correctness': 0.7634}
```

### 4.5 批量评估工具类

```python
import pandas as pd
from typing import List, Dict
from datetime import datetime

class RAGEvaluator:
    """
    完整的 RAG 评估工具，支持：
    - 批量评估
    - 多配置对比
    - 结果持久化
    - 趋势追踪
    """
    
    def __init__(self, judge_llm, judge_embeddings):
        self.judge_llm = judge_llm
        self.judge_embeddings = judge_embeddings
        self._configure_metrics()
        self.history = []  # 历史评估记录
    
    def _configure_metrics(self):
        """配置所有指标使用相同的 Judge"""
        for metric in [faithfulness, answer_relevancy, context_precision, 
                       context_recall, answer_correctness]:
            metric.llm = self.judge_llm
        for metric in [answer_relevancy, answer_correctness]:
            metric.embeddings = self.judge_embeddings
    
    def build_dataset(self, rag_chain, retriever, test_cases: List[Dict]) -> Dataset:
        """
        自动构建评估数据集
        
        Args:
            rag_chain: RAG 问答链
            retriever: 检索器
            test_cases: [{"question": ..., "ground_truth": ...}, ...]
        """
        eval_data = []
        for i, case in enumerate(test_cases):
            q = case["question"]
            print(f"  处理 {i+1}/{len(test_cases)}: {q[:40]}...")
            
            contexts = [doc.page_content for doc in retriever.invoke(q)]
            answer = rag_chain.invoke(q)
            
            eval_data.append({
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": case.get("ground_truth", ""),
            })
        
        return Dataset.from_list(eval_data)
    
    def evaluate(self, dataset: Dataset, run_name: str = "") -> pd.DataFrame:
        """运行评估，返回结果 DataFrame"""
        metrics_to_use = [faithfulness, answer_relevancy, context_precision]
        
        # 如果有 ground_truth，加上需要它的指标
        sample = dataset[0]
        if sample.get("ground_truth"):
            metrics_to_use += [context_recall, answer_correctness]
        
        results = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            raise_exceptions=False,
        )
        
        df = results.to_pandas()
        
        # 记录历史
        summary = {
            "run_name": run_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "faithfulness": df["faithfulness"].mean(),
            "answer_relevancy": df["answer_relevancy"].mean(),
            "context_precision": df["context_precision"].mean(),
            "context_recall": df.get("context_recall", pd.Series([None])).mean(),
            "answer_correctness": df.get("answer_correctness", pd.Series([None])).mean(),
            "n_samples": len(df),
        }
        self.history.append(summary)
        
        return df
    
    def compare_configs(
        self, 
        configs: Dict[str, dict],  # {config_name: {pipeline_kwargs}}
        test_cases: List[Dict],
        pipeline_builder  # 构建 pipeline 的函数
    ) -> pd.DataFrame:
        """对比不同 RAG 配置的评估分数"""
        all_summaries = []
        
        for config_name, config_kwargs in configs.items():
            print(f"\n{'='*50}")
            print(f"📋 评估配置：{config_name}")
            
            pipeline = pipeline_builder(**config_kwargs)
            dataset = self.build_dataset(
                rag_chain=pipeline.query,
                retriever=lambda q: pipeline._retrieve(q),
                test_cases=test_cases
            )
            
            df = self.evaluate(dataset, run_name=config_name)
            
            summary = {
                "配置": config_name,
                "Faithfulness": f"{df['faithfulness'].mean():.3f}",
                "Context Precision": f"{df['context_precision'].mean():.3f}",
                "Answer Relevancy": f"{df['answer_relevancy'].mean():.3f}",
            }
            if "context_recall" in df.columns:
                summary["Context Recall"] = f"{df['context_recall'].mean():.3f}"
            
            all_summaries.append(summary)
            print(f"  Faithfulness: {summary['Faithfulness']}")
        
        comparison_df = pd.DataFrame(all_summaries)
        print("\n\n📊 配置对比汇总：")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def save_history(self, path="evaluation_history.csv"):
        """保存评估历史，追踪改进趋势"""
        pd.DataFrame(self.history).to_csv(path, index=False)
        print(f"📈 评估历史已保存：{path}")
```

---

## 5. 构建评估数据集

### 5.1 评估数据集的要求

好的评估数据集是整个评估体系的基础：

```
数量：20-100 条（学习阶段 20 条足够）
覆盖：不同类型的问题（事实型、推理型、比较型、程序型）
Ground Truth：尽量手写，而不是直接用 LLM 生成
多样性：覆盖文档库的不同主题，包含容易检索和难以检索的问题
边界情况：包含文档中没有答案的问题（测试 Faithfulness）
```

### 5.2 手工标注 vs 自动生成

```python
# 方法1：手工标注（最可靠，但费时）
manual_test_cases = [
    {
        "question": "AM04 错误码的含义是什么？",
        "ground_truth": "AM04 表示余额不足（Insufficient Funds），是 ISO 20022 标准中的标准错误码。",
        "difficulty": "easy",
        "category": "factual"
    },
    {
        "question": "FTH 系统在什么情况下会触发自动重试？",
        "ground_truth": "当支付请求超时（>30s）且错误码为可重试类型时，FTH 最多自动重试 3 次。",
        "difficulty": "medium",
        "category": "procedural"
    },
    {
        "question": "谁发明了量子计算机？",  # 文档中没有的内容
        "ground_truth": "文档中未包含此信息。",
        "difficulty": "hard",
        "category": "out_of_scope"  # 测试 Faithfulness 的重要案例
    },
]


# 方法2：用 LLM 自动生成测试问题（再手工审核）
def generate_test_questions(documents: list, llm, n_per_doc=3) -> list:
    """从文档中自动生成测试问答对"""
    test_cases = []
    
    for doc in documents[:10]:  # 取前10个文档块
        prompt = f"""根据以下文档内容，生成 {n_per_doc} 个测试问答对。

文档内容：
{doc.page_content}

要求：
- 问题要多样：包含事实型、程序型、解释型
- 答案必须完全来自文档，不要添加额外信息
- 用 JSON 数组格式输出：[{{"question": "...", "ground_truth": "..."}}]
- 只输出 JSON，不要其他内容"""
        
        response = llm.invoke(prompt)
        try:
            text = response.content
            start, end = text.find('['), text.rfind(']') + 1
            pairs = json.loads(text[start:end])
            for pair in pairs:
                pair["source"] = doc.metadata.get("source", "unknown")
                test_cases.append(pair)
        except:
            pass
    
    print(f"✅ 自动生成 {len(test_cases)} 个测试案例（建议手工审核）")
    return test_cases
```

### 5.3 问题类型覆盖建议

```
事实型（Factual）—— 约 40%
  "AM04 是什么？"
  "FTH 系统的超时时间是多少？"
  → 测试 Context Recall 和 Faithfulness

推理型（Reasoning）—— 约 20%
  "AM04 和 AC01 的区别是什么？"
  "为什么需要在 pacs.002 中包含错误码？"
  → 测试模型综合多块内容的能力

程序型（Procedural）—— 约 20%
  "如何处理 AM04 错误？步骤是什么？"
  "支付超时后的处理流程？"
  → 测试 Answer Relevancy

范围外（Out-of-Scope）—— 约 20%
  "比特币的价格是多少？"（与文档无关）
  "请推荐一本书"
  → 专门测试 Faithfulness（应该说不知道，而不是编造）
```

---

## 6. 评估结果分析与诊断

### 6.1 分数解读速查

```
Faithfulness 低（< 0.6）：
  严重幻觉，用户不能信任答案
  ─── 修复路径 ───
  1. temperature 降到 0.1
  2. Prompt 加强："只基于上下文，不要添加文档中没有的信息"
  3. 检查 num_ctx（是否因截断导致上下文丢失）
  4. 查看具体哪些问题得分低 → 找规律

Context Recall 低（< 0.6）：
  检索在漏关键信息，答案不完整
  ─── 修复路径 ───
  1. 增大 k（从 5 → 10）
  2. 加 Multi-Query
  3. 换更好的 Embedding（nomic-embed-text → BGE-M3）
  4. 尝试 Hybrid Search（BM25 + 向量）
  5. 减小 chunk_size（语义更聚焦）

Context Precision 低（< 0.5）：
  检索噪音太多，LLM 被无关内容干扰
  ─── 修复路径 ───
  1. 加 Reranker
  2. 减小 k（从 10 → 5）
  3. 增大 score_threshold（只取高相关度的块）
  4. 增大 chunk_size（减少破碎化）

Answer Relevancy 低（< 0.6）：
  答案没有直接回答问题（跑题）
  ─── 修复路径 ───
  1. 优化 Prompt："请直接回答问题，不要绕弯子"
  2. 减少 context 长度（太多内容让模型分心）
  3. 降低 temperature
  4. 检查是否 num_ctx 不足导致问题被截断

所有分数都低：
  → 数据入库有问题（检查 chunk_size 和 embedding）
  → Judge LLM 能力不足（换更强的 Judge）
  → 测试集本身质量差（手工审核 ground_truth）
```

### 6.2 问题级别分析

整体分数只是平均值，更有价值的是找出低分问题的共同特征：

```python
def analyze_failures(df: pd.DataFrame, threshold=0.6):
    """
    分析评估失败的案例，找出共同问题
    """
    print("\n🔍 失败案例分析")
    print("=" * 60)
    
    # Faithfulness 失败案例
    faith_failures = df[df["faithfulness"] < threshold]
    if len(faith_failures) > 0:
        print(f"\n❌ Faithfulness 低分案例（{len(faith_failures)} 个）：")
        for _, row in faith_failures.iterrows():
            print(f"\n  问题：{row['question']}")
            print(f"  得分：{row['faithfulness']:.2f}")
            print(f"  答案（前100字）：{str(row['answer'])[:100]}")
    
    # Context Recall 失败案例
    if "context_recall" in df.columns:
        recall_failures = df[df["context_recall"] < threshold]
        if len(recall_failures) > 0:
            print(f"\n❌ Context Recall 低分案例（{len(recall_failures)} 个）：")
            for _, row in recall_failures.iterrows():
                print(f"\n  问题：{row['question']}")
                print(f"  得分：{row['context_recall']:.2f}")
                print(f"  标准答案：{str(row['ground_truth'])[:80]}")
                n_ctx = len(row.get('contexts', []))
                print(f"  检索到的文档块数：{n_ctx}")
    
    # 统计低分问题类型
    print(f"\n📊 各指标低分比例：")
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in df.columns:
            low_pct = (df[col] < threshold).mean() * 100
            print(f"  {col}: {low_pct:.1f}% 的问题低于 {threshold}")


# 使用
df = results.to_pandas()
analyze_failures(df, threshold=0.6)
```

### 6.3 优化前后对比

```python
def compare_before_after(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    change_description: str
):
    """可视化优化前后的分数变化"""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metrics = [m for m in metrics if m in before_df.columns and m in after_df.columns]
    
    print(f"\n📊 优化效果对比：{change_description}")
    print(f"{'指标':<25} {'优化前':>8} {'优化后':>8} {'变化':>8}")
    print("-" * 55)
    
    for metric in metrics:
        before = before_df[metric].mean()
        after = after_df[metric].mean()
        delta = after - before
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        color = "✅" if delta > 0.02 else "❌" if delta < -0.02 else "➖"
        print(f"{color} {metric:<23} {before:>8.3f} {after:>8.3f} {arrow}{abs(delta):>6.3f}")
```

---

## 7. 持续评估体系（CI/CD）

### 7.1 为什么需要持续评估

每次修改系统（换 embedding、调 chunk_size、改 prompt），都应该跑一次评估，确保没有退步：

```
开发工作流：
  修改代码
      ↓
  运行 evaluate.py
      ↓
  分数 > 上次基准？
      ├── 是 → 提交代码，更新基准分数
      └── 否 → 找出原因，回滚或修复
```

### 7.2 评估基准文件

```python
# baseline_scores.json（提交到 git，记录当前最佳分数）
import json

BASELINE_FILE = "evaluation_baseline.json"

def save_baseline(scores: dict, description: str = ""):
    """保存当前评估分数为基准"""
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "description": description,
        "scores": scores
    }
    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"✅ 基准分数已保存：{BASELINE_FILE}")

def check_regression(current_scores: dict, threshold=0.02):
    """检查是否有性能退步（regression）"""
    try:
        with open(BASELINE_FILE) as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print("⚠️ 未找到基准文件，跳过退步检查")
        return True
    
    has_regression = False
    print("\n🔍 退步检查：")
    
    for metric, current in current_scores.items():
        baseline_val = baseline["scores"].get(metric, 0)
        delta = current - baseline_val
        
        if delta < -threshold:  # 下降超过阈值
            print(f"  ❌ {metric}: {baseline_val:.3f} → {current:.3f} (下降 {abs(delta):.3f}!)")
            has_regression = True
        else:
            status = "✅" if delta > 0 else "➖"
            print(f"  {status} {metric}: {baseline_val:.3f} → {current:.3f}")
    
    return not has_regression  # True = 无退步
```

---

## 8. 完整项目实战

### 8.1 项目结构

```
06_evaluation/
├── test_cases.json          # 测试数据集（手工标注）
├── evaluate.py              # 主评估脚本
├── judge.py                 # 自定义 LLM Judge
├── analyze.py               # 结果分析和可视化
├── baseline_scores.json     # 当前最佳分数基准
└── results/
    ├── ragas_results.csv    # 最新评估结果
    └── history.csv          # 历史评估趋势
```

### 8.2 Claude Code 提示词

```
在 06_evaluation/ 目录创建完整的 RAG 评估系统：

1. test_cases.json
   为支付系统 RAG 场景创建 20 条测试数据：
   - 5条事实型（直接查错误码含义）
   - 5条程序型（问处理流程）
   - 5条推理型（问为什么、区别等）
   - 5条范围外（文档中没有的问题，测试幻觉）
   每条包含：question, ground_truth, category, difficulty

2. evaluate.py
   - 用 Groq 作为 Judge LLM（ChatGroq, llama-3.3-70b）
   - 本地 nomic-embed-text 作为 Judge Embedding
   - 评估5个指标：faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
   - 自动对比5种配置：基础RAG、+MultiQuery、+Reranking、+HyDE、全部开启
   - 输出 results/comparison_table.csv

3. analyze.py
   - 读取 ragas_results.csv
   - 生成失败案例报告（分数 < 0.6 的问题详情）
   - 用 matplotlib 画雷达图对比5种配置
   - 输出诊断建议（基于各指标分数自动判断优化方向）

4. judge.py
   - 实现 LLMJudge 类（faithfulness + answer_quality + compare）
   - 写5条 out-of-scope 问题的评估，验证 Faithfulness 检测是否有效

注意：
- 所有 LLM 调用加 try/except，避免 API 超时导致中断
- 评估结果自动追加到 results/history.csv 做趋势记录
```

---

## 总结

RAGAS 评估体系把 RAG 的"感觉好用"变成了"数据可证"：

```
五个核心指标，分两类：

不需要 Ground Truth（可以零成本持续跑）：
  Faithfulness      → 幻觉检测，最重要
  Context Precision → 检索干净度
  Answer Relevancy  → 答案是否跑题

需要 Ground Truth（需要手工标注，但提供最完整信息）：
  Context Recall    → 检索完整性
  Answer Correctness → 与标准答案一致性

学习阶段推荐：
  先用 Groq 免费 API 当 Judge（速度快）
  20条测试数据 × 3个无需GT的指标 = 最快的评估起点
  跑通后再扩展到全部5个指标
```

---

*文档版本：v1.0 | 2026年3月 | 对应学习计划 Week 4-5*
