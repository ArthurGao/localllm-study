# 02_rag — RAG 基础学习模块

基于 LangChain + ChromaDB + Ollama 实现的 RAG（检索增强生成）学习实验室。

---

## 文件结构

```
02_rag/
├── README.md                   # 本文件
│
├── 核心模块（基础功能）
│   ├── document_loader.py      # 文档加载 → 分块 → Embedding → ChromaDB 入库
│   ├── rag_chain.py            # LangChain RAG Chain 构建与问答
│   ├── app.py                  # FastAPI RAG 服务（REST API）
│   └── rag_pipeline.ipynb      # Jupyter 交互式完整流程教程
│
└── 学习实验（新增）
    ├── embedding_explorer.py   # Embedding 向量深度探索（6 个实验）
    ├── chunking_experiments.py # 分块策略对比实验
    ├── retrieval_quality_test.py # 检索质量系统测试
    ├── rag_comparison.py       # RAG vs 无RAG 对比演示
    └── prompt_engineering_rag.py # Prompt 工程对比实验
```

---

## VSCode 快速运行（类似 PyCharm）

**不需要给每个文件写 launch.json 配置！**

| 方式 | 操作 | 说明 |
|------|------|------|
| **F5** | 打开 `.py` 文件 → 按 F5 | 带调试运行当前文件（可打断点） |
| **Ctrl+F5** | 打开 `.py` 文件 → 按 Ctrl+F5 | 不调试直接运行（最快） |
| **终端** | 在终端输入命令 | 需要传参数时用这个 |

> `launch.json` 已配置 `"运行当前文件"` 模式，打开哪个文件就运行哪个，和 PyCharm 的右键 Run 一样。

---

## 前置条件

```bash
conda activate llm-learn

# 确保 Ollama 服务运行
ollama serve

# 拉取所需模型
ollama pull qwen3:8b          # LLM
ollama pull nomic-embed-text  # Embedding 模型
```

---

## 推荐学习顺序

### Step 1 — 理解 Embedding 是什么

```bash
python 02_rag/embedding_explorer.py
```

**不需要先入库文档，直接运行。** 包含 6 个实验：

| 实验 | 学习内容 |
|------|---------|
| 实验1：基础属性 | 向量维度、数值范围、是否归一化 |
| 实验2：语义相似度 | 相似文本 vs 不相关文本的向量距离 |
| 实验3：同义词距离 | 同义词/近义词/反义词在向量空间的位置 |
| 实验4：查询vs文档匹配 | 问题形式 vs 答案形式的语义匹配（RAG 核心） |
| 实验5：向量运算 | 加法、减法、语义类比 |
| 实验6：相似度矩阵 | 多个文本之间的完整相似度矩阵 |

运行单个实验：
```bash
python 02_rag/embedding_explorer.py --exp 4   # 只运行实验4
```

---

### Step 2 — 理解分块策略的影响

```bash
python 02_rag/chunking_experiments.py
```

对同一批文档测试 4 种分块策略，输出对比表格：

| 策略 | 特点 |
|------|------|
| 固定小块 (256字符) | 块多、精准，但可能缺乏上下文 |
| **递归中块 (512字符) ⭐** | 推荐默认，按自然边界切分 |
| 递归大块 (1024字符) | 上下文完整，但检索精度下降 |
| 递归高重叠 (512/200) | 边界信息不丢失，但冗余多 |

还可以运行 chunk_size 敏感度分析（128 到 2048 范围扫描）：
```bash
python 02_rag/chunking_experiments.py --mode sensitivity
```

---

### Step 3 — 系统评估检索质量

先入库测试文档：
```bash
python 02_rag/document_loader.py --file data/rag_knowledge_base.txt
python 02_rag/document_loader.py --file data/sample_rag_test.txt
```

然后运行检索质量测试：
```bash
# 标准测试（12道测试题，自动诊断问题）
python 02_rag/retrieval_quality_test.py

# 自动入库 + 测试（一步完成）
python 02_rag/retrieval_quality_test.py --auto-ingest

# 同时运行 top_k 对比实验
python 02_rag/retrieval_quality_test.py --auto-ingest --compare-k
```

输出包括：
- 每道题的召回情况（✓/△/✗）
- 按难度分组的召回率
- 未命中问题的自动诊断和优化建议
- 不同 top_k 值的效果对比

---

### Step 4 — 理解 RAG 何时有用

```bash
python 02_rag/rag_comparison.py --auto-ingest
```

对 3 类问题分别对比有 RAG 和无 RAG 的回答：

| 问题类型 | 预期结果 |
|---------|---------|
| 文档内事实题 | RAG 更准确，有文档依据 |
| 通用常识题 | LLM 本身就知道，RAG 可能反而干扰 |
| 文档外的问题 | 好的 RAG 应该说"无法回答" |

只测试某一类：
```bash
python 02_rag/rag_comparison.py --auto-ingest --category 1
```

---

### Step 5 — 掌握 Prompt 工程

```bash
python 02_rag/prompt_engineering_rag.py --auto-ingest
```

对比 5 种 Prompt 模板的效果：

| 模板 | 特点 |
|------|------|
| 最简单（无约束） | 无规则，模型自由发挥 |
| **严格约束型** ⭐ | 只根据文档回答，推荐生产使用 |
| 引用来源型 | 回答中标注 [来源X] |
| 分步推理型 | 先列关键信息，再给结论 |
| 坏的 Prompt（反面教材） | 展示糟糕 Prompt 的危害 |

还有 temperature 对比实验（0.0 到 1.0）：
```bash
python 02_rag/prompt_engineering_rag.py --auto-ingest --mode temperature
```

---

### Step 6 — 完整 RAG 流程（基础模块）

**命令行方式：**
```bash
# 1. 入库文档
python 02_rag/document_loader.py --file data/rag_knowledge_base.txt

# 2. 单次问答
python 02_rag/rag_chain.py --query "ChromaDB 有什么特点？"

# 3. 交互式对话
python 02_rag/rag_chain.py

# 4. 入库时顺便测试检索
python 02_rag/document_loader.py --file data/sample_rag_test.txt --query "什么是 RAG？"
```

**Jupyter Notebook：**
```bash
jupyter notebook 02_rag/rag_pipeline.ipynb
```

**FastAPI 服务：**
```bash
uvicorn 02_rag.app:app --reload --port 8000
# API 文档: http://localhost:8000/docs
```

---

## 测试数据

| 文件 | 内容 | 适合实验 |
|------|------|---------|
| `data/sample_rag_test.txt` | RAG 基础概念（原始） | 快速验证 |
| `data/rag_knowledge_base.txt` | RAG 深度知识库（7章） | 检索质量测试 |
| `data/python_ml_basics.txt` | Python ML 基础 | 多文档 RAG |

---

## 关键参数速查

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `chunk_size` | 512 | 分块大小（字符数），太小缺上下文，太大精度差 |
| `chunk_overlap` | 50 | 重叠大小，防止边界截断，建议 chunk_size 的 10% |
| `top_k` | 3~5 | 检索文档数量 |
| `temperature` | 0.1 | RAG 场景用低温度，减少幻觉 |
| `num_ctx` | 8192 | Ollama 上下文窗口，**必须手动设置**，默认 2048 太小 |

---

## 常见问题

**Q: Embedding 很慢？**
A: 第一次需要加载模型，后续会快很多。可用 `--batch-size` 调整批处理大小。

**Q: 检索结果不相关？**
A: 先跑 `retrieval_quality_test.py` 诊断，通常是 chunk_size 或 top_k 需要调整。

**Q: 答案有幻觉？**
A: 检查 `num_ctx` 是否设为 8192，`temperature` 是否为 0.1，Prompt 约束是否够严格。

**Q: ChromaDB 数据想重置？**
A: 删除 `chroma_db/` 目录后重新入库：`rm -rf chroma_db/`
