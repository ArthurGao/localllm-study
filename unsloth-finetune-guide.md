# Unsloth 微调完整指南
## QLoRA → GGUF → 导入 Ollama 运行

> 阶段：Week 5-7 | 前置：基础 RAG + 评估体系已建立
>
> 核心目标：把通用 Qwen 模型微调成专精你的业务领域的专家模型，再导入 Ollama 本地使用。

---

## 目录

1. [微调的目的与时机](#1-微调的目的与时机)
2. [LoRA / QLoRA 原理](#2-lora--qlora-原理)
3. [Unsloth 环境搭建](#3-unsloth-环境搭建)
4. [数据集准备](#4-数据集准备)
5. [QLoRA 微调全流程](#5-qlora-微调全流程)
6. [导出 GGUF 格式](#6-导出-gguf-格式)
7. [导入 Ollama 运行](#7-导入-ollama-运行)
8. [微调效果评估](#8-微调效果评估)
9. [免费 GPU 资源](#9-免费-gpu-资源)
10. [完整项目实战](#10-完整项目实战)

---

## 1. 微调的目的与时机

### 1.1 RAG vs 微调：不是非此即彼

很多人认为微调和 RAG 是竞争关系，实际上它们解决的是不同问题：

```
RAG 解决的问题：
  ✅ 需要引用最新/私有文档
  ✅ 需要答案可溯源（能指出来自哪段文档）
  ✅ 文档内容经常更新
  ✅ 需要精确的数字/名称（向量库存原文）

微调解决的问题：
  ✅ 改变模型的输出风格和格式
  ✅ 让模型学会领域专业术语和习惯表达
  ✅ 固化特定的行为模式（如始终按某种格式回答）
  ✅ 减少 Prompt 工程的负担（把 prompt 里的规则"烧"进模型）

最佳实践：RAG + 微调组合
  微调 → 让模型学会你的领域语言和回答风格
  RAG  → 提供最新、精确的文档内容
```

### 1.2 什么时候值得微调

```
值得微调的场景：
  ✅ 模型不懂你的领域缩写（如 "FTH" 总被误解）
  ✅ 模型回答风格与业务要求不符（太啰嗦、格式不对）
  ✅ 特定任务准确率低（如：提取 ISO 20022 报文字段）
  ✅ 需要模型按特定 JSON 格式输出
  ✅ 有 100+ 条高质量的领域问答数据

不值得微调的场景：
  ❌ 数据集 < 50 条（效果不稳定，容易过拟合）
  ❌ 只需要访问最新文档（用 RAG 更合适）
  ❌ 没有 GPU（CPU 微调极慢，不实际）
  ❌ 只想让模型更"聪明"（微调不增加世界知识）
```

### 1.3 微调类型速览

| 类型 | 全名 | 更新参数比例 | 资源需求 | 本指南涵盖 |
|------|------|------------|---------|-----------|
| **Full Fine-tuning** | 全参数微调 | 100% | 极高（A100×8） | ❌ |
| **LoRA** | Low-Rank Adaptation | ~1% | 中等（RTX 4090） | ⭐ |
| **QLoRA** | Quantized LoRA | ~1% | 低（RTX 3060/免费GPU） | ⭐ |
| **Prefix Tuning** | 前缀调优 | < 0.1% | 低 | ❌ |

本指南专注 **QLoRA**：免费 GPU 可运行，效果接近全参数微调。

---

## 2. LoRA / QLoRA 原理

### 2.1 LoRA：低秩适配（Low-Rank Adaptation）

**直觉理解：** 模型的全参数微调需要更新数十亿个权重，代价极高。LoRA 的洞察是：微调时权重的**变化量**（delta W）具有低秩（Low-Rank）结构，可以用两个小矩阵的乘积来近似。

```
原始权重矩阵 W（维度 d×d，如 4096×4096 = 1600万参数）：

全参数微调：W_new = W + ΔW
            ΔW 有 d×d 个参数 → 太大！

LoRA：      W_new = W + A × B
            A 是 d×r 矩阵（如 4096×16）
            B 是 r×d 矩阵（如 16×4096）
            r = rank（秩），通常 4-64

参数量对比：
  LoRA：  2 × 4096 × 16 = 131,072 个参数
  全参数：4096 × 4096 = 16,777,216 个参数
  节省：  约 128 倍！
```

**为什么有效：** 研究表明，神经网络在微调时的"本质维度（Intrinsic Dimensionality）"远低于参数空间维度。模型只需沿少数几个"方向"调整就能适应新任务，这正是低秩矩阵擅长表达的。

### 2.2 QLoRA：量化 + LoRA

**QLoRA = 4-bit 量化基础模型 + LoRA 适配器（FP16）**

```
标准 LoRA 内存需求（7B 模型）：
  FP16 模型加载：     14 GB
  梯度 + 优化器状态： 28 GB
  合计：             ~42 GB → 需要 A100 80GB

QLoRA 内存需求（7B 模型）：
  NF4 量化模型：      4-5 GB  ← 压缩 3 倍
  LoRA 适配器（FP16）：0.3 GB
  梯度（仅 LoRA 部分）：0.5 GB
  合计：             ~6 GB  → RTX 3060 12GB 可跑！
```

**NF4（Normal Float 4）量化：** QLoRA 专用格式，基于正态分布优化，比普通 INT4 精度更高，专为神经网络权重设计。

```
训练时：
  基础模型权重（W，4bit NF4）→ 冻结，不更新梯度
  LoRA 矩阵（A, B，FP16）   → 可训练，梯度回传到这里

推理时两种选择：
  在线合并：每次推理时计算 W + A×B（略慢）
  离线合并：先合并 W_merged = W + A×B，和原始模型一样快
```

---

## 3. Unsloth 环境搭建

### 3.1 为什么选 Unsloth

```
速度：比 HuggingFace PEFT 快 2-5x
内存：减少 60-80% 显存使用
精度：自定义 CUDA kernel，不损失计算精度
兼容：完全兼容 HuggingFace 生态（datasets, transformers, trl）
支持模型：Qwen3/2.5、Llama3、Gemma3 等主流模型
```

### 3.2 本地安装（有 NVIDIA GPU）

```bash
# 创建独立虚拟环境（避免依赖冲突）
python -m venv unsloth_env
source unsloth_env/bin/activate

# 安装 PyTorch CUDA 版本
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 安装 Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# 验证
python -c "from unsloth import FastLanguageModel; print('✅ Unsloth 安装成功')"
```

### 3.3 Kaggle / Colab（免费 GPU）

```python
# Kaggle Notebook 第一个 cell：
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

# 验证 GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# 预期输出：Tesla T4, 15.8 GB
```

---

## 4. 数据集准备

### 4.1 指令微调数据格式（ShareGPT 格式）

```python
# ShareGPT 格式（最通用，Unsloth 直接支持）
{
    "conversations": [
        {"role": "system",    "value": "你是一个专业的支付系统技术专家。"},
        {"role": "user",      "value": "AM04 错误码是什么意思？"},
        {"role": "assistant", "value": "AM04 是 ISO 20022 标准中的错误码，"
                                       "表示账户余额不足（Insufficient Funds）。\n\n"
                                       "处理建议：\n"
                                       "1. 通知客户补充余额\n"
                                       "2. 记录错误日志供分析\n"
                                       "3. 在 pacs.002 报文中返回此错误码"}
    ]
}
```

### 4.2 数据集质量标准

```
数量：100-1000 条（学习阶段 100 条可出效果）
质量 > 数量：10条高质量 > 100条低质量

必须满足：
  ✅ 每条答案都正确（错误数据会训练出错误行为）
  ✅ 所有答案风格统一
  ✅ 覆盖不同问题类型（事实、程序、对比、边界情况）
  ✅ 包含 10-20% 的通用对话（防止灾难性遗忘）
```

### 4.3 自动生成数据集（再手工审核）

```python
import json
from langchain_ollama import ChatOllama

def build_training_dataset(documents: list, llm, n_per_doc=5,
                            system_prompt="你是一个专业的技术文档专家。"):
    """从文档自动生成微调数据集"""
    dataset = []
    
    for i, doc in enumerate(documents):
        print(f"处理 {i+1}/{len(documents)}: {doc.metadata.get('source','')[:30]}")
        
        prompt = f"""基于以下技术文档内容，生成 {n_per_doc} 个高质量问答对。

文档内容：
{doc.page_content}

要求：
- 问题类型多样：定义型、程序型、场景型、对比型
- 答案完全来自文档，不添加额外内容
- 答案要专业完整，步骤用编号列出
- JSON 数组格式，每项含 instruction 和 output 字段

只输出 JSON："""
        
        try:
            response = llm.invoke(prompt)
            text = response.content
            start, end = text.find('['), text.rfind(']') + 1
            pairs = json.loads(text[start:end])
            
            for pair in pairs:
                if pair.get("instruction") and pair.get("output"):
                    dataset.append({
                        "conversations": [
                            {"role": "system",    "value": system_prompt},
                            {"role": "user",      "value": pair["instruction"]},
                            {"role": "assistant", "value": pair["output"]}
                        ]
                    })
        except Exception as e:
            print(f"  ⚠️ 跳过：{e}")
    
    print(f"\n✅ 生成 {len(dataset)} 条（请手工审核后再训练！）")
    return dataset
```

### 4.4 数据集示例（支付系统领域）

```json
[
  {
    "conversations": [
      {"role": "system", "value": "你是一个专业的支付系统工程师，熟悉 ISO 20022 和 FTH 系统。"},
      {"role": "user",   "value": "AM04 和 AC01 错误码的区别是什么？"},
      {"role": "assistant", "value": "两者都是 ISO 20022 的拒绝原因码，但含义不同：\n\n**AM04（InsufficientFunds）：**\n- 含义：账户余额不足\n- 触发：可用余额 < 交易金额\n- 处理：通知客户充值后重试\n\n**AC01（IncorrectAccountNumber）：**\n- 含义：账号格式错误或不存在\n- 触发：目标账号无法识别\n- 处理：核实收款账号后重新发起\n\n关键区别：AM04 是余额问题（可通过充值解决），AC01 是账号问题（需核实账号信息）。"}
    ]
  },
  {
    "conversations": [
      {"role": "system", "value": "你是一个专业的支付系统工程师，熟悉 ISO 20022 和 FTH 系统。"},
      {"role": "user",   "value": "FTH 系统收到 AM04 后，完整的处理流程是什么？"},
      {"role": "assistant", "value": "FTH 系统处理 AM04 的完整流程：\n\n**1. 接收阶段**\n- Finacle API 返回包含 AM04 的错误响应\n- FTH 解析错误码并记录日志\n\n**2. 报文构建阶段**\n- 构建 pacs.002（FIToFIPaymentStatusReport）拒绝报文\n- ReasonCode 字段填入 AM04\n- TransactionStatus 设为 RJCT\n\n**3. 通知阶段**\n- 通过 IBM MQ 将 pacs.002 发送给发起行
- 触发客户通知（短信/邮件）

**4. 监控阶段**
- 记录 AM04 发生频率
- 超过阈值触发风控告警"}
    ]
  }
]
```

---

## 5. QLoRA 微调全流程

### 5.1 完整微调脚本

```python
"""
finetune_qwen.py - 完整的 QLoRA 微调脚本
"""
import torch, json
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ====== 配置 ======
MODEL_NAME     = "unsloth/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
BATCH_SIZE     = 2
GRAD_ACCUM     = 4      # 等效 batch = 8
LEARNING_RATE  = 2e-4
NUM_EPOCHS     = 3
OUTPUT_DIR     = "./finetuned_model"

# ====== 1. 加载模型 ======
print("📥 加载基础模型（4bit 量化）...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,   # QLoRA 核心：4bit 加载
)
print(f"✅ 显存使用：{torch.cuda.memory_allocated()/1e9:.2f} GB")

# ====== 2. 注入 LoRA ======
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"✅ 可训练参数：{trainable:,}（{trainable/total*100:.2f}%）")

# ====== 3. 准备数据 ======
with open("training_data.json") as f:
    raw_data = json.load(f)

def format_conv(example):
    text = tokenizer.apply_chat_template(
        example["conversations"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = Dataset.from_list(raw_data).map(format_conv, remove_columns=["conversations"])
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"✅ 训练：{len(dataset['train'])} 条 | 验证：{len(dataset['test'])} 条")

# ====== 4. 训练 ======
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        seed=42,
    ),
)

print("🚀 开始训练...")
stats = trainer.train()
print(f"✅ 训练完成 | Loss: {stats.training_loss:.4f} | 步数: {stats.global_step}")

# ====== 5. 测试 ======
FastLanguageModel.for_inference(model)
test_input = tokenizer.apply_chat_template(
    [{"role": "user", "content": "AM04 错误码是什么意思？"}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")
output = model.generate(input_ids=test_input, max_new_tokens=300, temperature=0.1)
print("\n模型回答：")
print(tokenizer.decode(output[0][test_input.shape[1]:], skip_special_tokens=True))
```

### 5.2 训练 Loss 解读

```
正常收敛（好）：
  step  10: loss = 2.41
  step  50: loss = 1.73
  step 100: loss = 1.24
  step 200: loss = 1.05  ← 趋于平稳，可停止

过拟合警告（危险）：
  train loss: 1.0 → 0.6 → 0.3（一直下降）
  eval  loss: 1.0 → 1.2 → 1.6（开始上升！）
  → 立即停止，用最低 eval loss 的 checkpoint

Loss 发散（参数问题）：
  step 0:  2.5
  step 10: 3.1 → 3.8  ← 上升！
  → 降低 learning_rate（试 5e-5），检查数据格式
```

---

## 6. 导出 GGUF 格式

### 6.1 Unsloth 直接导出（推荐）

```python
# 训练完成后，一行代码导出
model.save_pretrained_gguf(
    "finetuned_qwen_gguf",
    tokenizer,
    quantization_method="q4_k_m",   # 推荐默认
)
print("✅ GGUF 文件：finetuned_qwen_gguf/")

# 导出多种量化版本（方便对比）
for quant in ["q4_k_m", "q5_k_m", "q8_0"]:
    model.save_pretrained_gguf(f"gguf_{quant}", tokenizer,
                               quantization_method=quant)
    print(f"  ✅ {quant} 导出完成")
```

### 6.2 手动导出（llama.cpp，备用方案）

```bash
# 克隆 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt

# Step 1：合并 LoRA 到基础模型
python scripts/merge_lora.py \
    --base_model unsloth/Qwen2.5-7B-Instruct \
    --lora_model ./finetuned_model \
    --output_dir ./merged_model

# Step 2：转 GGUF
python convert_hf_to_gguf.py ./merged_model \
    --outfile finetuned_qwen.gguf --outtype f16

# Step 3：量化
./llama-quantize finetuned_qwen.gguf finetuned_qwen_q4km.gguf Q4_K_M
```

---

## 7. 导入 Ollama 运行

### 7.1 创建 Modelfile

```dockerfile
# Modelfile

FROM ./finetuned_qwen_gguf/unsloth.Q4_K_M.gguf

# 系统提示词（与训练时保持一致！）
SYSTEM """
你是一个专业的支付系统工程师，熟悉 ISO 20022 标准和 FTH 支付系统。
请根据问题提供准确、专业的技术答案。
如果不确定，请直接说明，不要猜测。
"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
```

### 7.2 导入并运行

```bash
# 导入到 Ollama
ollama create payment-expert -f ./Modelfile

# 验证
ollama list

# 测试
ollama run payment-expert "AM04 错误码是什么意思？"

# 在代码中使用（与基础模型完全相同的接口）
```

```python
from langchain_ollama import ChatOllama

# 只需换模型名，其他代码不变
llm = ChatOllama(model="payment-expert", num_ctx=8192, temperature=0.1)
answer = llm.invoke("FTH 系统的超时机制是什么？")
print(answer.content)
```

### 7.3 版本管理

```bash
# 每次微调后用版本号管理
ollama create payment-expert-v1 -f Modelfile_v1
ollama create payment-expert-v2 -f Modelfile_v2

# 对比
ollama run payment-expert-v1 "AM04 是什么？"
ollama run payment-expert-v2 "AM04 是什么？"

# 清理旧版本
ollama rm payment-expert-v1
```

---

## 8. 微调效果评估

### 8.1 基础模型 vs 微调模型对比

```python
from langchain_ollama import ChatOllama

base_llm      = ChatOllama(model="qwen3:8b",        temperature=0.1)
finetuned_llm = ChatOllama(model="payment-expert",  temperature=0.1)

questions = [
    "AM04 错误码是什么？",
    "pacs.002 报文的主要字段有哪些？",
    "FTH 超时后会发生什么？",
    "AM04 和 AC01 的区别？",
]

print(f"\n{'问题':<35} | {'基础模型（前60字）':<30} | {'微调模型（前60字）'}")
print("-" * 100)
for q in questions:
    base = base_llm.invoke(q).content[:60].replace('\n', ' ')
    fine = finetuned_llm.invoke(q).content[:60].replace('\n', ' ')
    print(f"{q:<35} | {base:<30} | {fine}")
```

### 8.2 常见问题与修复

```
问题：灾难性遗忘（Catastrophic Forgetting）
症状：微调后通用问题质量大幅下降
修复：
  1. 训练数据混入 10-20% 通用对话（Alpaca/ShareGPT）
  2. 减小 r（8 代替 16）
  3. 减少 epochs（1-2 代替 3）

问题：过度记忆训练数据（Memorization）
症状：对训练集问题一字不差复述
修复：
  1. 增大 lora_dropout（0.05 → 0.1）
  2. 增大 weight_decay（0.01 → 0.05）
  3. 增加数据多样性

问题：不遵守格式要求（如不按 JSON 输出）
修复：
  1. 增加格式示例（50+ 条 JSON 格式训练数据）
  2. Modelfile SYSTEM 明确格式要求
  3. user 字段中始终包含格式指令
```

---

## 9. 免费 GPU 资源

| 平台 | GPU | 显存 | 每周限额 | 推荐场景 |
|------|-----|------|---------|---------|
| **Kaggle** | T4×2 / P100 | 16GB | 30小时 | ⭐ 首选，稳定 |
| **Google Colab** | T4 / A100 | 16-80GB | 不固定 | 短时训练 |
| **Lightning.ai** | A10G | 24GB | 22小时/月 | 中等任务 |

**训练耗时参考（T4 GPU）：**
```
100条数据，7B 模型，3 epochs：约 30-60 分钟
500条数据，7B 模型，3 epochs：约 2-3 小时
```

### 训练完成后下载 GGUF

```python
# Kaggle：在右侧 Output 面板点击下载

# Colab：
from google.colab import files
files.download("finetuned_qwen_gguf/unsloth.Q4_K_M.gguf")

# 或上传到 HuggingFace Hub 永久保存：
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj="finetuned_qwen_gguf/unsloth.Q4_K_M.gguf",
    path_in_repo="unsloth.Q4_K_M.gguf",
    repo_id="your-name/payment-expert-qwen",
    repo_type="model",
)
```

---

## 10. 完整项目实战

### 10.1 项目结构

```
07_finetuning/
├── data/
│   ├── raw_generated.json          # LLM 自动生成（待审核）
│   └── training_data.json          # 手工审核后的最终数据
├── finetune_qwen.py                # 主训练脚本
├── Modelfile                       # Ollama 配置
├── evaluate_finetuned.py           # 与基础模型对比
└── README.md                       # 训练参数记录
```

### 10.2 Claude Code 提示词

```
在 07_finetuning/ 目录创建完整微调项目：

1. generate_dataset.py
   - 从 ./docs/ 目录 PDF 自动生成训练数据
   - 每个文档块生成 3-5 个问答对（定义/程序/对比/场景类型）
   - 输出 ShareGPT JSON 格式
   - 打印样本统计和3条示例预览

2. finetune_qwen.py
   基础模型：unsloth/Qwen2.5-7B-Instruct
   QLoRA 配置：r=16, alpha=32, dropout=0.05
   训练配置：batch=2, grad_accum=4, lr=2e-4, epochs=3
   - 每10步打印 loss，每50步评估
   - 训练完立即用5个问题测试
   - 自动导出 q4_k_m GGUF

3. Modelfile
   - 引用导出的 GGUF
   - system: 支付系统工程师角色
   - temperature=0.1, num_ctx=8192

4. evaluate_finetuned.py
   对10个测试问题对比：
   (a) 基础 qwen3:8b
   (b) 微调后 payment-expert
   (c) 微调模型 + RAG
   用 RAGAS 评估 faithfulness 和 answer_relevancy
   输出对比表格 + 结论建议

注意：
  适配 Kaggle（kaggle-new 安装命令）
  支持命令行参数：--epochs N --lr 2e-4 --r 16
  GGUF 导出失败时打印手动导出步骤
```

---

## 总结

```
QLoRA → GGUF → Ollama 完整流程：

数据准备（2-3天）：
  LLM 自动生成 → 手工审核 → 格式化 ShareGPT JSON

训练（Kaggle T4，30-120分钟）：
  4bit 加载基础模型（~5GB 显存）
  注入 LoRA（r=16，仅 1% 参数可训练）
  SFTTrainer 训练 3 个 epoch
  监控 train/eval loss，防止过拟合

导出（10-30分钟）：
  Unsloth 一行代码导出 Q4_K_M GGUF

部署（5分钟）：
  创建 Modelfile → ollama create → 即可使用

关键数字：
  显存需求：~6 GB（7B QLoRA）
  最小训练数据：100条
  可训练参数：~1%
  预期效果：领域专业性 +20-40%（视数据质量）
```

---

*文档版本：v1.0 | 2026年3月 | 对应学习计划 Week 5-7*
