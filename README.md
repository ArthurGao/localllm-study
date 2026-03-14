# Local LLM Study

A hands-on learning project for running LLMs locally with **Ollama + Qwen3**, covering RAG pipelines, evaluation, and fine-tuning across an 8-week curriculum.

## Project Structure

```
localllm-study/
├── 01_basics/              # Ollama API, chat modes, parameter experiments
├── 02_rag/                 # RAG pipeline: document loading, retrieval, FastAPI server
├── 03_rag_advanced/        # (planned) Multi-query, reranking, HyDE
├── 04_evaluation/          # (planned) RAGAS evaluation framework
├── 05_finetune/            # (planned) LoRA/QLoRA fine-tuning with Unsloth
├── 06_rag_finetune/        # (planned) Capstone: RAG + fine-tuning + evaluation
├── data/                   # Test documents for RAG
├── config.py               # Unified LLM backend (Ollama / Groq)
├── requirements.txt        # Core dependencies
└── requirements-finetune.txt  # Fine-tuning deps (NVIDIA GPU only)
```

## Prerequisites

- **Python 3.13** (via conda)
- **Ollama** installed and running
- (Optional) **Groq API key** for cloud inference

### Hardware

| Setup | Recommended Models |
|-------|-------------------|
| CPU only / low-end | Qwen3:0.6B / 1.7B (Q4) |
| 16GB RAM / Apple Silicon | Qwen3:4B / 8B (Q4_K_M) |
| RTX 4070+ / 16GB VRAM | Qwen3:14B (Q4) |

## Installation

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull qwen3:8b            # Main LLM
ollama pull nomic-embed-text    # Embedding model (required for RAG)

# Verify
ollama list
```

### 2. Create conda environment

```bash
conda create -n llm-learn python=3.13 -y
conda activate llm-learn
```

### 3. Install Python dependencies

```bash
# Install PyTorch first
pip install torch torchvision

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** On Linux with no GPU, use the CPU-only index to save disk space:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

> **Fine-tuning (Stage 5):** Requires NVIDIA GPU with CUDA. Install separately on a GPU machine or use Google Colab / Kaggle:
> ```bash
> pip install -r requirements-finetune.txt
> ```

## Configuration

### 1. Create `.env` file

```bash
cp .env.example .env
```

### 2. Configure your backend

Edit `.env` to choose between local Ollama or cloud Groq:

**Option A — Ollama (local, default):**

```env
LLM_BACKEND=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

**Option B — Groq (cloud, fast):**

```env
LLM_BACKEND=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=qwen-qwq-32b
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Verify setup

```bash
# Test Ollama is running
ollama run qwen3:8b "Hello, introduce yourself briefly"

# Test Python environment
python 01_basics/ollama_client.py basic
```

## Usage

### Stage 1: LLM Basics

```bash
# Basic chat, streaming, multi-turn, thinking modes
python 01_basics/ollama_client.py basic
python 01_basics/ollama_client.py stream
python 01_basics/ollama_client.py think
python 01_basics/ollama_client.py chat      # Interactive CLI

# Parameter experiments (temperature, top_p, seed, etc.)
python 01_basics/parameter_experiments.py temp
python 01_basics/parameter_experiments.py top_p
python 01_basics/parameter_experiments.py seed
```

### Stage 2: RAG Pipeline

```bash
# Step 1: Ingest a document into ChromaDB
python 02_rag/document_loader.py --file data/sample_rag_test.txt --query "What is RAG?"

# Step 2: Query the RAG chain
python 02_rag/rag_chain.py --query "How does retrieval work?"
python 02_rag/rag_chain.py --query "Explain embeddings" --stream

# Step 3: Run as REST API
uvicorn 02_rag.app:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

**API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload PDF/TXT/MD for ingestion |
| POST | `/ask` | Ask a question with RAG retrieval |
| GET | `/health` | Check backend status |

## Switching Backends

The project supports switching between Ollama and Groq at runtime via `config.py`:

```python
from config import chat, chat_stream

# Uses whichever backend is set in .env
response = chat([{"role": "user", "content": "Hello"}])

# Streaming
for token in chat_stream([{"role": "user", "content": "Hello"}]):
    print(token, end="")
```

Change backend by editing `LLM_BACKEND` in `.env` — no code changes needed.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `torch` install fails | On macOS: `pip install torch torchvision` (default index). On Linux CPU-only: add `--index-url https://download.pytorch.org/whl/cpu` |
| `unsloth` / `bitsandbytes` fails on Mac | These require NVIDIA GPU — use Colab/Kaggle for fine-tuning |
| Ollama connection refused | Make sure Ollama is running: `ollama serve` |
| ChromaDB collection not found | Run document ingestion first: `python 02_rag/document_loader.py --file <your-file>` |
| Slow inference | Use a smaller model (`qwen3:4b`) or switch to Groq cloud backend |
| `num_ctx` errors | Set `OLLAMA_NUM_CTX=8192` in `.env` or reduce chunk count |

## Learning Curriculum

| Week | Stage | Topic |
|------|-------|-------|
| 1-2 | 1 | Ollama basics, chat modes, parameters |
| 2-3 | 2 | RAG pipeline: load, chunk, embed, retrieve, generate |
| 3-4 | 3 | Advanced RAG: multi-query, reranking, HyDE |
| 4-5 | 4 | RAGAS evaluation framework |
| 5-7 | 5 | Fine-tuning with LoRA/QLoRA (Unsloth) |
| 7-8 | 6 | Capstone: integrated RAG + fine-tuning system |

See [local-llm-learning-plan.md](local-llm-learning-plan.md) for the full curriculum with detailed explanations.
