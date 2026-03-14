"""
LLM 后端配置
============
统一管理 Ollama（本地）和 Groq（云端）两种后端。

使用方式：
    from config import get_llm_config, get_chat_client

环境变量（通过 .env 文件配置）：
    LLM_BACKEND=ollama          # 或 groq
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_MODEL=qwen3:8b
    GROQ_API_KEY=gsk_xxx
    GROQ_MODEL=qwen-qwq-32b    # Groq 上可用的 Qwen 模型
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


# ============================================================
# 配置读取
# ============================================================
def get_llm_config() -> dict:
    """
    读取 LLM 配置，返回统一的配置字典。

    返回:
        {
            "backend": "ollama" | "groq",
            "model": "qwen3:8b" | "qwen-qwq-32b" | ...,
            "ollama_host": "http://...",    # 仅 ollama
            "groq_api_key": "gsk_...",      # 仅 groq
        }
    """
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    config = {"backend": backend}

    if backend == "groq":
        config["model"] = os.getenv("GROQ_MODEL", "qwen-qwq-32b")
        config["groq_api_key"] = os.getenv("GROQ_API_KEY", "")
        if not config["groq_api_key"]:
            raise ValueError("GROQ_API_KEY 未设置，请在 .env 文件中配置")
    else:
        config["model"] = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        config["ollama_host"] = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    return config


# ============================================================
# Ollama 客户端（支持远程）
# ============================================================
def get_ollama_client():
    """获取 Ollama 客户端，支持本地和远程。"""
    import ollama

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if host != "http://localhost:11434":
        return ollama.Client(host=host)
    return ollama


# ============================================================
# Groq 客户端
# ============================================================
def get_groq_client():
    """获取 Groq 客户端。"""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY 未设置，请在 .env 文件中配置")
    return Groq(api_key=api_key)


# ============================================================
# 统一聊天接口
# ============================================================
def chat(messages: list[dict], **options) -> str:
    """
    统一聊天接口，根据 LLM_BACKEND 自动选择后端。

    Args:
        messages: [{"role": "user", "content": "..."}]
        **options: temperature, top_p, top_k, seed, max_tokens 等

    Returns:
        模型回复文本
    """
    cfg = get_llm_config()

    if cfg["backend"] == "groq":
        return _chat_groq(messages, cfg, **options)
    else:
        return _chat_ollama(messages, cfg, **options)


def chat_stream(messages: list[dict], **options):
    """
    统一流式聊天接口，yield 每个 token。

    Args:
        messages: [{"role": "user", "content": "..."}]
        **options: temperature, top_p, top_k, seed, max_tokens 等

    Yields:
        str: 每次生成的文本片段
    """
    cfg = get_llm_config()

    if cfg["backend"] == "groq":
        yield from _stream_groq(messages, cfg, **options)
    else:
        yield from _stream_ollama(messages, cfg, **options)


# ============================================================
# 内部实现
# ============================================================
def _chat_ollama(messages: list[dict], cfg: dict, **options) -> str:
    client = get_ollama_client()
    ollama_options = _to_ollama_options(options)
    response = client.chat(
        model=cfg["model"],
        messages=messages,
        options=ollama_options,
    )
    return response["message"]["content"]


def _stream_ollama(messages: list[dict], cfg: dict, **options):
    client = get_ollama_client()
    ollama_options = _to_ollama_options(options)
    stream = client.chat(
        model=cfg["model"],
        messages=messages,
        options=ollama_options,
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


def _chat_groq(messages: list[dict], cfg: dict, **options) -> str:
    client = get_groq_client()
    groq_params = _to_groq_params(options)
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        **groq_params,
    )
    return response.choices[0].message.content


def _stream_groq(messages: list[dict], cfg: dict, **options):
    client = get_groq_client()
    groq_params = _to_groq_params(options)
    stream = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        stream=True,
        **groq_params,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def _to_ollama_options(options: dict) -> dict:
    """将统一参数转换为 Ollama 格式。"""
    mapping = {
        "max_tokens": "num_predict",
    }
    result = {}
    for k, v in options.items():
        result[mapping.get(k, k)] = v
    return result


def _to_groq_params(options: dict) -> dict:
    """将统一参数转换为 Groq API 格式。"""
    result = {}
    # Groq 支持的参数
    if "temperature" in options:
        result["temperature"] = options["temperature"]
    if "top_p" in options:
        result["top_p"] = options["top_p"]
    if "max_tokens" in options or "num_predict" in options:
        result["max_tokens"] = options.get("max_tokens", options.get("num_predict"))
    if "seed" in options:
        result["seed"] = options["seed"]
    if "stop" in options:
        result["stop"] = options["stop"]
    # Groq 不支持: top_k, repeat_penalty, num_ctx — 静默忽略
    return result


# ============================================================
# 辅助函数
# ============================================================
def print_config():
    """打印当前 LLM 配置。"""
    cfg = get_llm_config()
    print(f"LLM 后端: {cfg['backend']}")
    print(f"模型: {cfg['model']}")
    if cfg["backend"] == "ollama":
        print(f"Ollama 地址: {cfg['ollama_host']}")
    else:
        print(f"Groq API Key: {cfg['groq_api_key'][:10]}...")


if __name__ == "__main__":
    print_config()

    # 快速测试
    response = chat(
        [{"role": "user", "content": "用一句话介绍你自己"}],
        temperature=0.7,
    )
    print(f"\n回复: {response}")
