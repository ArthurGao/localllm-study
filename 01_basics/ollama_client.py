"""
LLM 基础交互练习
============================
学习目标：
1. 理解 LLM 客户端的基本用法（支持 Ollama 本地 / Groq 云端）
2. 掌握普通调用 vs 流式输出
3. 实现多轮对话（维护 messages history）
4. 演示 Qwen3 的 /think 和 /no_think 模式
5. 简单 CLI 聊天界面

使用方式：
    conda activate llm-learn
    python 01_basics/ollama_client.py

后端切换：
    修改 .env 文件中的 LLM_BACKEND=ollama 或 LLM_BACKEND=groq
"""

import sys
import os

# 让 import config 能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import chat, chat_stream, get_llm_config, print_config


# ============================================================
# 1. 基础调用 - 最简单的一次性问答
# ============================================================
def basic_chat(prompt: str) -> str:
    """
    最基本的调用方式：发送一条消息，等待完整回复。

    核心概念：
    - 通过 config.chat() 统一接口，自动选择 Ollama 或 Groq 后端
    - messages: 对话历史，格式为 [{"role": "user/assistant/system", "content": "..."}]
    """
    return chat(
        [{"role": "user", "content": prompt}],
    )


# ============================================================
# 2. 流式输出 - 逐 token 输出，体验更好
# ============================================================
def stream_chat(prompt: str):
    """
    流式调用：模型边生成边输出，不用等全部生成完。

    核心概念：
    - stream=True: 启用流式模式
    - 返回的是一个迭代器，每次 yield 一小段文本
    - 适合长回答或需要实时显示的场景
    """
    cfg = get_llm_config()
    print(f"\n{'='*50}")
    print(f"[流式输出] 问题: {prompt}")
    print(f"[后端: {cfg['backend']}, 模型: {cfg['model']}]")
    print(f"{'='*50}")

    full_response = ""
    for text in chat_stream([{"role": "user", "content": prompt}]):
        print(text, end="", flush=True)
        full_response += text

    print()  # 换行
    return full_response


# ============================================================
# 3. 多轮对话 - 维护对话历史
# ============================================================
def multi_turn_demo():
    """
    多轮对话的关键：把之前的 messages 都传给模型。

    核心概念：
    - 模型本身是无状态的，每次调用都是独立的
    - 要实现"记忆"，必须手动维护 messages 列表
    - 每次把完整的对话历史发送给模型
    - messages 越长，消耗的 tokens 越多（注意 context window 限制）
    """
    cfg = get_llm_config()
    print(f"\n{'='*50}")
    print(f"[多轮对话演示] 后端: {cfg['backend']}")
    print(f"{'='*50}")

    messages = []

    # 可选：设置 system prompt 定义助手角色
    messages.append(
        {
            "role": "system",
            "content": "你是一个友好的AI助手，回答简洁明了，用中文回答。",
        }
    )

    # 模拟3轮对话
    questions = [
        "Python 的 GIL 是什么？一句话解释",
        "那它对多线程有什么影响？",  # 这个问题依赖上文的"它"
        "有什么替代方案？",  # 继续追问
    ]

    for q in questions:
        print(f"\n用户: {q}")
        messages.append({"role": "user", "content": q})

        answer = chat(messages)

        # 把助手回复也加入历史，这样下一轮模型能看到
        messages.append({"role": "assistant", "content": answer})
        print(f"助手: {answer}")

    print(f"\n[对话历史共 {len(messages)} 条消息]")


# ============================================================
# 4. Think / No-Think 模式 - Qwen3 特有功能
# ============================================================
def thinking_mode_demo():
    """
    Qwen3 支持两种思考模式：

    /think  - 深度思考模式：模型会先在内部推理，再给出答案
             适合复杂问题（数学、逻辑、代码）
    /no_think - 快速回答模式：直接给答案，跳过推理过程
               适合简单问题（翻译、知识问答）

    核心概念：
    - 在 prompt 前加 /think 或 /no_think 来切换
    - think 模式会返回 <think>...</think> 标签包裹的推理过程
    - 可以通过解析 <think> 标签来分离思考过程和最终答案

    注意：/think /no_think 是 Qwen3 特有功能，Groq 上的其他模型不支持。
    """
    cfg = get_llm_config()
    print(f"\n{'='*50}")
    print(f"[Think / No-Think 模式对比] 后端: {cfg['backend']}")
    print(f"{'='*50}")

    question = "计算 17 * 23 + 45 的结果"

    # --- No-Think 模式 ---
    print("\n--- /no_think 模式（快速回答）---")
    response = chat(
        [{"role": "user", "content": f"/no_think {question}"}],
    )
    print(response)

    # --- Think 模式 ---
    print("\n--- /think 模式（深度思考）---")
    response = chat(
        [{"role": "user", "content": f"/think {question}"}],
    )
    content = response

    # 解析 <think> 标签
    if "<think>" in content and "</think>" in content:
        think_start = content.index("<think>") + len("<think>")
        think_end = content.index("</think>")
        thinking = content[think_start:think_end].strip()
        answer = content[think_end + len("</think>") :].strip()
        print(f"思考过程: {thinking[:200]}...")  # 只显示前200字
        print(f"最终答案: {answer}")
    else:
        print(content)


# ============================================================
# 5. 调节参数 - Temperature, top_p 等
# ============================================================
def parameter_demo():
    """
    常用参数说明：

    - temperature: 控制随机性 (0=确定性, 1=较随机, 2=很随机)
      RAG 任务建议 0.1-0.3，创意任务 0.7+
    - num_ctx: 上下文窗口大小（token数）— Ollama 专用
    - top_p: nucleus sampling，和 temperature 配合使用
    - seed: 设置随机种子，相同 seed 输出一致（可复现）
    """
    cfg = get_llm_config()
    prompt = "/no_think 用一句话描述春天"

    # --- 实验 1: Temperature 对比 ---
    print(f"\n{'='*50}")
    print(f"[实验 1] Temperature 对比 — 后端: {cfg['backend']}")
    print(f"{'='*50}")
    print(f"Prompt: {prompt}")

    for temp in [0.1, 0.7, 1.5]:
        print(f"\n  temperature={temp}:")
        for run in range(2):
            result = chat(
                [{"role": "user", "content": prompt}],
                temperature=temp, num_ctx=4096,
            )
            print(f"    第{run+1}次: {result}")

    # --- 实验 2: top_p 对比 ---
    print(f"\n{'='*50}")
    print(f"[实验 2] top_p 对比（固定 temperature=0.8）")
    print(f"{'='*50}")
    print(f"Prompt: {prompt}")

    for top_p in [0.3, 0.9, 1.0]:
        print(f"\n  top_p={top_p}:")
        for run in range(2):
            result = chat(
                [{"role": "user", "content": prompt}],
                temperature=0.8, top_p=top_p, num_ctx=4096,
            )
            print(f"    第{run+1}次: {result}")

    # --- 实验 3: Temperature + top_p 组合 ---
    print(f"\n{'='*50}")
    print(f"[实验 3] Temperature + top_p 组合效果")
    print(f"{'='*50}")

    combos = [
        {"temperature": 0.1, "top_p": 0.9, "label": "精确模式 (RAG)"},
        {"temperature": 0.8, "top_p": 0.9, "label": "通用对话"},
        {"temperature": 1.2, "top_p": 0.95, "label": "创意模式"},
        {"temperature": 0.8, "top_p": 0.3, "label": "高温+窄采样"},
    ]

    creative_prompt = "/no_think 写一句关于月亮的诗句"
    print(f"Prompt: {creative_prompt}")

    for combo in combos:
        print(f"\n  {combo['label']} (temp={combo['temperature']}, top_p={combo['top_p']}):")
        for run in range(2):
            result = chat(
                [{"role": "user", "content": creative_prompt}],
                temperature=combo["temperature"],
                top_p=combo["top_p"],
                num_ctx=4096,
            )
            print(f"    第{run+1}次: {result}")

    # --- 实验 4: seed 可复现性 ---
    print(f"\n{'='*50}")
    print(f"[实验 4] seed 可复现性（temperature=0.8, 相同 seed）")
    print(f"{'='*50}")
    print(f"Prompt: {prompt}")

    print("\n  相同 seed=42，两次调用:")
    for run in range(2):
        result = chat(
            [{"role": "user", "content": prompt}],
            temperature=0.8, seed=42, num_ctx=4096,
        )
        print(f"    第{run+1}次: {result}")

    print("\n  不同 seed（42 vs 123）:")
    for seed in [42, 123]:
        result = chat(
            [{"role": "user", "content": prompt}],
            temperature=0.8, seed=seed, num_ctx=4096,
        )
        print(f"    seed={seed}: {result}")


# ============================================================
# 6. CLI 交互式聊天
# ============================================================
def interactive_chat():
    """
    简单的命令行聊天界面。

    特殊命令：
    - /quit   退出聊天
    - /clear  清空对话历史
    - /think  下一条消息使用深度思考
    - /no_think 下一条消息使用快速回答
    - /history 查看对话历史条数
    - /backend 查看当前后端配置
    """
    cfg = get_llm_config()
    print(f"\n{'='*50}")
    print(f"LLM Chat - 后端: {cfg['backend']}, 模型: {cfg['model']}")
    print("输入 /quit 退出, /clear 清空历史, /help 查看帮助")
    print(f"{'='*50}\n")

    messages = [
        {
            "role": "system",
            "content": "你是一个有用的AI助手，用中文回答问题。回答简洁明了。",
        }
    ]

    while True:
        try:
            user_input = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("再见！")
            break
        elif user_input == "/clear":
            messages = [messages[0]]  # 保留 system prompt
            print("[对话历史已清空]")
            continue
        elif user_input == "/history":
            print(f"[当前对话历史: {len(messages)} 条消息]")
            continue
        elif user_input == "/backend":
            print_config()
            continue
        elif user_input == "/help":
            print("命令: /quit 退出 | /clear 清空 | /history 查看历史 | /backend 查看后端")
            print("技巧: 消息前加 /think 或 /no_think 切换思考模式（Qwen3）")
            continue

        messages.append({"role": "user", "content": user_input})

        # 流式输出回复
        print("助手: ", end="", flush=True)
        full_response = ""
        for text in chat_stream(messages):
            print(text, end="", flush=True)
            full_response += text
        print()

        messages.append({"role": "assistant", "content": full_response})


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print_config()
        print("-" * 30)
        print("1. basic    - 基础调用")
        print("2. stream   - 流式输出")
        print("3. multi    - 多轮对话")
        print("4. think    - Think/NoThink 模式")
        print("5. params   - 参数调节")
        print("6. chat     - 交互式聊天")
        print("-" * 30)
        mode = input("选择模式 (1-6): ").strip()

    mode_map = {
        "1": "basic",
        "basic": "basic",
        "2": "stream",
        "stream": "stream",
        "3": "multi",
        "multi": "multi",
        "4": "think",
        "think": "think",
        "5": "params",
        "params": "params",
        "6": "chat",
        "chat": "chat",
    }

    mode = mode_map.get(mode, mode)

    if mode == "basic":
        result = basic_chat("用一句话解释什么是大语言模型")
        print(f"回复: {result}")
    elif mode == "stream":
        stream_chat("介绍一下 RAG（检索增强生成）的基本原理")
    elif mode == "multi":
        multi_turn_demo()
    elif mode == "think":
        thinking_mode_demo()
    elif mode == "params":
        parameter_demo()
    elif mode == "chat":
        interactive_chat()
    else:
        print(f"未知模式: {mode}")
