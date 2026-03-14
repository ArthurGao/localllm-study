"""
LLM 参数实验
============
通过实际调用 LLM，直观感受各参数对输出的影响。
支持 Ollama（本地）和 Groq（云端）两种后端。

每个实验都会多次调用同一 prompt，方便对比输出差异。

使用方式：
    conda activate llm-learn
    python 01_basics/parameter_experiments.py           # 运行全部实验
    python 01_basics/parameter_experiments.py temp      # 只跑 temperature
    python 01_basics/parameter_experiments.py top_p     # 只跑 top_p
    python 01_basics/parameter_experiments.py top_k     # 只跑 top_k（仅 Ollama）
    python 01_basics/parameter_experiments.py seed      # 只跑 seed
    python 01_basics/parameter_experiments.py repeat    # 只跑 repeat_penalty（仅 Ollama）
    python 01_basics/parameter_experiments.py num_ctx   # 只跑 num_ctx（仅 Ollama）
    python 01_basics/parameter_experiments.py num_predict  # 只跑 max_tokens
    python 01_basics/parameter_experiments.py combo     # 只跑组合实验

后端切换：
    修改 .env 文件中的 LLM_BACKEND=ollama 或 LLM_BACKEND=groq
"""

import sys
import os
import time

# 让 import config 能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import chat, get_llm_config, print_config


def header(title: str):
    cfg = get_llm_config()
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  后端: {cfg['backend']} | 模型: {cfg['model']}")
    print(f"{'='*60}")


def run_comparison(prompt: str, param_name: str, values: list, fixed_options: dict = None, runs: int = 2):
    """通用对比函数：对同一 prompt，变化某个参数，每个值跑 runs 次。"""
    fixed = fixed_options or {}
    for val in values:
        print(f"\n  {param_name}={val}:")
        opts = {**fixed, param_name: val}
        for r in range(runs):
            result = chat([{"role": "user", "content": prompt}], **opts)
            # 只取第一行，避免输出太长
            first_line = result.strip().split("\n")[0]
            print(f"    第{r+1}次: {first_line}")


def is_ollama() -> bool:
    """检查当前是否使用 Ollama 后端。"""
    return get_llm_config()["backend"] == "ollama"


# ============================================================
# 实验 1: Temperature
# ============================================================
def experiment_temperature():
    """
    Temperature 控制输出的随机性。
    - 低温 (0.1): 几乎确定性输出，两次结果基本相同
    - 中温 (0.7): 自然多样，适合对话
    - 高温 (1.5): 高度随机，可能出现意想不到的表达

    Ollama 和 Groq 都支持此参数。
    """
    header("实验 1: Temperature（温度）")

    prompt = "/no_think 用一句话描述春天"
    print(f"  Prompt: {prompt}")
    print(f"  观察: 温度越低，两次输出越相似")

    run_comparison(prompt, "temperature", [0.1, 0.5, 0.7, 1.0, 1.5],
                   fixed_options={"num_ctx": 4096})

    # 补充：temperature=0 的确定性
    print(f"\n  --- temperature=0（完全确定性，3次应完全相同）---")
    for r in range(3):
        result = chat([{"role": "user", "content": prompt}], temperature=0, num_ctx=4096)
        first_line = result.strip().split("\n")[0]
        print(f"    第{r+1}次: {first_line}")


# ============================================================
# 实验 2: top_p (Nucleus Sampling)
# ============================================================
def experiment_top_p():
    """
    top_p 截断候选 token 集合。Ollama 和 Groq 都支持。
    """
    header("实验 2: top_p（核采样）")

    prompt = "/no_think 用一句话描述春天"
    print(f"  Prompt: {prompt}")
    print(f"  固定: temperature=0.8")
    print(f"  观察: top_p 越小，输出越保守/重复")

    run_comparison(prompt, "top_p", [0.1, 0.3, 0.5, 0.9, 1.0],
                   fixed_options={"temperature": 0.8, "num_ctx": 4096})

    print(f"\n  --- 换一个更有创意空间的 prompt ---")
    creative = "/no_think 用一个比喻描述人工智能"
    print(f"  Prompt: {creative}")

    run_comparison(creative, "top_p", [0.3, 0.9],
                   fixed_options={"temperature": 0.8, "num_ctx": 4096}, runs=3)


# ============================================================
# 实验 3: top_k（仅 Ollama）
# ============================================================
def experiment_top_k():
    """top_k 限制只从概率最高的 K 个 token 中采样。Groq 不支持。"""
    if not is_ollama():
        print("\n  ⚠ top_k 参数仅 Ollama 支持，Groq 不支持此参数，跳过。")
        return

    header("实验 3: top_k")

    prompt = "/no_think 写一句关于月亮的诗句"
    print(f"  Prompt: {prompt}")
    print(f"  固定: temperature=0.8, num_ctx=4096")
    print(f"  观察: top_k 越小，输出越单调")

    run_comparison(prompt, "top_k", [1, 5, 10, 40, 100],
                   fixed_options={"temperature": 0.8, "num_ctx": 4096})


# ============================================================
# 实验 4: seed（可复现性）
# ============================================================
def experiment_seed():
    """seed 固定随机数生成器。Ollama 和 Groq 都支持。"""
    header("实验 4: seed（随机种子）")

    prompt = "/no_think 用一句话描述春天"
    print(f"  Prompt: {prompt}")
    print(f"  固定: temperature=0.8")

    print(f"\n  --- 相同 seed=42，3次调用（应完全相同）---")
    for r in range(3):
        result = chat([{"role": "user", "content": prompt}],
                      temperature=0.8, seed=42, num_ctx=4096)
        first_line = result.strip().split("\n")[0]
        print(f"    第{r+1}次: {first_line}")

    print(f"\n  --- 不同 seed（应各不相同）---")
    for seed in [42, 123, 456, 999]:
        result = chat([{"role": "user", "content": prompt}],
                      temperature=0.8, seed=seed, num_ctx=4096)
        first_line = result.strip().split("\n")[0]
        print(f"    seed={seed}: {first_line}")

    print(f"\n  --- seed 在 temperature=0.1 下（差异很小）---")
    for seed in [42, 123]:
        result = chat([{"role": "user", "content": prompt}],
                      temperature=0.1, seed=seed, num_ctx=4096)
        first_line = result.strip().split("\n")[0]
        print(f"    seed={seed}: {first_line}")


# ============================================================
# 实验 5: repeat_penalty（仅 Ollama）
# ============================================================
def experiment_repeat_penalty():
    """repeat_penalty 降低已出现 token 再次被选中的概率。Groq 不支持。"""
    if not is_ollama():
        print("\n  ⚠ repeat_penalty 参数仅 Ollama 支持，Groq 不支持此参数，跳过。")
        return

    header("实验 5: repeat_penalty（重复惩罚）")

    prompt = "/no_think 请详细列举春天的5个特征，每个特征用一句话描述"
    print(f"  Prompt: {prompt}")
    print(f"  固定: temperature=0.8, num_ctx=4096")
    print(f"  观察: penalty 越低越可能重复用词，越高用词越丰富但可能不自然")

    for penalty in [1.0, 1.1, 1.5, 2.0]:
        print(f"\n  repeat_penalty={penalty}:")
        result = chat([{"role": "user", "content": prompt}],
                      temperature=0.8, repeat_penalty=penalty, num_ctx=4096)
        for line in result.strip().split("\n")[:7]:
            print(f"    {line}")


# ============================================================
# 实验 6: num_ctx（仅 Ollama）
# ============================================================
def experiment_num_ctx():
    """num_ctx 决定模型能"看到"多少 token。Groq 由模型决定，不可手动调节。"""
    if not is_ollama():
        print("\n  ⚠ num_ctx 参数仅 Ollama 支持，Groq 由模型决定上下文窗口，跳过。")
        return

    header("实验 6: num_ctx（上下文窗口）")

    long_context = """以下是一段关于四季的描述，请仔细阅读后回答问题。

春天：万物复苏，花朵盛开，温度回暖，鸟儿歌唱。春雨绵绵，滋润大地。农民开始春耕播种。
夏天：烈日炎炎，蝉鸣阵阵，人们喜欢游泳消暑。雷阵雨频繁，西瓜是最受欢迎的水果。
秋天：金风送爽，落叶纷飞，丰收的季节。月饼和螃蟹是秋天的美食代表。天空格外湛蓝。
冬天：白雪皑皑，寒风刺骨，人们穿上厚重的棉衣。过年是最重要的节日，全家团聚吃年夜饭。

特别信息：小明最喜欢的季节是秋天，因为他喜欢吃螃蟹。小红最喜欢冬天，因为可以堆雪人。"""

    question = f"/no_think {long_context}\n\n问题：小明为什么最喜欢秋天？"

    print(f"  测试：模型能否正确引用长上下文中的信息")
    print(f"  Prompt 长度: ~{len(question)} 字符")

    for ctx in [512, 2048, 4096]:
        print(f"\n  num_ctx={ctx}:")
        start = time.time()
        result = chat([{"role": "user", "content": question}],
                      temperature=0.1, num_ctx=ctx)
        elapsed = time.time() - start
        first_line = result.strip().split("\n")[0]
        print(f"    回答: {first_line}")
        print(f"    耗时: {elapsed:.1f}s")


# ============================================================
# 实验 7: max_tokens（最大生成长度）
# ============================================================
def experiment_num_predict():
    """max_tokens 限制模型最多生成多少个 token。Ollama 和 Groq 都支持。"""
    header("实验 7: max_tokens（最大生成长度）")

    prompt = "/no_think 详细介绍 Python 这门编程语言的主要特点"
    print(f"  Prompt: {prompt}")
    print(f"  观察: max_tokens 越小，回答越短/被截断")

    for max_tokens in [20, 50, 150, 500]:
        print(f"\n  max_tokens={max_tokens}:")
        result = chat([{"role": "user", "content": prompt}],
                      temperature=0.3, max_tokens=max_tokens, num_ctx=4096)
        display = result.strip()
        if len(display) > 200:
            display = display[:200] + "..."
        for line in display.split("\n")[:5]:
            print(f"    {line}")
        print(f"    [输出总长: {len(result)} 字符]")


# ============================================================
# 实验 8: 参数组合对比
# ============================================================
def experiment_combo():
    """不同场景的推荐参数组合对比。"""
    header("实验 8: 参数组合 — 不同场景推荐配置")

    prompt = "/no_think 写一段关于AI未来发展的思考，3句话"

    combos = [
        {
            "label": "RAG / 精确问答",
            "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 4096},
        },
        {
            "label": "代码生成",
            "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
        },
        {
            "label": "通用对话",
            "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 4096},
        },
        {
            "label": "创意写作",
            "options": {"temperature": 1.0, "top_p": 0.95, "num_ctx": 4096},
        },
        {
            "label": "极端保守 (低温+窄top_p)",
            "options": {"temperature": 0.1, "top_p": 0.3, "num_ctx": 4096},
        },
        {
            "label": "极端随机 (高温+宽top_p)",
            "options": {"temperature": 1.5, "top_p": 1.0, "num_ctx": 4096},
        },
    ]

    print(f"  Prompt: {prompt}\n")

    for combo in combos:
        print(f"  --- {combo['label']} ---")
        opts_str = ", ".join(f"{k}={v}" for k, v in combo["options"].items() if k != "num_ctx")
        print(f"  参数: {opts_str}")
        for r in range(2):
            result = chat([{"role": "user", "content": prompt}], **combo["options"])
            display = result.strip().replace("\n", " ")
            if len(display) > 150:
                display = display[:150] + "..."
            print(f"    第{r+1}次: {display}")
        print()


# ============================================================
# 主入口
# ============================================================
EXPERIMENTS = {
    "temp": ("Temperature", experiment_temperature),
    "top_p": ("top_p", experiment_top_p),
    "top_k": ("top_k（仅 Ollama）", experiment_top_k),
    "seed": ("seed", experiment_seed),
    "repeat": ("repeat_penalty（仅 Ollama）", experiment_repeat_penalty),
    "num_ctx": ("num_ctx（仅 Ollama）", experiment_num_ctx),
    "num_predict": ("max_tokens", experiment_num_predict),
    "combo": ("参数组合", experiment_combo),
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key in EXPERIMENTS:
            name, func = EXPERIMENTS[key]
            print_config()
            print(f"\n运行实验: {name}")
            func()
        else:
            print(f"未知实验: {key}")
            print(f"可选: {', '.join(EXPERIMENTS.keys())}")
    else:
        print("LLM 参数实验")
        print("=" * 40)
        print_config()
        print(f"共 {len(EXPERIMENTS)} 个实验\n")

        for key, (name, _) in EXPERIMENTS.items():
            print(f"  {key:>12} → {name}")

        print()
        choice = input("选择实验（回车=全部运行）: ").strip()

        if choice and choice in EXPERIMENTS:
            EXPERIMENTS[choice][1]()
        else:
            for key, (name, func) in EXPERIMENTS.items():
                func()
                print()

    print("\n实验完成！详细参数说明见 01_basics/llm_parameters.md")
