# Domain 1 · 2.6 — 任务分解策略
> Task Statement 1.6 | 考试权重中等 ⭐⭐

---

## 📌 核心概念速查

| 中文 | 英文关键词 |
|------|-----------|
| 固定顺序流水线 | Fixed Sequential Pipeline / Prompt Chaining |
| 动态自适应分解 | Dynamic Adaptive Decomposition |
| 注意力稀释 | Attention Dilution |
| 本地分析传递 | Per-file Local Analysis Pass |
| 跨文件集成传递 | Cross-file Integration Pass |
| 开放式调查 | Open-ended Investigation |

---

## 🧠 核心知识点

### 1. 两种分解模式的选择

```
场景判断：

"可预测的多方面任务" → Prompt Chaining（固定流水线）
  ✅ 代码审查（每个文件 → 集成检查）
  ✅ 文档生成（提取 → 格式化 → 发布）
  ✅ 数据处理（清洗 → 转换 → 验证）

"开放式调查任务" → Dynamic Decomposition（动态自适应）
  ✅ "给遗留代码库添加全面测试"
  ✅ "调查生产 bug 的根本原因"
  ✅ "评估迁移到微服务的可行性"
```

---

### 2. Prompt Chaining：代码审查示例

```python
def review_pull_request(files: list[str]) -> dict:
    """
    固定流水线：每文件本地分析 + 跨文件集成传递
    解决注意力稀释问题（14个文件同时分析 → 遗漏和矛盾）
    """
    
    # 阶段一：每文件独立分析（本地问题）
    local_results = {}
    for file in files:
        local_results[file] = analyze_single_file(file)
        # 每次只关注一个文件，注意力集中
    
    # 阶段二：跨文件集成分析（系统性问题）
    integration_result = analyze_cross_file_integration(
        files=files,
        local_findings=local_results
        # 专注于：数据流、接口兼容性、循环依赖
    )
    
    return {
        "local_issues": local_results,
        "integration_issues": integration_result
    }
```

**为什么拆分**：14个文件同时分析时，注意力会稀释，导致：
- 某些文件的反馈过于简单
- 在一个文件标记的问题，在另一个文件相同代码却被通过
- 明显的 bug 被遗漏

---

### 3. Dynamic Decomposition：开放式任务

```python
def add_tests_to_legacy_codebase(codebase_path: str):
    """
    动态自适应分解：每步根据发现生成子任务
    不能预先知道需要多少步骤
    """
    
    # Step 1: 探索结构（发现驱动后续步骤）
    structure = map_codebase_structure(codebase_path)
    
    # Step 2: 基于结构识别高影响区域
    high_impact_areas = identify_coverage_gaps(structure)
    
    # Step 3: 动态生成测试计划（基于 Step 2 的发现）
    test_plan = create_prioritized_test_plan(
        areas=high_impact_areas,
        dependencies=structure["dependencies"]  # 运行时才知道
    )
    
    # Step 4: 按优先级执行（计划可能随发现调整）
    for area in test_plan:
        generate_tests(area)
        # 每个区域的测试可能揭示新的依赖关系
        # 动态调整后续步骤
```

---

### 4. 分解模式对比

| 维度 | Prompt Chaining | Dynamic Decomposition |
|------|----------------|----------------------|
| 步骤数量 | 预先确定 | 基于发现动态生成 |
| 适用任务 | 已知结构 | 未知复杂度 |
| 可预测性 | 高 | 低 |
| 灵活性 | 低 | 高 |
| 示例 | 代码审查 | 遗留代码测试 |

---

## 💡 Claude Code 提示词

```
实现两种任务分解模式对比演示：
1. Prompt Chaining：14文件 PR 审查
   - 每文件独立分析 pass
   - 跨文件集成 pass
   - 对比：单次全部分析 vs 分步分析的质量差异
2. Dynamic Decomposition：遗留代码库测试添加
   - 先映射结构
   - 识别高影响区域
   - 动态生成优先级计划

文件名：task_decomposition_demo.py
```

---

## 📋 例题

### 例题 1 ⭐⭐
一个 PR 包含 14 个文件，单次全量审查产生不一致结果（某些文件深度反馈，其他表浅；在一个文件标记的模式在另一文件被通过）。最佳重构方案？

**A)** 拆分为聚焦的 pass：每文件独立分析本地问题，然后单独的集成 pass 检查跨文件数据流

**B)** 要求开发者在自动审查运行前将大 PR 拆分为 3-4 个文件的提交

**C)** 切换到有更大上下文窗口的高级模型，以便一次给 14 个文件充足的注意力

**D)** 对完整 PR 运行三个独立 review pass，只标记至少两次出现的问题

> **答案：A** — 直接解决根本原因：注意力稀释。每文件分析确保深度一致，集成 pass 捕获跨文件问题。C 错误地认为更大上下文窗口能解决注意力质量问题。

---

### 例题 2 ⭐⭐
任务是"给遗留代码库添加全面测试"。最合适的分解方法？

**A)** Prompt chaining：先分析所有文件，然后生成所有测试，然后验证

**B)** 先映射结构、识别高影响区域，然后创建基于发现动态调整的优先级计划

**C)** 直接从最重要的文件开始写测试，不需要前期分析

**D)** 让模型一次性生成所有测试，然后人工筛选

> **答案：B** — 开放式任务需要动态分解，因为依赖关系和优先级只有在探索后才能知道。固定流水线（A）假设预先知道结构，不适合遗留代码库。

---

*文档版本：v1.0 | Domain 1 Task Statement 1.6*
