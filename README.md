# Medical LLM Alignment & RAG
Medical QA system with Qwen2.5-3B LoRA SFT, DPO alignment, and RAG enhancement.

> 基于 Qwen2.5-3B 的医疗问答系统，涵盖 LoRA SFT 微调、DPO 偏好对齐与 RAG 检索增强，并对各模块进行了系统性 ablation 实验。

## 📊 实验结果

| 方法 | MedMCQA Accuracy |
|------|-----------------|
| Baseline (Qwen2.5-3B-Instruct) | 47.39% |
| + LoRA SFT | 62.85% |
| + SFT + BM25 RAG | **64.67%** |
| + SFT + Rerank RAG | 63.67% |
| + DPO | 53.38% |

### DPO 对齐指标

| 指标 | SFT | DPO |
|------|-----|-----|
| 格式遵循率 | 100% | 87% |
| 平均回答长度 | 415 字符 | 928 字符 |
| 重复输出率 | 3.5% | 0% |

> DPO 后 MCQ accuracy 略有下降，但回答更详细（长度翻倍），幻觉重复率从 3.5% 降至 0%，符合 DPO 对齐而非准确率优化的预期。

---

## 📁 项目结构

    medical-llm-alignment/
    ├── scripts/
    │   ├── build_sft_data.py       # SFT 数据构造
    │   ├── build_dpo_data.py       # DPO 偏好数据构造（基于 badcase 自动生成）
    │   └── build_rag_corpus.py     # RAG 文档库构建
    ├── rag/
    │   ├── retrieve.py             # BM25 + Dense 双路检索
    │   ├── rerank.py               # bge-reranker CrossEncoder 重排
    │   └── query_rewrite.py        # 基于 SFT 模型的 Query 改写
    ├── reports/
    │   ├── sft_vs_base.md          # Baseline vs SFT 报告
    │   ├── dpo_vs_sft.md           # SFT vs DPO 对齐报告
    │   └── rag_ablation.md         # RAG ablation 报告
    ├── train_sft.py                # SFT 训练
    ├── train_dpo.py                # DPO 训练
    ├── eval_mcq.py                 # MCQ 准确率评估
    ├── eval_alignment.py           # 对齐指标评估
    └── eval_rag.py                 # RAG ablation 评估

---

## 📦 数据

| 数据集 | 原始数量 | 清洗后 | 用途 |
|--------|---------|--------|------|
| MedMCQA | 182,822 | 158,764 | SFT 训练 |
| PubMedQA | 1,000 | 1,000 | SFT 训练 |
| DPO 偏好数据 | - | 905 | DPO 训练 |
| RAG 文档库 | - | 11,494 | 检索增强 |

- **SFT 数据**：MedMCQA + PubMedQA 统一格式化为 system/user/assistant 对话格式，共 143,787 条
- **DPO 数据**：从 3,000 条评估样本中提取 1,006 个 badcase，自动构造 chosen/rejected 对
- **RAG 文档库**：PubMedQA abstracts + MedMCQA explanations 混合构建

---

## 🔧 方法

### 1️⃣ SFT 监督微调

- **模型**：Qwen2.5-3B-Instruct + LoRA（r=16, alpha=32，覆盖全部 attention + FFN 层）
- **数据**：143,787 条，1 epoch，lr=2e-4，effective batch size=16
- **结果**：Baseline 47.39% → SFT 62.85%，提升 **+15.46%**

### 2️⃣ DPO 偏好对齐

- **基础**：在 SFT checkpoint 上继续训练
- **数据构造**：运行模型推理 → 提取 badcase → 以数据集标准答案为 chosen，模型错误输出为 rejected
- **训练**：905 条偏好对，2 epoch，lr=1e-5
- **效果**：MCQ accuracy 略降，但回答长度翻倍，幻觉重复率从 3.5% 降至 0%

### 3️⃣ RAG 检索增强

- **文档库**：PubMedQA + MedMCQA，共 11,494 条
- **双路检索**：BM25（关键词匹配）+ bge-m3（语义向量）各召回 5 条，合并去重
- **重排**：bge-reranker-base CrossEncoder 精排，取 top-3
- **Query 改写**：SFT 模型将用户问题改写为检索友好格式
- **效果**：SFT 62.85% → SFT + BM25 RAG **64.67%**

---

## 🚀 复现步骤

```bash
# 安装依赖
pip install transformers peft trl datasets accelerate \
            rank_bm25 faiss-cpu sentence-transformers

# 数据准备
python scripts/build_sft_data.py
python scripts/build_dpo_data.py
python scripts/build_rag_corpus.py

# 训练
python train_sft.py
python train_dpo.py

# 评估
python eval_mcq.py
python eval_alignment.py
python eval_rag.py
```

> ⚠️ 国内服务器需设置 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 🖥️ 环境

| 项目 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.8.0 |
| CUDA | 12.8 |
| Transformers | 5.9.0 |
| PEFT | 0.19.1 |
| TRL | 1.4.0 |
