# Experiment Results Summary

## 1. Baseline vs SFT
| Model | Accuracy |
|-------|---------|
| Baseline (Qwen2.5-3B-Instruct) | 47.39% |
| SFT (LoRA, 1 epoch, 143787 samples) | 62.85% |
| Delta | +15.46% |

## 2. SFT vs DPO (MCQ Accuracy)
| Model | Accuracy |
|-------|---------|
| SFT | 62.85% |
| DPO v1 (lr=5e-5, 1 epoch) | 54.22% |
| DPO v2 (lr=1e-5, 2 epoch) | 53.38% |

## 3. SFT vs DPO (Alignment Metrics)
| Metric | SFT | DPO |
|--------|-----|-----|
| 格式遵循率 | 100% | 87% |
| 拒答/谨慎率 | 0% | 0% |
| 平均回答长度 | 415 字符 | 928 字符 |
| 重复输出率 | 3.5% | 0% |

## 4. RAG Ablation
| Method | V1 (PubMedQA only) | V2 (PubMedQA + MedMCQA) |
|--------|-------------------|------------------------|
| No RAG | 62.33% | 62.33% |
| BM25 | 52.00% | 64.67% |
| Dense | 55.67% | 60.67% |
| Rerank | 53.67% | 63.67% |

## 5. Full Pipeline Summary
| Method | Accuracy |
|--------|---------|
| Baseline | 47.39% |
| SFT | 62.85% |
| SFT + BM25 RAG | 64.67% |
| SFT + Rerank RAG | 63.67% |
| DPO | 53.38% |
