# Baseline vs SFT Evaluation

## Model
- Base: Qwen2.5-3B-Instruct
- SFT: Qwen2.5-3B-Instruct + LoRA (r=16, MedMCQA + PubMedQA)

## Dataset
- Eval: MedMCQA subset (498 samples)

## Results

| Model    | Accuracy | Delta  |
|----------|----------|--------|
| Baseline | 47.39%   | -      |
| SFT      | 62.85%   | +15.46% |

## Notes
- no_answer=0，格式遵循率 100%
- 1 epoch，143787 条训练数据
- eval loss: 1.46 -> 1.32
