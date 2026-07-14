---
name: sdrtrc-experiment-results
description: SDRTRC v1/v2.1/v2.2 三轮实验完整结果、架构诊断与 v3 方向建议
metadata: 
  node_type: memory
  type: project
  last_updated: 2026-07-06
  originSessionId: 77a0d8a2-2712-4eda-91ca-a15cc9b4723a
---

## SDRTRC 实验全景 (截至 2026-07-06)

### 核心结论

**三轮实验（v1 → v2.1 → v2.2）均无法稳定超越 XLinear。Attention-based residual dependency learner 架构需要根本性反思。**

```
Mean |diff| vs XLinear:  v1=0.000655  |  v2.1=0.001302  |  v2.2=0.001281
Win/Loss/Tie (|diff|>0.0005): v1=8W/7L/12T | v2.1=5W/10L/12T | v2.2=5W/12L/10T
27/27 任务三版统计无差异
```

### 三轮演进

| 维度 | v1 (baseline) | v2.1 (trust region) | v2.2 (state encoder) |
|---|---|---|---|
| **State Encoder** | 6 stats → MLP | 同 v1 | Conv1D(k=7,5) + stats dual-path |
| **dep_dim** | 16 | 16 | **64** (4x) |
| **state_dim** | 32 | 32 | **128** (4x) |
| **trust_logit** | -4.0 (σ≈0.018) | -1.0 (σ≈0.269) | -2.0 (σ≈0.119) |
| **residual_scale** | 0.08 | 0.15 | 0.15 |
| **learnable_gate** | 禁用 | 启用 → 饱和~0.995 | 启用 → 饱和 0.119→0.999 |
| **额外参数** | ~75K | ~75K | ~300K |
| **Mean \|diff\|** | 0.000655 | 0.001302 | 0.001281 |
| **结论** | 与 XLinear 无差异 | **更差** | **更差** |

### 关键发现

1. **Gate 总是饱和到 ~1.0**：训练 loss (MSE) 驱动 gate 扩大修正以降低训练误差，但残差分支不能准确区分信号和噪声
2. **State Encoder 质量不是瓶颈**：Conv1D + 4x 容量提升后结果不变，说明不是"状态信息不足"
3. **容量不是瓶颈**：dep_dim 16→64, state_dim 32→128 无改善
4. **根本问题在架构**：attention 机制对残差预测无效——无论 gate 多大、encoder 多强、容量多大，Δ_state 始终是噪声

### 实验文件位置

- **v2.1**: `experiments/exp_001_v2.1_trust_region/{config.yaml, results.csv, notes.md}`
- **v2.2**: `experiments/exp_002_v2.2_state_encoder/{config.yaml, results.csv, notes.md}`
- **三版对比 xlsx**: `/Users/ezzay/Desktop/论文文件夹/sdrtr_v2.1_v2.2_结果对比.xlsx`
- **v1 原始 xlsx**: `/Users/ezzay/Desktop/论文文件夹/sdrtr_v1.xlsx`

### 建议的 v3 方向

1. **简化残差路径**：去掉 attention，直接用 State → MLP → residual (additive correction)
2. **HyperNetwork**：State 生成 XLinear head 权重调整，而非直接预测残差
3. **Mixture of Experts**：多个 residual mode，State 决定混合权重
4. **反思假设**：Y = F_stable + Δ_state 的分解在 XLinear 级别上可能不成立；XLinear embedding 已足够灵活

**Why:** 三轮实验测试了 gate 大小、encoder 质量、模型容量三个维度，无一改善 → 问题不在调参，在架构设计
**How to apply:** 新对话开始时读此文件即可了解完整实验历史；v3 实施应从上述 4 个方向中选择一个做根本性重构
