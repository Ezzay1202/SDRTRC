# notes.md — exp_002_v2.2_state_encoder

## 改动动机

v2.1 证明：单纯放开 trust region 不会改善性能（反而更差）。
残差分支在 gate 全开后输出的内容 ≈ 噪声，因为：
1. State Encoder 只有 6 个手工统计量，无法捕获真正的状态信息
2. dep_dim=16 太小，无法表示有意义的依赖偏移

v2.2 同时解决这两个瓶颈。

## 方案 B+C 改动

1. **重建 State Encoder** (方案B)：
   - Conv1D temporal encoder (2层, kernel=7,5) 从完整序列中学习时序模式
   - 保留原 6 个统计特征作为辅助输入
   - Dual-path fusion → state_dim=128

2. **增加容量** (方案C)：
   - dep_dim: 16 → 64
   - state_dim: 32 → 128

3. **Tuned trust region**：
   - trust_logit: -2.0 (sigmoid=0.119)

## 结果 vs baseline

```
Mean |diff|: v1=0.000655, v2.1=0.001302, v2.2=0.001281

Best version: all three versions tie on 27/27 tasks
```

v2.2 与 v1, v2.1 无法区分。Conv1D State Encoder + 4x 容量提升没有带来任何改善。

## 意外发现

**三版实验的核心教训：**

经过 v1 → v2.1 → v2.2 三轮迭代，我们测试了：
- gate 大小: 0.018 → ~1.0 → 0.119→~1.0 (learnable gate 总饱和到 ~1.0)
- State Encoder: 6 stats → 6 stats → Conv1D+stats dual-path
- 容量: dep=16/state=32 → 同 → dep=64/state=128

**结果：所有三个版本都无法稳定超越 XLinear。**

这说明 attention-based residual dependency learner 的架构本身存在根本性问题。无论怎么调参，残差分支学不到有用的 Δ_state。

gate 总是饱和到 ~1.0 的原因可能是因为：
- 训练 loss (MSE) 驱动 gate 扩大修正以降低训练误差
- 但残差分支不能准确区分信号和噪声，所以修正质量低
- 结果：训练 loss 下降了一点，但测试性能无改善或变差

## 下一步

需要更根本的架构变更，而非调参：

**方向1: 简化残差路径**
- 去掉 attention，直接用 state → residual (MLP)
- 去掉 risk gate，减少乘法链中的梯度衰减
- 考虑 additive correction 而非 multiplicative gating

**方向2: 换用 HyperNetwork**
- State → 生成 XLinear head 的权重调整
- 而非直接预测残差

**方向3: 换用 Mixture of Experts**
- 学习多个 residual "mode"
- State 决定各 mode 的混合权重

**方向4: 反思问题定义**
- 也许 Y = F_stable + Δ_state 的分解假设在 XLinear 级别上不成立
- XLinear 的 embedding 已经足够灵活，额外的残差分支是冗余的
- 可能需要完全不同的 backbone 配合残差学习
