# notes.md — exp_001_v2.1_trust_region

## 改动动机

v1 实验结果显示 SDRTR 与 XLinear 几乎无差异（平均 |MSE Diff| = 0.000646）。
根本原因假设：trust_logit=-4.0 使 gate≈0.018，模型被锁死在 XLinear 附近。

## 方案A 改动

1. sdr_trust_logit: -4.0 → -1.0 (gate 从 0.018 → 0.269, ~15x)
2. sdr_residual_scale: 0.08 → 0.15 (~2x)
3. sdr_use_learnable_gate: 0 → 1 (gate 可学习, 自适应各 horizon group)

总效果: 最大修正量从 ~0.0014 → ~0.040 (~29x)

## 结果 vs baseline

```
Mean |v1 Diff|  = 0.000655
Mean |v2.1 Diff| = 0.001302   ← 翻倍了！
Total |diff| reduction: -0.017461  ← 负值 = v2.1 更差

Winners: v2.1=7, v1=12, tie=8
```

### Per-dataset avg diff vs XLinear:
| Dataset | v1 avg diff | v2.1 avg diff | Change |
|---------|-------------|---------------|--------|
| ETTh1 | -0.000019 | -0.000739 | +0.000720 |
| ETTh2 | -0.000087 | +0.000541 | -0.000628 (更差) |
| ETTm1 | -0.000359 | +0.000414 | -0.000774 (更差) |
| ETTm2 | -0.000335 | -0.000366 | +0.000031 |
| weather | -0.000254 | -0.000038 | -0.000216 (更差) |
| electricity | -0.001195 | -0.000637 | -0.000559 (更差) |
| traffic | +0.000359 | +0.001415 | -0.001056 (更差) |

### 关键观察:
1. learnable gate 收敛到 ~0.995（几乎全开），correction_abs_mean 从 ~0 增大到 ~0.04
2. 但增大修正量后，结果是**变差**而非变好
3. 残差分支在 gate 全开后输出的内容 ≈ 噪声，而非有意义的 Δ_state

## 意外发现

**这条路线走不通。** 单纯放开 trust region 让残差分支自由表达，不会改善预测。原因：
- State Encoder 只有 6 个手工统计量 (mean/std/trend/last/half_shift/range)，根本捕获不了"状态"
- dep_dim=16, state_dim=32 容量太小，学不到有意义的依赖偏移
- 残差分支在容量不足 + 状态信息贫乏时，只能学到噪声

## 下一步

❌ 方案 A 单独不够 → 需要 **v3: 重建 State Encoder (方案B) + 增加容量 (方案C)**

具体方向:
1. State Encoder: 用 Conv1D 或轻量 Transformer 编码完整时序，而非 6 个手工特征
2. 增加 dep_dim (16→64) 和 state_dim (32→128)
3. 可能考虑让 trust_logit 回到较小值 (-2.0)，因为大修正量在当前架构下只会引入噪声

参见: [[exp_001_v2.1_trust_region]]
