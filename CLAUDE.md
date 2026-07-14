# SDRTRC — State-Dependent Residual Learning with Trust-Region Correction

时序预测论文项目。核心模型 SDRTR vs 基线 XLinear。代码在 jx GPU 服务器 Docker 容器内运行。

## 当前状态 (2026-07-06)

**v1/v2.1/v2.2 三轮实验均无法超越 XLinear。** Attention-based residual dependency learner 架构需要根本性重构。

详见 `.claude/memory/sdrtrc-experiment-results.md`。

## 快速链接

- 项目 memory: `.claude/memory/` (仅在本项目目录下加载)
- Skill: `/sdrtrc` — 跑实验、管理 pipeline、提取结果
- 服务器: jx (10.1.19.41), Docker 容器 `6db5e70ff0fd`
- 实验结果: `/Users/ezzay/Desktop/论文文件夹/sdrtr_v2.1_v2.2_结果对比.xlsx`

## 关键文件

| 文件 | 作用 |
|---|---|
| `models/SDRTR.py` | 核心模型 (XLinear + HorizonStateDependencyBranch) |
| `run_longExp.py` | 训练入口，含所有 SDRTR 超参默认值 |
| `tools/sdrtrc_multi_runner.py` | 多数据集串行实验 runner |
| `experiments/` | 实验记录 (config.yaml + results.csv + notes.md) |

## Agent 规则（强制性）

**每个新 agent / subagent 必须首先读取以下文件：**
1. 本文件 (`CLAUDE.md`)
2. `.claude/memory/sdrtrc-experiment-results.md` — 实验历史 + 模型问题诊断 + v3 方向
3. `.claude/memory/sdrtrc-project-workflow.md` — Git 工作流 + PPU 选择 + 代码同步
4. `.claude/memory/jx-server-default-container.md` — 服务器连接信息

**不读这些文件就开始工作的 agent 会产生幻觉**（见远程 CLAUDE.md 的 Known Pitfalls 章节 — 5 类典型幻觉：虚构目录、虚构文件、误报 Git 状态、误报错误、误报脚本 bug）。

启动 agent 时在 prompt 中明确要求先 Read 上述 4 个文件。
