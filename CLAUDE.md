# SDRTRC — State-Dependent Residual Learning with Trust-Region Correction

时序预测论文项目。核心模型 SDRTR vs 基线 XLinear。代码通过 Git 管理，在 jx GPU 服务器 Docker 容器内运行。

## 当前状态 (2026-07-14)

**v1/v2.1/v2.2 三轮实验均无法超越 XLinear。** Attention-based residual dependency learner 架构需要根本性重构。
容器已重建 (旧容器 `6db5e70ff0fd` → 新容器 `b795fb0848ab`)，Git 仓库已推送 GitHub。

详见 `.claude/memory/sdrtrc-experiment-results.md`。

## 快速链接

- **GitHub**: https://github.com/Ezzay1202/SDRTRC (主项目), https://github.com/Ezzay1202/MUSE (基线)
- 项目 memory: `.claude/memory/` (仅在本项目目录下加载)
- Skill: `/sdrtrc` — 跑实验、管理 pipeline、提取结果
- 服务器: jx (10.1.19.41), Docker 容器 `b795fb0848ab` (HogPricePrediction), 端口 31101
- 实验结果: `/Users/ezzay/Desktop/论文文件夹/sdrtr_v2.1_v2.2_结果对比.xlsx`

## 关键文件

| 文件 | 作用 |
|---|---|
| `models/SDRTR.py` | 核心模型 (XLinear + HorizonStateDependencyBranch) |
| `run_longExp.py` | 训练入口，含所有 SDRTR 超参默认值 |
| `tools/sdrtrc_multi_runner.py` | 多数据集串行实验 runner |
| `experiments/` | 实验记录 (config.yaml + results.csv + notes.md) |

## Git 工作流（GitHub SSH）

```
本地 Mac                       GitHub                        jx 服务器容器
  │                              │                              │
  ├─ git add/commit              │                              │
  ├─ git push ──────────────►  Ezzay1202/SDRTRC                │
  │                              │                              │
  │                              ├── git clone/pull ───────► /workspace/SDRTRC-main/
  │                              │                              │
  │  git push                    │                      git pull (容器内更新)
```

### 工作流步骤

```bash
# 1. 本地开分支
git checkout main && git pull
git checkout -b exp/v<版本号>-<简述>

# 2. 改代码 + 小步提交
git commit -m "feat: ..."    # 新功能
git commit -m "fix: ..."     # 修bug
git commit -m "tune: ..."    # 调参
git commit -m "exp: ..."     # 实验配置
git commit -m "refactor: ..." # 重构

# 3. 推送到 GitHub (SSH)
git push -u origin exp/v<版本号>-<简述>

# 4. 在容器里拉取
ssh root@10.1.19.41 "docker exec b795fb0848ab bash -c 'cd /workspace/SDRTRC-main && git pull'"

# 5. 跑实验（详见 .claude/memory/sdrtrc-project-workflow.md）

# 6. 实验完成后合并
git checkout main && git merge exp/vX-xxx
git tag vX.Y-label -m "描述"
git push origin main --tags
```

### Commit 前缀约定

| 前缀 | 用途 |
|------|------|
| `feat:` | 新功能 |
| `fix:` | 修bug |
| `tune:` | 调参 |
| `exp:` | 实验配置 |
| `refactor:` | 重构 |

### 关键约束

- **铁律**: `main` 始终可运行，只通过 merge 进入
- 失败的实验分支保留不合并，notes.md 写清楚原因
- **GitHub HTTPS 被墙** → 只用 SSH (`git@github.com:Ezzay1202/...`)
- MUSE 是只读基线，不改代码；SDRTRC 是主项目

## Agent 规则（强制性）

**每个新 agent / subagent 必须首先读取以下文件：**
1. 本文件 (`CLAUDE.md`)
2. `.claude/memory/sdrtrc-experiment-results.md` — 实验历史 + 模型问题诊断 + v3 方向
3. `.claude/memory/sdrtrc-project-workflow.md` — Git 工作流 + PPU 选择 + 容器内操作
4. `.claude/memory/jx-server-default-container.md` — 服务器连接信息

**不读这些文件就开始工作的 agent 会产生幻觉**（5 类典型幻觉：虚构目录、虚构文件、误报 Git 状态、误报错误、误报脚本 bug）。

启动 agent 时在 prompt 中明确要求先 Read 上述 4 个文件。
