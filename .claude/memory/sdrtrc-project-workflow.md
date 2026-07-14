---
name: sdrtrc-project-workflow
description: SDRTRC 论文项目工作流：Git 版本管理、远程 GPU 训练、Excel 结果分析的规范流程
metadata: 
  node_type: memory
  type: project
  originSessionId: b4dcde5e-5b35-4492-94c6-425c84be8ac9
  last_updated: "2026-07-06"
---

> **实验历史**: 详见 [[sdrtrc-experiment-results]] — v1/v2.1/v2.2 三轮实验结果、架构诊断、v3 方向建议

## SDRTRC 项目概览

这是一个时序预测论文项目，核心模型是 **SDRTR (State-space Dual-path Temporal-spatial Reconciled Transformer)**，对标 XLinear。代码在远程 GPU 服务器 (jx: 10.1.19.41) 的 Docker 容器内运行，结果分析在本地 Mac 上做。

### 关键资源

- **代码**: jx 服务器 Docker 容器 `6db5e70ff0fd` 内 `/workspace/SDRTRC-main/`
- **数据集**: `/workspace/MUSE-main/dataset/` (ETTh1/2, ETTm1/2, weather, electricity, traffic, hogprice)
- **结果 Excel**: `/Users/ezzay/Desktop/论文文件夹/experiments_results.xlsx`
- **服务器连接**: [[ssh-server-credentials]] 中 jx 部分

---

## Git 工作流 (5 步循环)

```
① 开分支 → ② 改代码+小步提交 → ③ 跑实验 → ④ 记录结果 → ⑤ 合并+打tag
   git         改模型参数           pipeline     experiments/   git merge
   checkout     git commit          GPU 训练     写 config.yaml  git tag
   -b exp/v*                                   results.csv    vX.Y-label
                                               notes.md
```

### ① 开实验分支

```bash
git checkout main
git checkout -b exp/v<版本号>-<改动简述>
# 例: exp/v2-add-attention, exp/v3-huber-loss
```

### ② 改代码 + 小步提交

```bash
# 每次一个逻辑改动
git commit -m "feat: add cross-attention to encoder"
git commit -m "tune: d_model 2048 -> 4096"
```

Commit 前缀约定：`feat:` 新功能 | `fix:` 修bug | `tune:` 调参 | `exp:` 实验配置 | `refactor:` 重构

### ③ 跑实验

有两种训练模式：
- **Short pred_len (12/24/48)**: `bash script/run_short_pipeline.sh` → 5 步 pipeline (train_xlinear + train_sdrtr 并行 → export → check → collect)
- **Long pred_len (96/192/336/720)**: 用 `script/sdrtrc_multi_forcatsing/` 下的独立脚本

关键注意：shell 脚本的 `ROOT_PATH` 需指向 `/workspace/MUSE-main/dataset`，Python runner 的默认 `--root_path` 已修复为此路径。

### ④ 记录实验结果

每次实验在 `experiments/exp_NNN_vX_description/` 下创建 3 个文件：

```yaml
# config.yaml - 超参快照
model_version: v2.0
date: 2026-07-02
changes: "描述改了什么"
hparams:
  d_model: 4096
  learning_rate: 0.0002
```

```csv
# results.csv - 指标汇总 (从 pipeline log 提取)
dataset,pred_len,xl_mse,xl_mae,sd_mse,sd_mae,mse_diff,winner
```

```markdown
# notes.md - 实验结论
## 改动动机
## 结果 vs baseline
## 意外发现
## 下一步
```

### ⑤ 合入主分支

```bash
git checkout main
git merge exp/v2-xxx
git tag v2.0-label -m "描述"
git branch -d exp/v2-xxx  # 删除已合并分支
```

**失败的实验**：分支保留不合并，notes.md 里写清楚为什么失败。

### 分支全景

```
main ──●────●────●──→
       v1.0 v2.0 v3.0
        │    │
        └── exp/v2-attention (合并后删除)
             └── exp/v2-attention-v2 (失败，保留不删)
```

**铁律**: `main` 始终可运行，不改代码只通过 merge 进入。

---

## 实验执行规范

### 连接 jx 服务器

```expect
spawn ssh -o StrictHostKeyChecking=no -p 22 root@10.1.19.41
expect "password:"; send "Alibaba%1688\r"
```

### 容器操作

```bash
docker exec 6db5e70ff0fd <command>           # 执行命令
docker exec -it 6db5e70ff0fd bash            # 交互式
docker exec 6db5e70ff0fd bash -c '<script>'  # 脚本
```

### 跑 Short Pipeline

```bash
docker exec -d 6db5e70ff0fd bash -c \
  'nohup /workspace/SDRTRC-main/run_pipeline.sh \
   > /workspace/SDRTRC-main/logs/pipeline_master.log 2>&1 &'
```

日志位置：
- `logs/pipeline_master.log` — 总进度
- `logs/pipeline_step01.log` — XLinear 训练
- `logs/pipeline_step02.log` — SDRTR 训练
- `logs/pipeline_step03~05.log` — export/check/collect

### 监控

```bash
docker exec 6db5e70ff0fd cat logs/pipeline_master.log
docker exec 6db5e70ff0fd bash -c 'grep -c ">>>testing" logs/pipeline_step01.log'
```

### 提取结果填 Excel

从 log 中提取 test metrics（每条 `testing` 标记后的**第一个** `mse:` 行是 test，第二个是 val）：
```bash
docker exec 6db5e70ff0fd bash -c 'grep -E "mse:|>>>>>>testing" logs/pipeline_step01.log'
```

---

## 关键约束

1. **jx 安全规则**: 任何操作前必须用 AskUserQuestion 确认，每个独立操作单独确认
2. **数据路径**: 永远是 `/workspace/MUSE-main/dataset/`，不是 `/Users/ezzay/...` 或 `/home/data/zsh/...`
3. **Shell 脚本转义**: 通过 Docker exec 写文件时，用 base64 编码避免多层转义问题
4. **Docker overlay 路径**: 直接修改容器内文件可通过 `/bmcp_lvm_fs/docker-data/overlay2/<container_id>/diff/workspace/SDRTRC-main/`

---

## Long Pred_len 实验（96/192/336/720）

除了 short pipeline，还有 long pred_len 多数据集实验。使用 `tools/sdrtrc_multi_runner.py` 在同一 PPU 上串行跑 8 个数据集 × 3-4 个 pred_len。

### 多数据集 Runner 用法

```bash
docker exec <容器ID> bash -c \
  'cd /workspace/SDRTRC-main && nohup python -u tools/sdrtrc_multi_runner.py \
   --gpu 0 --epochs 30 --batch_size 32 --lr 0.0002 \
   --sdr_state_dim 128 --sdr_dep_dim 64 --sdr_trust_logit -2.0 \
   --sdr_use_learnable_gate 1 --sdr_residual_scale 0.15 \
   > logs/multi_v2.2.log 2>&1 &'
```

### PPU 检查（阿里云 PPU，非 NVIDIA GPU）

```bash
ssh root@10.1.19.41 "ppu-smi"     # 查看 PPU 占用
ssh root@10.1.19.41 "ppu-smi | grep -E 'PPU|Used'"  # 快速看空闲 PPU
```

优先选用无进程或显存 < 2GB 的 PPU。

### 代码同步流程

```bash
# 本地打包 → scp 到服务器 → docker cp 到容器 → 解压
tar czf /tmp/sdrtrc_code.tar.gz -C /Users/ezzay/PycharmProjects/SDRTRC-main .
scp /tmp/sdrtrc_code.tar.gz root@10.1.19.41:/tmp/
ssh root@10.1.19.41 "docker cp /tmp/sdrtrc_code.tar.gz <容器ID>:/tmp/ && \
  docker exec <容器ID> bash -c 'cd /workspace/SDRTRC-main && tar xzf /tmp/sdrtrc_code.tar.gz'"
```
