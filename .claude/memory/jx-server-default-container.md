---
name: jx-server-default-container
description: /server jx 的默认工作路径是 HogPricePrediction Docker 容器
metadata: 
  node_type: memory
  type: reference
  originSessionId: 8c3b67d4-2be6-4975-9fb1-1a122392d4c8
  last_updated: "2026-07-14"
---

## /server jx 默认路径

当用户调用 `/server jx` 时，所有操作默认在以下 Docker 容器内执行：

### 当前容器（2026-07-14 重建）

- **容器 ID**: `b795fb0848ab`
- **容器名称**: `HogPricePrediction`
- **宿主机**: root@10.1.19.41:22 (密码: Alibaba%1688)
- **端口映射**: 宿主机 `31101` → 容器 `8000`
- **镜像**: `6ced546aad63`

> 旧容器 `6db5e70ff0fd` 已删除（2026-07-14）。端口 31100 被 JxPigMarketingDecision 占用，故新容器用 31101。

### 容器操作

```bash
# SSH 到 jx 服务器
ssh -p 22 root@10.1.19.41  # 密码: Alibaba%1688

# 进入容器
docker exec -it b795fb0848ab bash

# 执行单条命令
docker exec b795fb0848ab <command>

# 执行脚本
docker exec b795fb0848ab bash -c '<script>'
```

### 关键路径

| 项目 | 容器内路径 |
|------|-----------|
| SDRTRC 项目 | `/workspace/SDRTRC-main/` |
| MUSE 基线 | `/workspace/MUSE-main/` |
| 数据集 | `/workspace/MUSE-main/dataset/` |
| 脚本 | `/workspace/MUSE-main/script/` |
| 日志 | `/workspace/SDRTRC-main/logs/` |

### 代码同步（Git-based）

```bash
# 容器内 git pull（从 GitHub SSH）
docker exec b795fb0848ab bash -c 'cd /workspace/SDRTRC-main && git pull'
docker exec b795fb0848ab bash -c 'cd /workspace/MUSE-main && git pull'
```

### PPU 检查

```bash
ssh root@10.1.19.41 "ppu-smi"                           # 查看全部 PPU
ssh root@10.1.19.41 "ppu-smi | grep -E 'PPU|Used'"      # 快速看空闲
```

### 同宿主机其他容器

| 容器 | 端口 | 用途 |
|------|------|------|
| `JxPigMarketingDecision` | 31100 | 猪价营销决策 |
| `JxPigPriceForecast` | - | 猪价预测（另一版本） |
| `JxPigPriceForecast-claude` | - | 猪价预测 Claude 版 |
| `meta-llama-3-8b-instruct-*` | - | LLM 推理 |
| `qwen3.6-35b-a3b-*` | - | LLM 推理 |

参见 [[ssh-server-credentials]] 和 [[sdrtrc-project-workflow]]。
