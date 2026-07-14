---
name: jx-server-default-container
description: /server jx 的默认工作路径是 HogPricePrediction Docker 容器
metadata: 
  node_type: memory
  type: reference
  originSessionId: 8c3b67d4-2be6-4975-9fb1-1a122392d4c8
---

## /server jx 默认路径

当用户调用 `/server jx` 时，所有操作默认在以下 Docker 容器内执行：

- **容器 ID**: `6db5e70ff0fd`
- **容器名称**: `HogPricePrediction`
- **宿主机**: root@10.1.19.41:22 (密码: Alibaba%1688)

所有命令通过 `docker exec 6db5e70ff0fd <command>` 在容器内执行。

关键路径：
- MUSE-main: `/workspace/MUSE-main/`
- 数据集: `/workspace/MUSE-main/dataset/`
- 脚本: `/workspace/MUSE-main/script/`
- 日志: `/workspace/MUSE-main/log/`
- SDRTRC: `/workspace/SDRTRC-main/`

参见 [[ssh-server-credentials]] 和 [[sdrtrc-project-workflow]]。
