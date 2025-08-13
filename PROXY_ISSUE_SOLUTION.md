# SGLang Server Proxy Issue with --host 0.0.0.0

## 问题描述

当使用 `--host 0.0.0.0` 启动 SGLang 服务器时，会出现以下错误：

```
[2025-08-13 11:03:03] Initialization failed. warmup error: 
AssertionError: res=<Response [403]>, res.text='<!DOCTYPE html...ERROR: The requested URL could not be retrieved...Access Denied...'
```

服务器本身启动成功，但在 warmup（预热）阶段失败，收到来自 Squid 代理服务器（172.16.37.200:3138）的 403 Access Denied 错误。

## 根本原因分析

### 问题流程
1. SGLang 服务器使用 `--host 0.0.0.0` 参数启动，监听所有网络接口
2. 服务器启动后，执行 `_execute_server_warmup()` 函数进行预热
3. 预热函数直接使用 `--host` 参数构建 URL：`http://0.0.0.0:25600/get_model_info`
4. Python 的 requests 库发送请求到 `0.0.0.0` 地址
5. 系统配置的 HTTP 代理（Squid）拦截了这个请求
6. 企业代理出于安全考虑，禁止访问 `0.0.0.0` 地址，返回 403 错误

### 为什么 127.0.0.1 能正常工作？
- `127.0.0.1` 是本地回环地址
- 大多数代理配置会自动跳过 localhost 地址
- 请求直接到达本地服务，不经过代理

## 解决方案

### 方案 1：设置环境变量跳过代理（推荐）

最简单的解决方案是设置 `NO_PROXY` 环境变量：

```bash
# 方式 1：导出环境变量
export NO_PROXY=0.0.0.0,localhost,127.0.0.1
export no_proxy=0.0.0.0,localhost,127.0.0.1

# 方式 2：在命令前添加环境变量
NO_PROXY=0.0.0.0,localhost,127.0.0.1 \
no_proxy=0.0.0.0,localhost,127.0.0.1 \
$PYTHONHOME/bin/python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 \
    --port 3000 \
    --disable-cuda-graph \
    --trust-remote-code \
    --context-length 40960 \
    --tp 8 \
    --watchdog-timeout 600 \
    --log-level debug \
    2>&1 >> $SCRIPT_DIR/sglang.log &
```

### 方案 2：修改 SGLang 源代码（永久修复）

已经应用的补丁修改了 `/workspace/python/sglang/srt/entrypoints/http_server.py` 文件：

```python
def _execute_server_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection],
):
    headers = {}
    # Use localhost for warmup when binding to 0.0.0.0 to avoid proxy issues
    if server_args.host == "0.0.0.0":
        url = f"http://127.0.0.1:{server_args.port}"
    else:
        url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"
```

这个修改确保即使服务器绑定到 `0.0.0.0`，预热请求也会使用 `127.0.0.1` 以避免代理问题。

### 方案 3：使用包装脚本

创建了 `/workspace/sglang_server_wrapper.sh` 脚本，自动处理代理设置：

```bash
#!/bin/bash
# 使用包装脚本启动服务器
/workspace/sglang_server_wrapper.sh \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 \
    --port 3000 \
    --disable-cuda-graph \
    --trust-remote-code \
    --context-length 40960 \
    --tp 8 \
    --watchdog-timeout 600 \
    --log-level debug \
    2>&1 >> $SCRIPT_DIR/sglang.log &
```

### 方案 4：跳过预热（不推荐）

添加 `--skip-server-warmup` 参数：

```bash
$PYTHONHOME/bin/python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 \
    --port 3000 \
    --skip-server-warmup \  # 添加此参数
    --disable-cuda-graph \
    --trust-remote-code \
    --context-length 40960 \
    --tp 8 \
    --watchdog-timeout 600 \
    --log-level debug \
    2>&1 >> $SCRIPT_DIR/sglang.log &
```

**注意**：跳过预热可能会影响服务器的初始性能。

### 方案 5：使用具体 IP 地址

使用容器的实际 IP 地址而不是 `0.0.0.0`：

```bash
# 获取容器 IP
CONTAINER_IP=$(hostname -I | awk '{print $1}')

# 使用具体 IP 启动
$PYTHONHOME/bin/python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host $CONTAINER_IP \
    --port 3000 \
    # ... 其他参数
```

## 已创建的文件

1. **`/workspace/fix_proxy_issue.py`** - 分析脚本，提供详细的问题分析和解决方案
2. **`/workspace/sglang_warmup_fix.patch`** - 源代码补丁文件
3. **`/workspace/sglang_server_wrapper.sh`** - 自动设置环境变量的包装脚本
4. **源代码已修改** - `python/sglang/srt/entrypoints/http_server.py` 已应用修复

## 建议

1. **短期解决**：使用方案 1（环境变量）或方案 3（包装脚本）
2. **长期解决**：源代码已修改（方案 2），下次重启后生效
3. **最佳实践**：在 Docker 容器中，建议在 Dockerfile 或启动脚本中设置 `NO_PROXY` 环境变量

## 验证修复

修复后，服务器应该能够成功完成预热并正常启动：

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:25600
# 不再出现 403 错误
```