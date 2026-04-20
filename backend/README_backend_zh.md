# 后端总览

[中文](../README_zh.md) | [English](./README_backend.md)

这个后端由三个核心文件组成：

- [run_backend.sh](./run_backend.sh)：启动 worker 和 API 服务
- [backend_api.py](./backend_api.py)：面向前端的调度层、队列和任务状态管理
- [backend_worker.py](./backend_worker.py)：单卡推理 worker

整体是一个两层结构：

- API 进程不直接做 GPU 推理，只负责接收请求、创建任务、分配空闲 worker，并向前端返回状态。
- 每个 worker 进程绑定一张 GPU，负责加载模型、执行 backbone 生成、执行超分、解码音频，并把生成文件提供给 API 拉取。

## 请求流程

一次前端请求会经过下面这条链路：

1. 前端向 `backend_api.py` 的 `POST /generate` 发送请求。
2. API 清洗歌词、标准化请求参数、创建 job，并判断是立即派发还是进入队列。
3. API 通过 `POST /generate` 把 worker payload 发送给一个或多个 worker。
4. `backend_worker.py` 加载或复用 tokenizer 和模型，然后依次执行：
   - prompt 准备
   - backbone 生成
   - super-resolution
   - decoder 波形重建
   - MP3 导出
5. worker 把输出文件写到 `backend/generated_audio/`，并返回文件名和元数据。
6. API 从 worker 拉取生成文件，把结果挂到对应 job 上，并通过 `/job/{job_id}` 和 `/job/{job_id}/track/{track_idx}/mp3|wav` 对前端暴露。

## 文件职责

### `run_backend.sh`

这是本地或容器环境下的启动入口。

它负责：

- 定义用户可修改的启动配置
- 按 `GPU_IDS` 启动一个或多个 worker
- 等待 worker 健康检查通过
- 启动 API 调度层
- 在前台 tail 日志

最重要的配置都在文件顶部：

- `GPU_IDS`：指定哪些物理 GPU 用来跑 worker
- `NUM_WORKERS`：由 `GPU_IDS` 自动推导
- `API_PORT`
- `WORKER_BASE_PORT`
- `BASE_MASTER_PORT`
- `BASE_SEED`
- `MEGATRON_ARGS`

### `backend_api.py`

这是面向前端的编排层。

它负责：

- 接收生成请求
- 清洗歌词和标准化请求字段
- 组装 worker payload
- 创建并维护 job
- 在所有 worker 忙碌时排队
- 轮询 worker 健康状态并同步任务进度
- 返回 job 状态和生成音频文件

它本身不执行 GPU 推理。

### `backend_worker.py`

这是单卡推理运行时。

它负责：

- 加载 tokenizer
- 加载 Megatron backbone
- 加载 super-resolution 模型
- 加载 DAC RVQ decoder
- 准备 prompts
- 生成 q0/q1 backbone tokens
- 扩展到 q0..q63
- 解码波形音频
- 导出 WAV 和 MP3

当前的运行策略是：

- tokenizer 常驻
- backbone 常驻
- superres 按请求加载并在请求结束后释放
- decoder 按请求加载并在请求结束后释放

## 如何启动

在 backend 目录下执行：

```bash
cd backend
bash run_backend.sh
```

停止所有后端进程：

```bash
bash run_backend.sh stop
```

## 日志

日志输出在：

- `backend/logs/api.log`
- `backend/logs/worker_0.log`
- `backend/logs/worker_1.log`
- ...

常用查看方式：

```bash
tail -f backend/logs/api.log
tail -f backend/logs/worker_0.log
```

## 输出文件

生成结果会写到：

- `backend/generated_audio/*.wav`
- `backend/generated_audio/*.mp3`
- `backend/generated_audio/*.json`

每次请求通常会写出：

- 一个 WAV 文件
- 一个 MP3 文件
- 一个 JSON 元数据文件

## 常见配置修改

### 修改使用的 GPU 数量

编辑 [run_backend.sh](./run_backend.sh) 顶部的 `GPU_IDS`：

```bash
GPU_IDS=(0)
```

例如：

```bash
GPU_IDS=(0)
GPU_IDS=(0 1)
GPU_IDS=(2 3)
```

`NUM_WORKERS` 会根据 `GPU_IDS` 的长度自动推导。

### 修改端口

编辑 `run_backend.sh` 里的：

- `API_PORT`
- `WORKER_BASE_PORT`
- `BASE_MASTER_PORT`

### 修改 checkpoint 或 tokenizer 路径

编辑 [backend_worker.py](./backend_worker.py) 里的：

- `CHECKPOINTS_DIR`
- `TOKENIZER_PATH`
- `BACKBONE_MODELS`
- `SUPERRES_MODELS`
- `DECODER_CONFIG_PATH`
- `DECODER_CHECKPOINT_PATH`

通过 `/config` 对外暴露的模型名目前是通用名称：

- `default_backbone`
- `default_superres`

## 健康检查与排错

### Worker 健康接口

每个 worker 会暴露：

- `/health`
- `/config`
- `/generate`
- `/download/{filename}`

API 通过 `/health` 判断 worker 当前是 idle、busy 还是 offline。

### 如果前端一直卡在 generating

优先看：

1. `backend/logs/api.log`
2. `backend/logs/worker_0.log`
3. `GET /status`
4. `GET /job/{job_id}`

### 如果 worker 启动失败

常见原因：

- tokenizer 路径不匹配
- Megatron import 路径有问题
- checkpoint 路径不匹配
- 模型 warmup 时 CUDA OOM

### 如果生成文件缺失

优先检查：

- worker 日志里是否有 `ffmpeg` 错误
- `backend/generated_audio/` 下是否真的生成了文件
- API 是否成功从 worker 的 `/download/{filename}` 拉到了文件
