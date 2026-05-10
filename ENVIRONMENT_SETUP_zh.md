# 环境安装说明

本文档记录当前版本的推荐运行环境，以及如何从一个干净的 NVIDIA NGC 容器配置到可运行状态。

这份说明对应当前仓库代码，不包含模型训练环境，只覆盖前后端推理与本地联调所需依赖。

> [!TIP]
> 当前提供两种环境配置方式：
>
> 1. 使用仓库根目录的 [Dockerfile](./Dockerfile) 构建环境镜像。
> 2. 按照本文档中的步骤，在一个干净的 NGC 容器里手动安装依赖。

---

以下内容是手动配置教程。

## 1. 基础环境

当前推荐从 NVIDIA NGC 的 PyTorch 容器开始：

- `25.02-py3`

这个基础镜像已经自带了当前项目依赖的核心 CUDA / PyTorch / Transformer Engine 环境，因此不建议在此基础上自行重新安装 `torch` 或 `transformer_engine`。

## 2. 项目目录

项目代码可以放在任意位置，后续路径均默认相对于仓库根目录。

例如：

```bash
git clone https://github.com/Khala-Music-AI/Khala.git

cd Khala
```

后续文档中的路径均默认相对于仓库根目录，而不是固定写死某个宿主机目录。

## 3. Python 依赖

当前项目额外依赖已经整理到仓库根目录的 [requirements.txt](./requirements.txt)。

在 NGC 容器内安装：

```bash
python3 -m pip install --break-system-packages -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 系统依赖

### 4.1 `ffmpeg`

后端会在生成 WAV 后调用 `ffmpeg` 导出 MP3，因此需要安装：

```bash
apt update
apt install -y ffmpeg
```

### 4.2 SSH（可选）

如果你希望像开发环境一样直接 SSH 登录容器，而不是先进入宿主机再 `docker exec`，可以额外安装并启动 SSH 服务。

例如：

```bash
apt update
apt install -y openssh-server
mkdir -p /var/run/sshd
```

这一步不是项目运行必需项，只是开发和远程调试更方便。

## 5. Node.js

前端当前使用 Vite 开发服务器运行，因此需要安装 Node.js。

目前验证可用的版本是：

- `node-v24.15.0-linux-x64`

安装方式如下：

```bash
curl -fsSLO https://nodejs.org/dist/v24.15.0/node-v24.15.0-linux-x64.tar.xz
mkdir -p /usr/local/lib/nodejs
tar -xJf node-v24.15.0-linux-x64.tar.xz -C /usr/local/lib/nodejs
ln -sf /usr/local/lib/nodejs/node-v24.15.0-linux-x64/bin/node /usr/local/bin/node
ln -sf /usr/local/lib/nodejs/node-v24.15.0-linux-x64/bin/npm /usr/local/bin/npm
ln -sf /usr/local/lib/nodejs/node-v24.15.0-linux-x64/bin/npx /usr/local/bin/npx
ln -sf /usr/local/lib/nodejs/node-v24.15.0-linux-x64/bin/corepack /usr/local/bin/corepack
```

说明：

- 上面的下载地址可能会随着 Node.js 版本更新而变化，实际使用时请以 Node.js 官网最新可用地址为准。

验证：

```bash
node -v
npm -v
npx -v
```

## 6. 前端依赖

在仓库根目录下执行：

```bash
cd frontend
npm install
```

当前版本前端推荐通过开发模式启动：

```bash
npm run dev
```

## 7. 模型文件放置方式

当前代码默认按仓库相对路径查找 tokenizer、decoder 配置和 checkpoints。

请确保目录结构如下：

```text
Khala/
├── backend/
├── frontend/
├── core/
├── models/
│   ├── Decoder/
│   ├── Megatron/
│   └── Tokenizer/
└── checkpoints/
    ├── ...
    └── ...
```

注意：

- `checkpoints/` 目录下的模型文件和目录结构需要与当前代码配置保持一致。
- 项目代码可以放在任意目录，只要仓库内部结构保持一致即可。

## 8. 启动方式

### 8.1 启动后端

```bash
cd backend
bash run_backend.sh
```

默认会启动：

- 1 个 API 进程
- `GPU_IDS` 中指定数量的 worker 进程

停止：

```bash
bash run_backend.sh stop
```

### 8.2 启动前端

```bash
cd frontend
npm run dev
```

默认情况下：

- 前端开发服务器监听 `30869`
- 后端 API 监听 `8889`

## 9. Docker 高级启动示例

如果你希望把宿主机上的项目目录挂载进容器，可以使用类似这样的命令：

```bash
docker run -d \
  --name khala_dev \
  --gpus all \
  -p 2222:22 \
  -p 30869:30869 \
  -p 8889:8889 \
  -v /path/to/your/workspace:/workspace \
  <your-image> \
  /usr/sbin/sshd -D
```

需要根据你自己的环境调整：

- 宿主机挂载路径
- 镜像名称
- 是否启用 SSH
