# Environment Setup

This document records the recommended runtime environment for the current release and explains how to configure a clean NVIDIA NGC container into a working setup.

It corresponds to the current repository codebase and only covers the dependencies needed for frontend/backend inference and local development. It does not cover the training environment.

> [!TIP]
> There are currently two supported environment setup paths:
>
> 1. Build an environment image from the repository-level [Dockerfile](./Dockerfile).
> 2. Follow the steps in this document to manually install dependencies inside a clean NGC container.

---

The rest of this document describes the manual setup path.

## 1. Base Environment

The current recommended starting point is the NVIDIA NGC PyTorch container:

- `25.02-py3`

This base image already includes the core CUDA / PyTorch / Transformer Engine stack used by the project, so it is not recommended to reinstall `torch` or `transformer_engine` on top of it.

## 2. Project Directory

The project code can be placed anywhere. All paths below are described relative to the repository root.

For example:

```bash
git clone https://github.com/Khala-Music-AI/Khala.git

cd Khala
```

The rest of this document assumes repository-relative paths rather than any fixed host-machine directory.

## 3. Python Dependencies

The additional Python dependencies for the current project are listed in [requirements.txt](./requirements.txt) at the repository root.

Install them inside the NGC container with:

```bash
python3 -m pip install --break-system-packages -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. System Dependencies

### 4.1 `ffmpeg`

The backend calls `ffmpeg` to export MP3 files after generating WAV files, so it must be installed:

```bash
apt update
apt install -y ffmpeg
```

### 4.2 SSH (optional)

If you want to SSH directly into the container instead of first entering the host machine and then using `docker exec`, you can additionally install and start an SSH service.

For example:

```bash
apt update
apt install -y openssh-server
mkdir -p /var/run/sshd
```

This is not required to run the project. It is only a convenience for development and remote debugging.

## 5. Node.js

The frontend currently runs through the Vite development server, so Node.js is required.

The currently verified version is:

- `node-v24.15.0-linux-x64`

Install it with:

```bash
curl -fsSLO https://nodejs.org/dist/v24.15.0/node-v24.15.0-linux-x64.tar.xz
mkdir -p /usr/local/lib/nodejs
tar -xJf node-v24.15.0-linux-x64.tar.xz -C /usr/local/lib/nodejs
ln -sf /usr/local/lib/nodejs/node-v24.15.0-linux-x64/bin/node /usr/local/bin/node
ln -sf /usr/local/lib/nodejs/node-v24.15.0-linux-x64/bin/npm /usr/local/bin/npm
ln -sf /usr/local/lib/nodejs/node-v24.15.0-linux-x64/bin/npx /usr/local/bin/npx
ln -sf /usr/local/lib/nodejs/node-v24.15.0-linux-x64/bin/corepack /usr/local/bin/corepack
```

Notes:

- The download URL above may change as Node.js versions are updated. In practice, please use the latest official Node.js distribution URL if needed.

Verify the installation:

```bash
node -v
npm -v
npx -v
```

## 6. Frontend Dependencies

From the repository root, run:

```bash
cd frontend
npm install
```

The current frontend is intended to run in development mode:

```bash
npm run dev
```

## 7. Model File Layout

The current code resolves tokenizer files, decoder configuration, and checkpoints through repository-relative paths.

Make sure the directory structure looks like this:

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

Important notes:

- The model files and directory layout inside `checkpoints/` must match the structure expected by the current code.
- The repository itself can live in any directory as long as the internal project structure stays the same.

## 8. Running the Project

### 8.1 Start the backend

```bash
cd backend
bash run_backend.sh
```

By default this starts:

- 1 API process
- as many worker processes as specified by `GPU_IDS`

Stop all backend processes with:

```bash
bash run_backend.sh stop
```

### 8.2 Start the frontend

```bash
cd frontend
npm run dev
```

By default:

- the frontend development server listens on `30869`
- the backend API listens on `8889`

## 9. Advanced Docker Run Example

If you want to mount a host-side project directory into the container, you can use a command like this:

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

Adjust the following for your own environment:

- host mount path
- image name
- whether SSH should be enabled
