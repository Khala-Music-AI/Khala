# Environment Setup

This document records the currently recommended runtime environment and how to go from a clean NVIDIA NGC container to a working local setup.

It is scoped to inference and local frontend/backend development for the current repository. It does not cover the full training environment.

[中文](./ENVIRONMENT_SETUP_zh.md)

> [!TIP]
> There are currently two supported setup paths:
>
> 1. Build an environment image from the repository-level [Dockerfile](./Dockerfile).
> 2. Follow this document step by step to configure a clean NGC container manually.

---

The rest of this document describes the manual setup path.

## 1. Base Image

Start from the NVIDIA NGC PyTorch container:

- `25.02-py3`

This base image already provides the core CUDA / PyTorch / Transformer Engine stack used by the project, so it is not recommended to reinstall `torch` or `transformer_engine` on top of it.

## 2. Project Location

The repository can live in any directory. There is no required absolute path.

For example:

```bash
git clone <your-repo-url>
cd Khala-Music-Generation
```

All paths in this document are described relative to the repository root, not to a fixed host-machine location.

## 3. Python Dependencies

The current extra Python dependencies are listed in [requirements.txt](./requirements.txt).

Install them inside the NGC container with:

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

## 4. System Dependencies

### 4.1 `ffmpeg`

The backend exports MP3 files from generated WAV files, so `ffmpeg` is required:

```bash
apt update
apt install -y ffmpeg
```

### 4.2 SSH (optional)

If you want to SSH directly into the container instead of entering the host machine first and then using `docker exec`, you can also install and run an SSH server.

For example:

```bash
apt update
apt install -y openssh-server
mkdir -p /var/run/sshd
```

This is optional and intended for development convenience.

## 5. Node.js

The frontend currently runs through the Vite development server, so Node.js is required.

The version verified with the current project is:

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

- The download URL may change as Node.js releases newer versions. Check the official Node.js distribution page if needed.

Verify the installation:

```bash
node -v
npm -v
npx -v
```

## 6. Frontend Dependencies

From the repository root:

```bash
cd frontend
npm install
```

The current frontend is intended to run in development mode:

```bash
npm run dev
```

## 7. Model File Layout

The current code resolves tokenizer, decoder config, and checkpoints by repository-relative paths.

Make sure the repository layout looks like this:

```text
Khala-Music-Generation/
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
- The repository can live in any directory as long as the internal project structure stays the same.

## 8. Running the Project

### 8.1 Start the backend

```bash
cd backend
bash run_backend.sh
```

By default this starts:

- 1 API process
- one worker process per entry in `GPU_IDS`

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

- the frontend dev server listens on `7869`
- the backend API listens on `8889`

## 9. Example Docker Run Command

If you want to mount a host workspace into the container, a typical command looks like this:

```bash
docker run -d \
  --name khala_dev \
  --gpus all \
  -p 2222:22 \
  -p 7869:7869 \
  -p 8889:8889 \
  -v /path/to/your/workspace:/workspace \
  <your-image> \
  /usr/sbin/sshd -D
```

Adjust the following for your own setup:

- host mount path
- image name
- whether SSH is enabled
