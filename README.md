<div align="center">

<img src="./assets/logo.png" width="280" alt="Khala Logo" />

# High-Fidelity Song Generation With a Unified Acoustic-Token Pipeline

English | [中文](./README_zh.md)

</div>

<div align="center">

<img alt="Demo" src="https://img.shields.io/badge/%F0%9F%8E%A7%20Demo-Coming%20Soon-gray">

<a href="https://arxiv.org/abs/2605.01790">
  <img alt="Paper" src="https://img.shields.io/badge/%F0%9F%93%84%20Paper-arXiv-b31b1b">
</a>

<a href="https://huggingface.co/liujiafeng/Khala-MusicGeneration-v1.0">
  <img alt="Model Weights" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hugging%20Face-ffc107">
</a>
<a href="./ENVIRONMENT_SETUP.md">
  <img alt="Environment Setup" src="https://img.shields.io/badge/%F0%9F%9B%A0%20Environment-Setup-4c8eda">
</a>
<a href="./backend/README_backend.md">
  <img alt="Backend Docs" src="https://img.shields.io/badge/%F0%9F%A7%A0%20Backend-Docs-6f42c1">
</a>

</div>

## ✨ What Is Khala?

Khala is an open-source system for high-fidelity song generation, capable of generating complete songs from text descriptions and lyric conditions. Unlike approaches built around semantic tokens, diffusion models, or multi-stage audio generation stacks, Khala follows a unified acoustic-token route and generates both coarse musical structure and fine acoustic detail within the same discrete audio representation space.

The core characteristics of Khala include:

- **Full-song generation**: designed for complete song generation rather than short clips or loop-style accompaniment.
- **Text and lyric control**: supports natural-language prompts and lyrics to control style, mood, vocals, and content.
- **Unified acoustic-token representation**: built on a 64-layer RVQ acoustic token hierarchy that represents audio as coarse-to-fine discrete acoustic tokens.
- **Two-stage generation pipeline**: a backbone first generates coarse acoustic tokens, then a super-resolution model completes higher RVQ token layers, and finally a decoder reconstructs the waveform.
- **Complete system implementation**: includes a frontend UI, a FastAPI backend dispatcher, a single-GPU inference worker, model loading, and the end-to-end audio generation path rather than just standalone inference scripts.

## 📰 News

- `⚠️ [2026-05-07]` We have identified a potential issue that may significantly affect inference quality. The problem is currently under investigation and may be related to numerical precision. Until this notice is removed, please treat current generation quality as unstable.

### ✅ Updated

- `[2026-05-05]` The arXiv paper is now available: [Khala: Scaling Acoustic Token Language Models Toward High-Fidelity Music Generation](https://arxiv.org/abs/2605.01790)
- `[2026-05-01]` The codebase, environment documentation, and Dockerfile have been cleaned up for release.
- `[2026-05-11]` Backend inference launch now supports single-GPU safe startup by default, plus multi-GPU and runtime-mode overrides for deployment compatibility.

### ⏳ TODOs

- `[Coming Soon]` A full deployment guide for musicians and beginner users.
- `[Coming Soon]` Discord community server.

### 🖥️ Web UI
#### Prompt Mode
![Khala Frontend Demo 1](./assets/front_1.png)
#### Tag Mode
![Khala Frontend Demo 2](./assets/front_2.png)

### 🎧 Audio Samples

The audio demo page is still under construction and will be released soon.

## ✅ Runtime Requirements

The current release is mainly intended for researchers and developers who are already familiar with GPU servers.

- NVIDIA GPU, with 24GB or more VRAM recommended for the full inference pipeline, such as an RTX 4090 or a higher-tier GPU.
- Docker and NVIDIA Container Toolkit.
- A CUDA-compatible NVIDIA driver.
- Python and Node.js are already included in the prebuilt image.
- Model weights need to be downloaded into the `checkpoints/` directory at the repository root.

## 🚀 Quick Start

This section is intended for researchers and developers who are already comfortable with basic Docker and CUDA workflows, and provides the shortest path to running the system.

If you want to configure the environment step by step from a clean NGC container, please read:

- [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md)
- [ENVIRONMENT_SETUP_zh.md](./ENVIRONMENT_SETUP_zh.md)

If you want to understand the backend structure and runtime logic, please read:

- [backend/README_backend.md](./backend/README_backend.md)
- [backend/README_backend_zh.md](./backend/README_backend_zh.md)

### 1. Prepare the runtime environment
The currently available prebuilt image is:
```bash
docker pull ghcr.io/davidliujiafeng/khala-env:ngc25.02-node24

docker run --gpus all -it --rm \
  --name khala \
  -p 30869:30869 \
  -p 8889:8889 \
  ghcr.io/davidliujiafeng/khala-env:ngc25.02-node24
```
> Note: the command above uses `--rm`, so files created inside the container will be removed after the container exits. If you want a long-lived development container or want to keep downloaded model weights, use a mounted directory or remove `--rm`.

### 2. Clone the repository
After entering the container, run:
```bash
cd /workspace
git clone https://github.com/Khala-Music-AI/Khala.git
cd Khala
```

### 3. Download the model checkpoints

Model repository:

- [Hugging Face: liujiafeng/Khala-MusicGeneration-v1.0](https://huggingface.co/liujiafeng/Khala-MusicGeneration-v1.0)

From the repository root, run:

```bash
mkdir -p checkpoints
hf download liujiafeng/Khala-MusicGeneration-v1.0 --local-dir checkpoints
```

This command downloads the model repository contents into the local `checkpoints/` directory.

### 4. Start the backend

```bash
cd /workspace/Khala/backend
bash run_backend.sh
```

The default launcher now starts in a single-GPU safe mode. Advanced users can also select specific GPU ids and switch between `one_shot` and `keep_loaded` runtime modes from the same script; see [backend/README_backend.md](./backend/README_backend.md) for details.

### 5. Start the frontend

In another terminal, run:

```bash
cd /workspace/Khala/frontend
npm install
npm run dev
```

### 6. Open the web UI

Default URL:

- [http://127.0.0.1:30869](http://127.0.0.1:30869)

## 🧠 System Overview

The current system has three layers:

- Frontend: accepts prompts, lyrics, and generation settings, and displays results.
- API dispatcher: receives requests, creates jobs, queues them, and dispatches them to idle workers.
- Inference worker: runs backbone, super-resolution, and decoder inference.

The request path is:

```mermaid
flowchart LR
    A["Frontend UI"] --> B["backend_api.py"]
    B --> C["backend_worker.py"]
    C --> D["Backbone"]
    D --> E["Super-resolution"]
    E --> F["Decoder"]
    F --> G["Generated Audio"]
    G --> B
    B --> A
```

## 🔗 Project Resources

- Demo page: Coming Soon
- arXiv paper: [Khala: Scaling Acoustic Token Language Models Toward High-Fidelity Music Generation](https://arxiv.org/abs/2605.01790)
- Model weights: `https://huggingface.co/liujiafeng/Khala-MusicGeneration-v1.0`
- Environment setup: [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md)
- Backend docs: [backend/README_backend.md](./backend/README_backend.md)

## 🗂 Repository Structure

```text
Khala/
├── backend/
├── frontend/
├── core/
├── models/
├── checkpoints/
├── assets/
├── Dockerfile
├── requirements.txt
├── ENVIRONMENT_SETUP.md
└── ENVIRONMENT_SETUP_zh.md
```

Main directories:

- `frontend/`: frontend pages and the Vite project.
- `backend/`: backend API, worker, and launcher scripts.
- `core/`: project-specific core modules.
- `models/`: Megatron, decoder, and tokenizer related code.
- `checkpoints/`: model checkpoint directory.
- `assets/`: images used by the README and demo materials.

## 📚 Citation

If this project is helpful to your research or development work, you are welcome to cite our paper:

- [Khala: Scaling Acoustic Token Language Models Toward High-Fidelity Music Generation](https://arxiv.org/abs/2605.01790)

The final BibTeX information will be added later to both the paper page and the repository documentation.

## 🙏 Acknowledgements

The current implementation builds on a number of excellent open-source projects and tools, including but not limited to:

- NVIDIA NGC
- Megatron / Megatron Core
- Hugging Face
- FastAPI
- Vite / React

## 📜 License

The model weights are currently intended to be released under `CC BY-NC 4.0` (Creative Commons Attribution-NonCommercial 4.0 International).

## 💬 Contact

Feel free to join the WeChat group for discussion, usage questions, and future updates:

<div align="center">
  <img src="./assets/wechat_group.jpg" width="320" alt="Khala WeChat Group QR Code" />
</div>
