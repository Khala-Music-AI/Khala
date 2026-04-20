# Khala Backend

[中文](./README_zh.md) | English

This repository currently ships the cleaned backend runtime for music generation.

The backend is organized into three files under [backend/](./backend/):

- [run_backend.sh](./backend/run_backend.sh): starts workers and the API server
- [backend_api.py](./backend/backend_api.py): frontend-facing dispatcher and job manager
- [backend_worker.py](./backend/backend_worker.py): single-GPU inference worker

For the full backend overview, request flow, launch instructions, and debugging notes, see:

- [backend/README_backend.md](./backend/README_backend.md)
