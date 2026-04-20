# Backend Overview

[中文](./README_backend_zh.md) | [English](../README.md)

This backend is split into three files:

- [run_backend.sh](./run_backend.sh): starts workers and the API server
- [backend_api.py](./backend_api.py): frontend-facing dispatcher, queue, and job tracking
- [backend_worker.py](./backend_worker.py): single-GPU inference worker

The design is intentionally two-layered:

- The API process does not run GPU inference. It accepts requests, creates jobs, assigns idle workers, and returns status to the frontend.
- Each worker process owns one GPU, loads models, runs backbone generation, runs super-resolution, decodes audio, and serves generated files back to the API.

## Request Flow

One frontend request goes through this path:

1. The frontend sends `POST /generate` to `backend_api.py`.
2. The API normalizes the request, cleans lyrics, creates a job, and either dispatches it immediately or puts it into the queue.
3. The API sends the worker payload to one or more workers through `POST /generate`.
4. `backend_worker.py` loads or reuses the tokenizer and models, then runs:
   - prompt preparation
   - backbone generation
   - super-resolution
   - decoder waveform reconstruction
   - MP3 export
5. The worker writes output files under `backend/generated_audio/` and returns filenames plus metadata.
6. The API downloads the generated files from the worker, stores them in memory for the job result, and exposes them through `/job/{job_id}` and `/job/{job_id}/track/{track_idx}/mp3|wav`.

## File Roles

### `run_backend.sh`

This is the entry point for local or containerized deployment.

It is responsible for:

- defining user-facing launch configuration
- starting one worker process per GPU in `GPU_IDS`
- waiting until workers become healthy
- starting the API dispatcher
- tailing logs in the foreground

Important config lives at the top of the file:

- `GPU_IDS`: physical GPUs used by workers
- `NUM_WORKERS`: derived automatically from `GPU_IDS`
- `API_PORT`
- `WORKER_BASE_PORT`
- `BASE_MASTER_PORT`
- `BASE_SEED`
- `MEGATRON_ARGS`

### `backend_api.py`

This file is the frontend-facing orchestration layer.

It is responsible for:

- accepting generation requests
- cleaning lyrics and normalizing request fields
- building worker payloads
- creating and tracking jobs
- queueing when all workers are busy
- polling worker health and syncing job progress
- returning job state and generated audio files

It does not perform GPU inference itself.

### `backend_worker.py`

This file is the single-GPU inference runtime.

It is responsible for:

- loading the tokenizer
- loading the Megatron backbone
- loading the super-resolution model
- loading the DAC RVQ decoder
- preparing prompts
- generating q0/q1 backbone tokens
- expanding to q0..q63 with super-resolution
- decoding waveform audio
- exporting WAV and MP3 files

Runtime strategy:

- tokenizer stays loaded
- backbone stays loaded
- superres is loaded per request and then released
- decoder is loaded per request and then released

## How To Run

From the backend directory:

```bash
cd backend
bash run_backend.sh
```

Stop all backend processes:

```bash
bash run_backend.sh stop
```

## Logs

Logs are written to:

- `backend/logs/api.log`
- `backend/logs/worker_0.log`
- `backend/logs/worker_1.log`
- ...

Useful commands:

```bash
tail -f backend/logs/api.log
tail -f backend/logs/worker_0.log
```

## Output Files

Generated files are written under:

- `backend/generated_audio/*.wav`
- `backend/generated_audio/*.mp3`
- `backend/generated_audio/*.json`

Each request writes:

- one WAV file
- one MP3 file
- one JSON metadata file

## Common Configuration Changes

### Change how many GPUs are used

Edit `GPU_IDS` in [run_backend.sh](./run_backend.sh):

```bash
GPU_IDS=(0)
```

Examples:

```bash
GPU_IDS=(0)
GPU_IDS=(0 1)
GPU_IDS=(2 3)
```

`NUM_WORKERS` is derived automatically from the length of `GPU_IDS`.

### Change ports

Edit these values in `run_backend.sh`:

- `API_PORT`
- `WORKER_BASE_PORT`
- `BASE_MASTER_PORT`

### Change model checkpoints or tokenizer paths

Edit these values in [backend_worker.py](./backend_worker.py):

- `CHECKPOINTS_DIR`
- `TOKENIZER_PATH`
- `BACKBONE_MODELS`
- `SUPERRES_MODELS`
- `DECODER_CONFIG_PATH`
- `DECODER_CHECKPOINT_PATH`

The public model names exposed by `/config` are intentionally generic:

- `default_backbone`
- `default_superres`

## Health And Debugging

### Worker health

Each worker exposes:

- `/health`
- `/config`
- `/generate`
- `/download/{filename}`

The API polls `/health` to decide whether a worker is idle, busy, or offline.

### If the frontend is stuck on generating

Check these first:

1. `backend/logs/api.log`
2. `backend/logs/worker_0.log`
3. `GET /status`
4. `GET /job/{job_id}`

### If the worker fails during startup

Common causes:

- tokenizer path mismatch
- Megatron import path problems
- checkpoint path mismatch
- CUDA OOM during model warmup

### If generated files are missing

Check:

- the worker log for `ffmpeg` errors
- whether the files exist under `backend/generated_audio/`
- whether the API successfully fetched `/download/{filename}` from the worker

