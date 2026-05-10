"""
Frontend-facing API dispatcher for music generation.

Responsibilities:
- accept frontend generation requests
- create and track multi-track jobs
- dispatch jobs to worker processes
- expose job status and generated audio downloads
"""

import argparse
import asyncio
import json
import os
import re
import time
import uuid
from collections import deque
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# ============================================================
'''
same_as_backbone
same_as_backbone_no_description
'''
SUPERRES_TEXT_MODE = "same_as_backbone_no_description"

# ============================================================
# CLI
# ============================================================

def parse_args():
    """Parse dispatcher CLI arguments."""
    p = argparse.ArgumentParser(description="Music Generation API")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--worker-base-port", type=int, default=8001)
    p.add_argument("--worker-host", type=str, default="127.0.0.1")
    p.add_argument("--health-interval", type=float, default=1.0)
    return p.parse_args()


# ============================================================
# Lyrics cleaning (from khala/backend/clean_lyrics_rules.py)
# ============================================================

_REPLACEMENTS = {
    '\u3002': '.', '\uff0c': ',', '\uff01': '!', '\uff1f': '?', '\uff1a': ':', '\uff1b': ';',
    '\uff08': '(', '\uff09': ')', '\u3010': '[', '\u3011': ']', '\u300c': '"', '\u300d': '"',
    '\u300e': "'", '\u300f': "'", '\u300a': '<', '\u300b': '>', '\u3001': ',', '\u201c': '"',
    '\u201d': '"', '\u2018': "'", '\u2019': "'", '\u2014\u2014': '-', '\u2026': '...', '\ufe4f': '_',
    '\uff5e': '~', '\u30fb': ' ', '\u060c': ',', '\u3016': '[', '\u3017': ']', '\u3008': '<', '\u3009': '>',
    '\u2013': '-', '\u2014': '-', '\u2015': '-', '\ufe63': '-',
    '\uff02': '"', '\uff03': '#', '\uff04': '$', '\uff05': '%', '\uff06': '&',
    '\uff07': "'", '\uff0a': '*', '\uff0b': '+',
    '\uff0d': '-', '\uff0e': '.', '\uff0f': '/',
    '\uff10': '0', '\uff11': '1', '\uff12': '2', '\uff13': '3', '\uff14': '4',
    '\uff15': '5', '\uff16': '6', '\uff17': '7', '\uff18': '8', '\uff19': '9',
    '\uff1c': '<', '\uff1d': '=', '\uff1e': '>',
    '\uff20': '@',
    '\uff21': 'A', '\uff22': 'B', '\uff23': 'C', '\uff24': 'D', '\uff25': 'E', '\uff26': 'F',
    '\uff27': 'G', '\uff28': 'H', '\uff29': 'I', '\uff2a': 'J', '\uff2b': 'K', '\uff2c': 'L',
    '\uff2d': 'M', '\uff2e': 'N', '\uff2f': 'O', '\uff30': 'P', '\uff31': 'Q', '\uff32': 'R',
    '\uff33': 'S', '\uff34': 'T', '\uff35': 'U', '\uff36': 'V', '\uff37': 'W', '\uff38': 'X',
    '\uff39': 'Y', '\uff3a': 'Z',
    '\uff3b': '[', '\uff3c': '\\', '\uff3d': ']', '\uff3e': '^', '\uff3f': '_', '\uff40': "'",
    '\uff41': 'a', '\uff42': 'b', '\uff43': 'c', '\uff44': 'd', '\uff45': 'e', '\uff46': 'f',
    '\uff47': 'g', '\uff48': 'h', '\uff49': 'i', '\uff4a': 'j', '\uff4b': 'k', '\uff4c': 'l',
    '\uff4d': 'm', '\uff4e': 'n', '\uff4f': 'o', '\uff50': 'p', '\uff51': 'q', '\uff52': 'r',
    '\uff53': 's', '\uff54': 't', '\uff55': 'u', '\uff56': 'v', '\uff57': 'w', '\uff58': 'x',
    '\uff59': 'y', '\uff5a': 'z',
    '\u2236': ':',
    '\u0435': 'e', '\u0430': 'a', '\u043e': 'o', '\u0440': 'p', '\u0441': 'c', '\u0443': 'y', '\u0445': 'x',
    '\u0391': 'A', '\u0392': 'B', '\u0395': 'E', '\u0397': 'H', '\u0399': 'I', '\u039a': 'K', '\u039c': 'M',
    '\u039d': 'N', '\u039f': 'O', '\u03a1': 'P', '\u03a4': 'T', '\u03a5': 'Y', '\u03a7': 'X', '\u0396': 'Z',
    '\u200b': '', '\u00a0': ' ', '\ufeff': '', '\u3000': ' ', '\u2005': ' ', '\u2006': ' ',
    '\u202a': '', '\u202c': '', '\u2028': ' ', '\u180e': '',
    'â\x80\x99': "'", '\xa0': ' ', '\ue3ac': '',
    '\u2032': "'", '\u02bc': "'", '\u00b4': "'", '^': "'", '\u00a8': '"', '\u00b7': '.', '\u00d7': 'x',
    '\u00f7': '/', '|': '-', '\u2605': '',
}

_REPLACEMENTS_PATTERN = re.compile(
    "|".join(re.escape(k) for k in _REPLACEMENTS), flags=re.UNICODE
)


def clean_lyrics(text: str) -> str:
    return _REPLACEMENTS_PATTERN.sub(lambda m: _REPLACEMENTS[m.group(0)], text).strip()


# ============================================================
# Application
# ============================================================

app = FastAPI(title="Music Generation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Worker pool and job state
# ============================================================

class WorkerInfo:
    """Runtime view of one worker process."""

    def __init__(self, worker_id: int, url: str):
        self.worker_id = worker_id
        self.url = url
        self.status = "offline"      # offline / idle / busy
        self.phase = "idle"          # idle / backbone / superres / decoding
        self.progress = 0            # 0-100, real progress from worker
        self.progress_detail = ""    # e.g. "1234/23552 tokens"
        self.gpu_id = -1
        self.seed = 0

    def to_dict(self):
        return {
            "worker_id": self.worker_id,
            "url": self.url,
            "status": self.status,
            "phase": self.phase,
            "gpu_id": self.gpu_id,
        }


WORKERS: List[WorkerInfo] = []
_WORKER_LOCK = asyncio.Lock()

NUM_TRACKS_PER_JOB = 2


def _workers_needed_per_job() -> int:
    """
    Number of workers required to start a job.

    If the deployment has fewer workers than tracks per job, we fall back to
    sequential generation on the available workers instead of leaving the job
    stuck in queue forever.
    """
    return max(1, min(NUM_TRACKS_PER_JOB, len(WORKERS)))


# ============================================================
# Job store
# ============================================================

class TrackResult:
    """Result state for one generated track inside a job."""

    def __init__(self):
        self.status = "queued"       # queued / generating / superres / decoding / done / error
        self.worker_id: int = -1
        self.worker_phase: str = "idle"
        self.phase_start_time: float = 0.0
        self.mp3_bytes: Optional[bytes] = None
        self.wav_bytes: Optional[bytes] = None
        self.mp3_filename: Optional[str] = None
        self.wav_filename: Optional[str] = None
        self.duration_sec: float = 0
        self.error: Optional[str] = None
        self.result: Optional[dict] = None


class Job:
    """One frontend generation request tracked by the dispatcher."""

    def __init__(self, job_id: str, params: dict):
        self.job_id = job_id
        self.num_tracks = NUM_TRACKS_PER_JOB
        self.params = params
        self.created_at = time.time()
        self.cleaned_lyrics = ""
        self.tracks: List[TrackResult] = [TrackResult() for _ in range(NUM_TRACKS_PER_JOB)]
        self.queue_position = 0       # 0 = not queued / dispatched
        self._dispatched = False

    @property
    def status(self) -> str:
        statuses = [t.status for t in self.tracks]
        if all(s == "done" for s in statuses):
            return "completed"
        if any(s == "error" for s in statuses) and all(s in ("done", "error") for s in statuses):
            return "partial"
        if any(s in ("generating", "superres", "decoding") for s in statuses):
            return "generating"
        if all(s == "queued" for s in statuses):
            return "queued"
        return "pending"

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "queue_position": self.queue_position,
            "cleaned_lyrics": self.cleaned_lyrics,
            "tracks": [
                _track_to_dict(i, t) for i, t in enumerate(self.tracks)
            ],
            "created_at": self.created_at,
        }


def _track_to_dict(index: int, t: TrackResult) -> dict:
    progress = _estimate_progress(t)
    phase_display = _phase_display(t, progress)
    return {
        "index": index,
        "status": t.status,
        "progress": progress,
        "phase_display": phase_display,
        "worker_id": t.worker_id,
        "worker_phase": t.worker_phase,
        "duration_sec": t.duration_sec,
        "error": t.error,
    }


def _get_worker_progress(worker_id: int) -> tuple[int, str]:
    """Look up current progress reported by the assigned worker."""
    for w in WORKERS:
        if w.worker_id == worker_id:
            return w.progress, w.progress_detail
    return 0, ""


def _estimate_progress(track: TrackResult) -> int:
    """Map worker phase progress into a stable 0-100 display value."""
    if track.status == "done":
        return 100
    if track.status == "error":
        return 0
    if track.status == "queued":
        return 0
    # Use real progress from worker
    progress, _ = _get_worker_progress(track.worker_id)
    return max(1, min(99, progress)) if progress > 0 else 1


def _phase_display(track: TrackResult, progress: int) -> str:
    """Human-readable phase label exposed to the frontend."""
    if track.status == "queued":
        return "Queued"
    _, detail = _get_worker_progress(track.worker_id)
    if track.worker_phase == "loading_backbone":
        return "Loading backbone model..."
    if track.worker_phase == "loading_superres":
        return "Loading super-resolution model..."
    if track.worker_phase == "loading_decoder":
        return "Loading decoder model..."
    if track.status == "generating":
        token_match = re.match(r"^\s*(\d+)\s*/\s*\d+\s+tokens\s*$", detail or "")
        if token_match:
            generated_tokens = int(token_match.group(1))
            elapsed = max(time.time() - track.phase_start_time, 1e-6)
            tokens_per_second = max(1, int(round(generated_tokens / elapsed)))
            return (
                f"Backbone generating · {progress}% · "
                f"{tokens_per_second} tok/s · {generated_tokens} tokens"
            )
        return f"Backbone generating · {progress}%"
    elif track.status == "superres":
        return f"Super-Resolution Generating · {progress}%"
    elif track.status == "decoding":
        return "Decoding audio..."
    elif track.status == "done":
        return "Done"
    elif track.status == "error":
        return "Error"
    return ""


JOBS: Dict[str, Job] = {}
JOB_QUEUE: deque = deque()


# ============================================================
# Health check background task
# ============================================================

async def _health_check_loop(interval: float):
    """Poll workers to refresh dispatcher-side worker and track state."""
    await asyncio.sleep(3)
    while True:
        async with httpx.AsyncClient(timeout=5.0) as client:
            for w in WORKERS:
                try:
                    r = await client.get(f"{w.url}/health")
                    data = r.json()
                    remote_status = data.get("status", "offline")
                    remote_phase = data.get("phase", "idle")
                    w.gpu_id = data.get("gpu_id", w.worker_id)
                    w.seed = data.get("seed", 0)
                    # Zombie protection: if worker says idle but dispatcher thinks busy,
                    # trust the worker (the task finished or errored).
                    if remote_status == "idle" and w.status == "busy":
                        print(f"[API] Worker {w.worker_id} stuck as busy in dispatcher "
                              f"but reports idle — releasing.")
                    w.status = remote_status
                    w.phase = remote_phase
                    w.progress = data.get("progress", 0)
                    w.progress_detail = data.get("progress_detail", "")
                except Exception:
                    w.status = "offline"
                    w.phase = "idle"
                    w.progress = 0
                    w.progress_detail = ""
        # Update track phases from worker phases
        _sync_track_phases()
        await asyncio.sleep(interval)


def _sync_track_phases():
    """Update in-progress track statuses based on worker-reported phases."""
    worker_by_id = {w.worker_id: w for w in WORKERS}
    for job in JOBS.values():
        for track in job.tracks:
            if track.status in ("generating", "superres", "decoding") and track.worker_id >= 0:
                w = worker_by_id.get(track.worker_id)
                if not w:
                    continue
                if w.phase != track.worker_phase:
                    track.worker_phase = w.phase
                    track.phase_start_time = time.time()
                new_status = _worker_phase_to_track_status(w.phase)
                if new_status and new_status != track.status:
                    track.status = new_status


def _worker_phase_to_track_status(phase: str) -> Optional[str]:
    mapping = {
        "loading_backbone": "generating",
        "backbone": "generating",
        "loading_superres": "superres",
        "superres": "superres",
        "loading_decoder": "decoding",
        "decoding": "decoding",
    }
    return mapping.get(phase)


# ============================================================
# Job TTL cleanup
# ============================================================

JOB_TTL_SECONDS = 1800  # 30 minutes after completion

async def _job_cleanup_loop():
    """Periodically remove completed jobs that are older than TTL."""
    while True:
        await asyncio.sleep(300)  # every 5 minutes
        now = time.time()
        to_delete = []
        for job_id, job in JOBS.items():
            if job.status in ("completed", "partial", "error"):
                age = now - job.created_at
                if age > JOB_TTL_SECONDS:
                    to_delete.append(job_id)
        for job_id in to_delete:
            del JOBS[job_id]
            print(f"[API] Cleaned up expired job {job_id}")


# ============================================================
# Queue processor
# ============================================================

async def _queue_processor_loop():
    """Continuously check queue and dispatch jobs when GPUs are available."""
    while True:
        await asyncio.sleep(1)
        if not JOB_QUEUE:
            continue

        async with _WORKER_LOCK:
            workers_needed = _workers_needed_per_job()
            idle_workers = [w for w in WORKERS if w.status == "idle"]
            if len(idle_workers) < workers_needed:
                # Update queue positions
                for i, job_id in enumerate(JOB_QUEUE):
                    job = JOBS.get(job_id)
                    if job:
                        job.queue_position = i + 1
                continue

            # Dispatch the first job in queue
            job_id = JOB_QUEUE.popleft()
            job = JOBS.get(job_id)
            if not job or job._dispatched:
                continue

            selected = idle_workers[:workers_needed]
            for w in selected:
                w.status = "busy"

            job._dispatched = True
            job.queue_position = 0

        # Update remaining queue positions
        for i, qid in enumerate(JOB_QUEUE):
            qjob = JOBS.get(qid)
            if qjob:
                qjob.queue_position = i + 1

        asyncio.create_task(_dispatch_job(job, selected))


async def _dispatch_job(job: Job, workers: List[WorkerInfo]):
    """
    Dispatch tracks to workers.

    When there are fewer workers than tracks, subsequent tracks are generated in
    later waves on the same reserved worker(s).
    """
    if not workers:
        return

    total_tracks = len(job.tracks)
    batch_size = len(workers)

    for batch_start in range(0, total_tracks, batch_size):
        batch_tracks = job.tracks[batch_start: batch_start + batch_size]
        is_last_batch = batch_start + batch_size >= total_tracks
        tasks = [
            _dispatch_to_worker(
                workers[i],
                track,
                job.params,
                seed_offset=batch_start + i,
                release_worker=is_last_batch,
            )
            for i, track in enumerate(batch_tracks)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


async def _dispatch_to_worker(
    worker: WorkerInfo,
    track: TrackResult,
    params: dict,
    timeout: float = 600.0,
    seed_offset: int = 0,
    release_worker: bool = True,
):
    """Send generation request to a single worker and collect results."""
    track.status = "generating"
    track.worker_id = worker.worker_id
    track.phase_start_time = time.time()

    try:
        request_params = dict(params)
        request_params["seed_override"] = int(worker.seed) + int(seed_offset)
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
            r = await client.post(f"{worker.url}/generate", json=request_params)
            result = r.json()

            if result.get("status") == "ok":
                track.result = result
                track.duration_sec = result.get("duration_sec", 0)
                track.status = "done"

                # Fetch audio files from worker
                mp3_fn = result.get("mp3_filename")
                wav_fn = result.get("wav_filename")
                if mp3_fn:
                    track.mp3_filename = mp3_fn
                    track.mp3_bytes = await _download_worker_file(client, worker, mp3_fn)
                if wav_fn:
                    track.wav_filename = wav_fn
                    track.wav_bytes = await _download_worker_file(client, worker, wav_fn)
            else:
                track.status = "error"
                track.error = result.get("error", "Unknown worker error")
    except Exception as exc:
        track.status = "error"
        track.error = str(exc)
    finally:
        if release_worker:
            _reset_worker_runtime_state(worker)


async def _download_worker_file(
    client: httpx.AsyncClient,
    worker: WorkerInfo,
    filename: str,
) -> Optional[bytes]:
    """Fetch one generated artifact from a worker, returning None on failure."""
    try:
        resp = await client.get(f"{worker.url}/download/{filename}")
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


def _reset_worker_runtime_state(worker: WorkerInfo) -> None:
    """Mark a reserved worker as idle after its assigned job wave finishes."""
    worker.status = "idle"
    worker.phase = "idle"
    worker.progress = 0
    worker.progress_detail = ""


# ============================================================
# Request normalization
# ============================================================


def clamp_generate_request(req: "GenerateRequest") -> tuple[int, int, float]:
    """Clamp frontend request controls into a safe range."""
    duration = max(1, min(10, req.duration))
    top_k_bb = max(1, min(200, req.top_k_bb))
    temperature = max(0.6, min(1.4, req.temperature))
    return duration, top_k_bb, temperature


def extract_user_input(req: "GenerateRequest") -> tuple[str, str]:
    """Return cleaned lyrics and the raw user prompt string."""
    cleaned_lyrics = clean_lyrics(req.lyrics) if req.mode == "vocal" else ""
    user_input = req.tags if req.prompt_mode == "tags" else req.prompt
    return cleaned_lyrics, user_input


def build_worker_params(
    *,
    req: "GenerateRequest",
    cleaned_lyrics: str,
    user_input: str,
    effective_mode: str,
    meta_genre: str,
    meta_language: str,
    meta_tags: str,
    meta_description: str,
    duration: int,
    top_k_bb: int,
    temperature: float,
    ) -> dict:
    """Convert frontend request fields into the worker payload."""
    return {
        "genre": meta_genre,
        "language": meta_language,
        "tags": meta_tags,
        "description": meta_description,
        "duration": duration,
        "lyrics": cleaned_lyrics,
        "backbone_name": "",
        "superres_name": "",
        "top_k_bb": top_k_bb,
        "top_k_sr": 10,
        "temperature": temperature,
        "superres_text_mode": effective_mode,
        "raw_user_input": user_input,
        "raw_mode": req.mode,
        "raw_prompt_mode": req.prompt_mode,
    }


async def resolve_generation_metadata(
    req: "GenerateRequest",
    *,
    user_input: str,
) -> tuple[str, str, str, str, str]:
    """
    Resolve the text metadata passed down to the worker without any LLM step.

    Returns:
        effective_mode, meta_genre, meta_language, meta_tags, meta_description
    """
    effective_mode = SUPERRES_TEXT_MODE
    meta_genre = ""
    meta_language = ""
    print("[API] LLM disabled, using raw frontend inputs.")

    if req.prompt_mode == "tags":
        meta_tags = user_input
        meta_description = ""
    else:
        meta_tags = ""
        meta_description = user_input

    if req.mode == "instrumental":
        meta_language = "Instrumental"

    return effective_mode, meta_genre, meta_language, meta_tags, meta_description


def create_job(params: dict, cleaned_lyrics: str) -> Job:
    """Create and register a new dispatcher job."""
    job_id = str(uuid.uuid4())[:12]
    job = Job(job_id=job_id, params=params)
    job.cleaned_lyrics = cleaned_lyrics
    JOBS[job_id] = job
    return job


async def try_dispatch_or_enqueue(job: Job) -> Optional[List[WorkerInfo]]:
    """Dispatch immediately if workers are available, otherwise enqueue."""
    async with _WORKER_LOCK:
        workers_needed = _workers_needed_per_job()
        idle_workers = [w for w in WORKERS if w.status == "idle"]
        if len(idle_workers) >= workers_needed:
            selected = idle_workers[:workers_needed]
            for w in selected:
                w.status = "busy"
            job._dispatched = True
            job.queue_position = 0
            return selected

        JOB_QUEUE.append(job.job_id)
        job.queue_position = len(JOB_QUEUE)
        return None


# ============================================================
# Frontend request schema
# ============================================================

class GenerateRequest(BaseModel):
    mode: str = "vocal"              # vocal / instrumental
    prompt_mode: str = "tags"        # natural / tags
    prompt: str = ""                 # natural language prompt
    tags: str = ""                   # comma-separated tags
    lyrics: str = ""
    duration: int = 3                # minutes 1-10
    top_k_bb: int = 80               # backbone top-k (user adjustable)
    temperature: float = 1.0         # backbone temperature (user adjustable)


def get_job_or_error(job_id: str) -> Optional[Job]:
    """Return a tracked job or None when the caller should respond with 404."""
    return JOBS.get(job_id)


def get_track_or_error(job: Job, track_idx: int) -> tuple[Optional[TrackResult], Optional[JSONResponse]]:
    """Validate a track index and return either the track or an error response."""
    if track_idx < 0 or track_idx >= job.num_tracks:
        return None, JSONResponse(status_code=400, content={"error": "Track index out of range."})
    return job.tracks[track_idx], None


# ============================================================
# API endpoints
# ============================================================

@app.get("/status")
def status():
    """Return worker pool and queue status for diagnostics."""
    idle_count = sum(1 for w in WORKERS if w.status == "idle")
    active_jobs = sum(1 for j in JOBS.values() if j.status in ("generating", "pending"))
    queued_jobs = len(JOB_QUEUE)
    return {
        "total_gpus": len(WORKERS),
        "idle_gpus": idle_count,
        "active_jobs": active_jobs,
        "queued_jobs": queued_jobs,
        "workers": [w.to_dict() for w in WORKERS],
    }


@app.get("/tags")
def get_tags():
    """Return the frontend tag catalog."""
    tags_path = os.path.join(os.path.dirname(__file__), "tags.json")
    try:
        with open(tags_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "tags.json not found"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Normalize a frontend request, create a job, then dispatch or enqueue it."""
    print(f"[API] POST /generate  body={req.model_dump()}")
    duration, top_k_bb, temperature = clamp_generate_request(req)
    cleaned_lyrics, user_input = extract_user_input(req)
    (
        effective_mode,
        meta_genre,
        meta_language,
        meta_tags,
        meta_description,
    ) = await resolve_generation_metadata(
        req,
        user_input=user_input,
    )

    worker_params = build_worker_params(
        req=req,
        cleaned_lyrics=cleaned_lyrics,
        user_input=user_input,
        effective_mode=effective_mode,
        meta_genre=meta_genre,
        meta_language=meta_language,
        meta_tags=meta_tags,
        meta_description=meta_description,
        duration=duration,
        top_k_bb=top_k_bb,
        temperature=temperature,
    )
    job = create_job(worker_params, cleaned_lyrics)
    selected = await try_dispatch_or_enqueue(job)

    if selected:
        print(f"[API] Dispatching job {job.job_id} immediately.")
        asyncio.create_task(_dispatch_job(job, selected))
    else:
        print(f"[API] Queued job {job.job_id} at position {job.queue_position}.")

    return {
        "status": "accepted" if selected else "queued",
        "job_id": job.job_id,
        "num_tracks": NUM_TRACKS_PER_JOB,
        "queue_position": job.queue_position,
    }


@app.get("/job/{job_id}")
def get_job(job_id: str):
    """Return job status and per-track progress."""
    job = get_job_or_error(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": f"Job {job_id} not found."},
        )
    return job.to_dict()


@app.get("/job/{job_id}/track/{track_idx}/mp3")
def get_track_mp3(job_id: str, track_idx: int):
    """Return the generated MP3 bytes for a completed track."""
    job = get_job_or_error(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found."})
    track, error = get_track_or_error(job, track_idx)
    if error:
        return error
    if track.status != "done" or not track.mp3_bytes:
        return JSONResponse(status_code=404, content={"error": "Track not ready or MP3 not available."})
    return Response(
        content=track.mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(track.mp3_bytes)),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{track.mp3_filename or f"track_{track_idx}.mp3"}"',
        },
    )


@app.get("/job/{job_id}/track/{track_idx}/wav")
def get_track_wav(job_id: str, track_idx: int):
    """Return the generated WAV bytes for a completed track."""
    job = get_job_or_error(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found."})
    track, error = get_track_or_error(job, track_idx)
    if error:
        return error
    if track.status != "done" or not track.wav_bytes:
        return JSONResponse(status_code=404, content={"error": "Track not ready or WAV not available."})
    return Response(
        content=track.wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(track.wav_bytes)),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'attachment; filename="{track.wav_filename or f"track_{track_idx}.wav"}"',
        },
    )


# ============================================================
# Entry point
# ============================================================

def main():
    """Initialize the dispatcher worker pool and start the API server."""
    args = parse_args()

    for i in range(args.num_workers):
        port = args.worker_base_port + i
        url = f"http://{args.worker_host}:{port}"
        WORKERS.append(WorkerInfo(worker_id=i, url=url))

    print(f"[API] Managing {args.num_workers} workers "
          f"({args.worker_host}:{args.worker_base_port}-"
          f"{args.worker_base_port + args.num_workers - 1})")
    workers_needed = _workers_needed_per_job()
    print(f"[API] {NUM_TRACKS_PER_JOB} tracks/job, "
          f"{workers_needed} worker(s) needed/job, "
          f"max {max(1, args.num_workers // workers_needed)} concurrent job wave(s)")

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(_health_check_loop(args.health_interval))
        asyncio.create_task(_job_cleanup_loop())
        asyncio.create_task(_queue_processor_loop())

    print(f"[API] Starting on http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
