"""
Single-GPU music generation worker.

Responsibilities:
- load tokenizer and models
- run backbone / superres / decoder inference
- expose health, config, generate, and download endpoints
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


# ============================================================
# Paths and code roots
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MEGATRON_ROOT = os.path.join(PROJECT_ROOT, "models", "Megatron")
DECODER_ROOT = os.path.join(PROJECT_ROOT, "models", "Decoder")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "Tokenizer")
OUTPUT_DIR = os.path.join(BACKEND_DIR, "generated_audio")
LOG_DIR = os.path.join(BACKEND_DIR, "logs")


def prepend_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


prepend_sys_path(PROJECT_ROOT)
prepend_sys_path(MEGATRON_ROOT)
prepend_sys_path(DECODER_ROOT)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# Core constants
# ============================================================

BOS = "<|begin_of_text|>"
EOM = "<|eom_id|>"
BOL = "<|start_header_id|>"
EOT = "<|eot_id|>"
BOA = "<|python_tag|>"
END_OF_LINE = "<|end_header_id|>"

PAD_TOKEN_ID = -1
NUM_QUANTIZERS = 64
VQ0_START_ID = 128256
VQ0_END_ID = VQ0_START_ID + 1024
VQ1_START_ID = VQ0_END_ID
VQ1_END_ID = VQ1_START_ID + 1024
TASK0_MARKER = 128255
DECODER_SAMPLE_RATE = 44100
CODEC_FPS = 21.5

BACKBONE_MAX_PROMPT_LEN = 4096
SUPERRES_MAX_PROMPT_LEN = 2048
TOKENS_PER_MINUTE = 2584
DECODER_CHUNK_SIZE = 1920
DECODER_CHUNK_OVERLAP = 480

# ============================================================
# Runtime configuration
# ============================================================

BACKBONE_MODELS = {
    "q01_354k_tag_desc_v0 (0217)": {
        "path": os.path.join(CHECKPOINTS_DIR, "backbone"),
        "vocab_size": 130304,
    }
}

SUPERRES_MODELS = {
    "q01_ft5k_super_v2 (0215)": {
        "path": os.path.join(CHECKPOINTS_DIR, "superresolution"),
        "vocab_size": 193792,
    }
}

DECODER_CONFIG_PATH = os.path.join(DECODER_ROOT, "dac_rvq_1024_64_golden.yaml")
DECODER_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "dac_rvq_2490000.ckpt")


GENRE_OPTIONS = [
    "Pop", "Rock", "R&B", "Hip-Hop", "Electronic", "Jazz", "Classical",
    "Folk", "Country", "Metal", "Latin", "Reggae", "Blues", "Funk",
    "Soul", "Indie", "Alternative", "Dance", "Acoustic",
]

LANGUAGE_OPTIONS = [
    "Chinese", "English", "Japanese", "Korean", "Cantonese", "Instrumental",
]


# ============================================================
# Shared runtime state
# ============================================================


@dataclass
class WorkerRuntimeState:
    status: str = "initializing"
    phase: str = "idle"
    gpu_id: int = -1
    seed: int = 0
    progress: int = 0
    progress_detail: str = ""


RESOURCES: dict = {
    "tokenizer": None,
    "backbone": None,
    "backbone_name": None,
    "backbone_engine": None,
    "superres": None,
    "superres_name": None,
    "decoder": None,
}

STATE = WorkerRuntimeState()
STATE_LOCK = threading.Lock()
RUNTIME_MODE = "one_shot"


# ============================================================
# Request / response schema
# ============================================================


class GenerateRequest(BaseModel):
    genre: str = "Pop"
    language: str = "Chinese"
    tags: str = ""
    description: str = ""
    duration: int = 2
    lyrics: str = ""
    backbone_name: str = ""
    superres_name: str = ""
    top_k_bb: int = 50
    top_k_sr: int = 10
    temperature: float = 1.0
    superres_text_mode: str = "same_as_backbone"
    raw_user_input: str = ""
    raw_mode: str = ""
    raw_prompt_mode: str = ""
    seed_override: int = 0


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="Music Worker API", version="1.0.0")


# ============================================================
# Small helpers
# ============================================================


def set_phase(phase: str, progress: int = 0, detail: str = "") -> None:
    STATE.phase = phase
    STATE.progress = progress
    STATE.progress_detail = detail


def set_status(status: str) -> None:
    STATE.status = status


def state_snapshot() -> dict:
    return {
        "status": STATE.status,
        "phase": STATE.phase,
        "gpu_id": STATE.gpu_id,
        "seed": STATE.seed,
        "progress": STATE.progress,
        "progress_detail": STATE.progress_detail,
        "runtime_mode": RUNTIME_MODE,
        "backbone_loaded": RESOURCES["backbone_name"],
        "superres_loaded": RESOURCES["superres_name"],
        "decoder_loaded": RESOURCES["decoder"] is not None,
    }


def write_status_snapshot(path: str) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as file:
        json.dump(state_snapshot(), file, ensure_ascii=False)


def start_status_writer(path: str) -> tuple[threading.Event, threading.Thread] | tuple[None, None]:
    if not path:
        return None, None

    stop_event = threading.Event()

    def loop() -> None:
        while not stop_event.is_set():
            try:
                write_status_snapshot(path)
            except Exception:
                pass
            stop_event.wait(0.5)
        try:
            write_status_snapshot(path)
        except Exception:
            pass

    thread = threading.Thread(target=loop, name="status-writer", daemon=True)
    thread.start()
    return stop_event, thread


def clear_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()


def free_resource(key: str) -> None:
    if RESOURCES[key] is not None:
        del RESOURCES[key]
        RESOURCES[key] = None
        clear_cuda_memory()


def clamp_prompt_ids(prompt_ids: list[int], max_len: int, label: str) -> list[int]:
    if len(prompt_ids) <= max_len:
        return prompt_ids
    last_token = prompt_ids[-1]
    trimmed = prompt_ids[: max_len - 1] + [last_token]
    print(f"[Worker] {label} prompt truncated to {len(trimmed)} tokens (max {max_len})")
    return trimmed


def default_backbone_name(name: str) -> str:
    return name or next(iter(BACKBONE_MODELS))


def default_superres_name(name: str) -> str:
    return name or next(iter(SUPERRES_MODELS))


# ============================================================
# Megatron setup
# ============================================================


def patch_language_model_embedding() -> None:
    """Swap Megatron's default word embedding for the project-specific multi-layer version."""
    import megatron.core.models.common.embeddings.language_model_embedding as embedding_module
    from megatron.core import tensor_parallel

    original_init = embedding_module.LanguageModelEmbedding.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.word_embeddings = tensor_parallel.MultiLayerVocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.embedding_init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
            tp_group=self.tp_group,
        )

    embedding_module.LanguageModelEmbedding.__init__ = patched_init
    print("[Worker] LanguageModelEmbedding patched -> MultiLayerVocabParallelEmbedding.")


def add_worker_args(parser):
    from examples.inference.gpt.utils import add_common_inference_args

    add_common_inference_args(parser)

    group = parser.add_argument_group(title="worker")
    group.add_argument("--worker-port", type=int, default=8001, help="FastAPI port")
    group.add_argument(
        "--runtime-mode",
        type=str,
        default="one_shot",
        choices=["keep_loaded", "one_shot"],
        help="Worker runtime mode: keep models loaded, or spawn one-shot subprocesses per request.",
    )
    group.add_argument("--stream", action="store_true", default=False)
    group.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        dest="max_batch_size",
        help="Deprecated alias for --inference-max-requests",
    )
    group.add_argument("--child-once", action="store_true", default=False, help=argparse.SUPPRESS)
    group.add_argument("--child-stage", type=str, default="", help=argparse.SUPPRESS)
    group.add_argument("--artifact-dir", type=str, default="", help=argparse.SUPPRESS)
    group.add_argument("--request-json", type=str, default="", help=argparse.SUPPRESS)
    group.add_argument("--result-json", type=str, default="", help=argparse.SUPPRESS)
    group.add_argument("--status-json", type=str, default="", help=argparse.SUPPRESS)
    return parser


class PassThroughTokenizer:
    """Small adapter used by Megatron's static inference engine for token-id prompts."""

    def __init__(self, vocab_size: int, eod_id: int = 128001):
        self.vocab_size = vocab_size
        self.eod = eod_id
        self.bos = eod_id
        self.eos = eod_id

    def tokenize(self, text):
        return text

    def detokenize(self, token_ids, **kwargs):
        return token_ids

    def offsets(self, ids, text):
        return [0] * len(ids)


def get_backbone_inference_engine(model, vocab_size: int):
    """Build the static inference engine used for backbone autoregressive decoding."""
    from megatron.core.inference.contexts import StaticInferenceContext
    from megatron.core.inference.engines import StaticInferenceEngine
    from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
        GPTInferenceWrapper,
    )
    from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
        InferenceWrapperConfig,
    )
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )
    from megatron.training import get_args

    args = get_args()

    if getattr(args, "max_batch_size", None) is not None:
        warnings.warn("`--max-batch-size` deprecated; use `--inference-max-requests`.")
        args.inference_max_batch_size = max(args.max_batch_size, args.inference_max_batch_size)

    tokenizer = PassThroughTokenizer(vocab_size=vocab_size)
    wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_requests=args.inference_max_batch_size,
        inference_max_seq_length=args.inference_max_seq_length,
        nccl_all_reduce_for_prefill=getattr(args, "nccl_all_reduce_for_prefill", False),
        fp8=getattr(args, "fp8", False),
    )

    inference_context = StaticInferenceContext.from_config(wrapper_config)
    wrapped_model = GPTInferenceWrapper(model, wrapper_config, inference_context)
    controller = TextGenerationController(
        inference_wrapped_model=wrapped_model,
        tokenizer=tokenizer,
    )
    return StaticInferenceEngine(text_generation_controller=controller)


def load_text_tokenizer() -> None:
    """Load the prompt tokenizer from local files only."""
    if RESOURCES["tokenizer"] is not None:
        return

    if not os.path.isdir(TOKENIZER_PATH):
        raise RuntimeError(f"Tokenizer directory not found: {TOKENIZER_PATH}")

    try:
        import transformers

        RESOURCES["tokenizer"] = transformers.AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"[Worker] Tokenizer loaded from {TOKENIZER_PATH}")
        return
    except Exception as exc:
        print(f"[Worker] AutoTokenizer load failed, trying fast tokenizer fallback: {exc}")

    try:
        from transformers import PreTrainedTokenizerFast

        tokenizer_file = os.path.join(TOKENIZER_PATH, "tokenizer.json")
        RESOURCES["tokenizer"] = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            bos_token=BOS,
            eos_token="<|end_of_text|>",
        )
        print(f"[Worker] Tokenizer loaded from tokenizer.json fallback: {tokenizer_file}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load tokenizer from {TOKENIZER_PATH}: {exc}") from exc


def load_backbone(name: str) -> None:
    """Load the backbone model and its static inference engine."""
    selected_name = default_backbone_name(name)
    if RESOURCES["backbone_name"] == selected_name and RESOURCES["backbone"] is not None:
        return

    free_resource("backbone_engine")
    free_resource("backbone")
    RESOURCES["backbone_name"] = None

    from megatron.core.inference.sampling_params import SamplingParams
    from megatron.training import get_args, get_model
    from megatron.training.checkpointing import load_checkpoint
    from model_provider import model_provider
    from gpt_builders import gpt_builder

    config = BACKBONE_MODELS[selected_name]
    args = get_args()
    args.load = config["path"]
    args.vocab_size = config["vocab_size"]
    args.padded_vocab_size = config["vocab_size"]

    model_list = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
    load_checkpoint(model_list, None, None, strict=False)
    model = model_list[0]
    model.eval()

    engine = get_backbone_inference_engine(model, config["vocab_size"])

    RESOURCES["backbone"] = model
    RESOURCES["backbone_engine"] = engine
    RESOURCES["backbone_name"] = selected_name
    print(f"[Worker] Backbone loaded: {selected_name}")

    if getattr(args, "enable_cuda_graph", False):
        print("[Worker] Running CUDA graph warmup for backbone.")
        engine.generate(
            prompts=[[0, 1, 2, 3]],
            sampling_params=SamplingParams(num_tokens_to_generate=8),
        )
        print("[Worker] Backbone warmup complete.")


def load_superres(name: str) -> None:
    """Load the super-resolution model."""
    selected_name = default_superres_name(name)
    if RESOURCES["superres_name"] == selected_name and RESOURCES["superres"] is not None:
        return

    free_resource("superres")
    RESOURCES["superres_name"] = None

    from megatron.core.transformer.enums import AttnMaskType
    from megatron.training import get_args, get_model
    from megatron.training.checkpointing import load_checkpoint
    from model_provider import model_provider
    from gpt_builders import gpt_builder

    config = SUPERRES_MODELS[selected_name]
    args = get_args()
    args.load = config["path"]
    args.vocab_size = config["vocab_size"]
    args.padded_vocab_size = config["vocab_size"]

    saved_flash_decode = getattr(args, "flash_decode", False)
    saved_enable_cuda_graph = getattr(args, "enable_cuda_graph", False)
    args.flash_decode = False
    args.enable_cuda_graph = False

    model_list = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
    load_checkpoint(model_list, None, None, strict=False)
    model = model_list[0]
    model.eval()

    args.flash_decode = saved_flash_decode
    args.enable_cuda_graph = saved_enable_cuda_graph

    inner_model = getattr(model, "module", model)
    for layer in inner_model.decoder.layers:
        layer.self_attention.attn_mask_type = AttnMaskType.padding

    RESOURCES["superres"] = model
    RESOURCES["superres_name"] = selected_name
    print(f"[Worker] Superres loaded: {selected_name}")


def load_decoder() -> None:
    """Load the DAC RVQ decoder."""
    if RESOURCES["decoder"] is not None:
        return

    from dac_rvq import DacRVQ
    from omegaconf import OmegaConf

    config = OmegaConf.load(DECODER_CONFIG_PATH)
    decoder = DacRVQ(config)

    checkpoint = torch.load(DECODER_CHECKPOINT_PATH, map_location="cuda", weights_only=False)
    generator_state_dict = {
        key[len("generator."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("generator.")
    }
    decoder.load_state_dict(generator_state_dict)
    decoder.to("cuda")
    decoder.eval()

    RESOURCES["decoder"] = decoder
    print("[Worker] Decoder loaded.")


# ============================================================
# Prompt preparation
# ============================================================


def compose_prompt_text(
    lyrics: str,
    genre: str,
    language: str,
    duration: int,
    tags: str,
    description: str,
) -> str:
    """Build the text prompt consumed by the tokenizer."""
    metadata_text = description.strip() or tags.strip() or genre

    if language != "Instrumental":
        lyrics_text = lyrics.strip().replace("\n", END_OF_LINE)
        return (
            f"{BOS}{metadata_text}{EOM}{language}{EOM}"
            f"{BOL}{END_OF_LINE}{lyrics_text}{EOM}"
            f"<|reserved_special_token_{duration}|>{EOT}{BOA}"
        )

    return (
        f"{BOS}{metadata_text}{EOM}{language}{EOM}"
        f"<|reserved_special_token_{duration}|>{EOT}{BOA}"
    )


def tokenize_prompt_text(prompt_text: str) -> list[int]:
    tokenizer = RESOURCES["tokenizer"]
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded.")

    return tokenizer.encode(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )[0].tolist()


def prepare_prompt_texts(
    genre: str,
    language: str,
    tags: str,
    description: str,
    duration: int,
    lyrics: str,
    superres_text_mode: str,
) -> tuple[str, str]:
    """Prepare backbone and superres prompt texts from a single user request."""
    if superres_text_mode == "same_as_backbone":
        backbone_language = language if language == "Instrumental" else ""
        backbone_prompt_text = compose_prompt_text(
            lyrics=lyrics,
            genre=genre,
            language=backbone_language,
            duration=duration,
            tags=tags,
            description=description,
        )
        superres_prompt_text = backbone_prompt_text
    elif superres_text_mode == "same_as_backbone_no_description":
        backbone_language = language if language == "Instrumental" else ""
        backbone_prompt_text = compose_prompt_text(
            lyrics=lyrics,
            genre=genre,
            language=backbone_language,
            duration=duration,
            tags=tags,
            description=description,
        )
        superres_tags = tags
        if description.strip():
            superres_tags = ""
        superres_prompt_text = compose_prompt_text(
            lyrics=lyrics,
            genre=genre,
            language=backbone_language,
            duration=duration,
            tags=superres_tags,
            description="",
        )
    else:
        backbone_prompt_text = compose_prompt_text(
            lyrics=lyrics,
            genre=genre,
            language=language,
            duration=duration,
            tags=tags,
            description=description,
        )
        superres_prompt_text = compose_prompt_text(
            lyrics=lyrics,
            genre=genre,
            language=language,
            duration=duration,
            tags="",
            description="",
        )

    return backbone_prompt_text, superres_prompt_text


def prepare_prompt_ids(
    genre: str,
    language: str,
    tags: str,
    description: str,
    duration: int,
    lyrics: str,
    superres_text_mode: str,
) -> tuple[list[int], list[int]]:
    """Prepare backbone and superres prompt ids from a single user request."""
    backbone_prompt_text, superres_prompt_text = prepare_prompt_texts(
        genre=genre,
        language=language,
        tags=tags,
        description=description,
        duration=duration,
        lyrics=lyrics,
        superres_text_mode=superres_text_mode,
    )
    backbone_prompt_ids = tokenize_prompt_text(backbone_prompt_text)
    if superres_prompt_text == backbone_prompt_text:
        superres_prompt_ids = backbone_prompt_ids
    else:
        superres_prompt_ids = tokenize_prompt_text(superres_prompt_text)

    backbone_prompt_ids = clamp_prompt_ids(
        backbone_prompt_ids,
        BACKBONE_MAX_PROMPT_LEN,
        "Backbone",
    )
    superres_prompt_ids = clamp_prompt_ids(
        superres_prompt_ids,
        SUPERRES_MAX_PROMPT_LEN,
        "Superres",
    )
    return backbone_prompt_ids, superres_prompt_ids


# ============================================================
# Backbone generation
# ============================================================


async def stream_backbone_generate(engine, prompt_ids, sampling_params, estimated_tokens):
    """Stream backbone decoding progress into the worker state."""
    request_id = engine.add_request(prompt=prompt_ids, sampling_params=sampling_params, streaming=True)
    stream_generator = engine.get_stream_generator(request_id)

    previous_length = 0

    async def collect_stream():
        nonlocal previous_length
        async for output in stream_generator:
            if hasattr(output, "generated_length"):
                current_length = (
                    int(output.generated_length.item())
                    if hasattr(output.generated_length, "item")
                    else int(output.generated_length)
                )
            else:
                current_length = previous_length + 1
            previous_length = current_length
            STATE.progress = min(99, int(current_length / max(estimated_tokens, 1) * 100))
            STATE.progress_detail = f"{current_length}/{estimated_tokens} tokens"

    collect_task = asyncio.create_task(collect_stream())
    await engine.run_engine_async()
    await collect_task
    STATE.progress = 100
    return engine.scheduler.completed_request_pool[request_id]


@torch.inference_mode()
def generate_backbone(prompt_ids: list[int], top_k: int, temperature: float, duration: int):
    """Run autoregressive backbone decoding and return q0/q1 tokens."""
    from megatron.core.inference.sampling_params import SamplingParams
    from megatron.training import get_args

    engine = RESOURCES["backbone_engine"]
    if engine is None:
        raise RuntimeError("Backbone engine not loaded.")

    args = get_args()
    sampling_params = SamplingParams(
        temperature=float(temperature),
        top_k=int(top_k),
        num_tokens_to_generate=args.num_tokens_to_generate,
    )
    estimated_tokens = round(TOKENS_PER_MINUTE * (float(duration) + 0.8))

    start_time = time.perf_counter()
    if getattr(args, "stream", False):
        result = asyncio.run(
            stream_backbone_generate(engine, prompt_ids, sampling_params, estimated_tokens)
        )
    else:
        result = engine.generate(prompts=[prompt_ids], sampling_params=sampling_params)[0]
        STATE.progress = 100
        STATE.progress_detail = ""

    elapsed = time.perf_counter() - start_time
    generated_tokens = result.generated_tokens.cpu().tolist()
    print(
        f"[Worker] Backbone generation finished in {elapsed:.1f}s "
        f"with {len(generated_tokens)} tokens."
    )
    return np.array(prompt_ids, dtype=np.int64), generated_tokens


# ============================================================
# Super-resolution generation
# ============================================================


def prepare_superres_inputs(text_tokens: np.ndarray, audio_tokens: np.ndarray, max_seq_len: int) -> dict:
    """Build the packed non-causal inputs expected by the superres model."""
    text_len = len(text_tokens)
    audio_tokens_2d = audio_tokens.reshape(-1, 2)
    audio_len = audio_tokens_2d.shape[0]

    tokens = torch.full((max_seq_len, 2), PAD_TOKEN_ID, dtype=torch.long)
    if text_len > 0:
        tokens[:text_len, 0] = torch.from_numpy(text_tokens.copy())
    tokens[text_len: text_len + audio_len] = torch.from_numpy(audio_tokens_2d.copy())

    attention_mask = torch.ones(max_seq_len, dtype=torch.bool)
    attention_mask[: text_len + audio_len] = False

    loss_mask = (tokens[:, -1] != PAD_TOKEN_ID).float()
    position_ids = torch.arange(max_seq_len, dtype=torch.long)

    return {
        "tokens": tokens.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        "loss_mask": loss_mask.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
        "text_len": text_len,
        "audio_len": audio_len,
    }


def generate_superres_manual_projection(
    model,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor,
    text_len: int,
    audio_len: int,
    top_k: int,
) -> torch.Tensor:
    """
    Expand q0/q1 into q0..q63 by projecting decoder hidden states only onto the
    current quantizer's 1024-token vocab window.
    """
    final_output_2d = torch.zeros(
        1,
        tokens.size(1),
        NUM_QUANTIZERS,
        device=tokens.device,
        dtype=tokens.dtype,
    )
    final_output_2d[..., :2] = tokens[..., :2]

    inner_model = getattr(model, "module", model)
    if inner_model.share_embeddings_and_output_weights:
        output_weight = inner_model.shared_embedding_or_output_weight()
    else:
        output_weight = inner_model.output_layer.weight

    total_steps = NUM_QUANTIZERS - 2
    for idx in range(2, NUM_QUANTIZERS):
        step = idx - 1
        STATE.progress = min(99, int(step / total_steps * 100))
        STATE.progress_detail = f"{step}/{total_steps} layers"

        final_output_2d[:, text_len - 1, 0] = TASK0_MARKER - idx
        hidden_states = inner_model.forward_hidden_states(
            input_ids=final_output_2d[..., :idx],
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        # GPTModel returns [seq, batch, hidden]. Convert to [batch, seq, hidden]
        # so we can project the active token window with a standard batched matmul.
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        min_id = VQ0_START_ID + idx * 1024
        max_id = min_id + 1024
        # Match Megatron's original output-layer dtype during projection, then upcast
        # the resulting logits to fp32 for numerically stable top-k / softmax sampling.
        output_slice = output_weight[min_id:max_id].to(hidden_states.dtype)
        logits = torch.matmul(hidden_states, output_slice.t())
        logits = logits.float()
        del hidden_states

        k = min(max(1, int(top_k)), logits.shape[-1])
        top_values, top_indices = torch.topk(logits, k, dim=-1)
        del logits

        probabilities = torch.softmax(top_values.float(), dim=-1)
        q = torch.empty_like(probabilities).exponential_(1.0)
        sampled_in_topk = torch.argmax(probabilities / q, dim=-1)
        sampled_tokens = (
            torch.gather(top_indices, -1, sampled_in_topk.unsqueeze(-1)).squeeze(-1) + min_id
        )
        sampled_tokens = torch.where(loss_mask.bool(), sampled_tokens, PAD_TOKEN_ID)
        final_output_2d[..., idx] = sampled_tokens.squeeze(-1)

    audio_tokens = final_output_2d[:, text_len: text_len + audio_len, :]
    return audio_tokens.permute(2, 0, 1).clone()


@torch.inference_mode()
def generate_superres(superres_prompt_ids: list[int], backbone_tokens: list[int], top_k: int) -> torch.Tensor:
    """Run the superres model from q0/q1 tokens to full 64-quantizer audio tokens."""
    model = RESOURCES["superres"]
    if model is None:
        raise RuntimeError("Superres model not loaded.")

    text_tokens = np.array(superres_prompt_ids, dtype=np.int64)
    audio_tokens = np.array(backbone_tokens, dtype=np.int64)

    text_len = len(text_tokens)
    audio_len = len(audio_tokens) // 2
    actual_seq_len = text_len + audio_len
    max_seq_len = 8192 if actual_seq_len < 8192 else actual_seq_len

    print(
        f"[Worker] Superres input: text_len={text_len}, audio_len={audio_len}, "
        f"max_seq_len={max_seq_len}"
    )

    input_dict = prepare_superres_inputs(text_tokens, audio_tokens, max_seq_len)
    tokens = input_dict["tokens"].to("cuda")
    attention_mask = input_dict["attention_mask"].to("cuda")
    loss_mask = input_dict["loss_mask"].to("cuda")
    position_ids = input_dict["position_ids"].to("cuda")

    start_time = time.perf_counter()
    audio_output = generate_superres_manual_projection(
        model,
        tokens,
        attention_mask,
        loss_mask,
        position_ids,
        text_len,
        audio_len,
        top_k,
    )
    torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - start_time
    print(f"[Worker] Superres finished in {elapsed_sec:.3f}s using manual_projection.")

    for quantizer_idx in range(NUM_QUANTIZERS):
        audio_output[quantizer_idx] -= (VQ0_START_ID + quantizer_idx * 1024)

    return audio_output


# ============================================================
# Decoder output
# ============================================================


@torch.inference_mode()
def decode_to_wav(audio_tokens: torch.Tensor, wav_path: str) -> str:
    """Decode quantized audio tokens into a waveform file."""
    import soundfile as sf

    decoder = RESOURCES["decoder"]
    if decoder is None:
        raise RuntimeError("Decoder not loaded.")

    decoder_device = next(decoder.parameters()).device
    chunk_size = DECODER_CHUNK_SIZE
    overlap_tokens = max(0, min(DECODER_CHUNK_OVERLAP, chunk_size - 1))
    stride = chunk_size - overlap_tokens if overlap_tokens > 0 else chunk_size

    waveform = None
    prev_token_len = 0
    prev_waveform_chunk_len = 0

    for start in range(0, audio_tokens.size(2), stride):
        token_chunk = audio_tokens[..., start: start + chunk_size]
        if token_chunk.device != decoder_device:
            token_chunk = token_chunk.to(decoder_device, non_blocking=True)
        waveform_chunk = decoder.decode(token_chunk).detach().cpu()
        current_token_len = token_chunk.size(2)
        current_waveform_chunk_len = waveform_chunk.size(2)

        if waveform is None or overlap_tokens == 0:
            waveform = waveform_chunk
            prev_token_len = current_token_len
            prev_waveform_chunk_len = current_waveform_chunk_len
            del token_chunk, waveform_chunk
            continue

        actual_overlap_tokens = min(overlap_tokens, prev_token_len, current_token_len)
        prev_overlap_samples = round(
            prev_waveform_chunk_len * actual_overlap_tokens / max(1, prev_token_len)
        )
        curr_overlap_samples = round(
            current_waveform_chunk_len * actual_overlap_tokens / max(1, current_token_len)
        )
        overlap_samples = min(prev_overlap_samples, curr_overlap_samples)

        if overlap_samples > 0:
            fade_out = torch.linspace(1.0, 0.0, overlap_samples, dtype=waveform.dtype).view(
                1, 1, -1
            )
            fade_in = 1.0 - fade_out
            blended = (
                waveform[:, :, -overlap_samples:] * fade_out
                + waveform_chunk[:, :, :overlap_samples] * fade_in
            )
            waveform = torch.cat(
                [
                    waveform[:, :, :-overlap_samples],
                    blended,
                    waveform_chunk[:, :, overlap_samples:],
                ],
                dim=2,
            )
        else:
            waveform = torch.cat([waveform, waveform_chunk], dim=2)

        prev_token_len = current_token_len
        prev_waveform_chunk_len = current_waveform_chunk_len
        del token_chunk, waveform_chunk

    waveform_np = waveform.squeeze(0).numpy().T
    sf.write(wav_path, waveform_np, DECODER_SAMPLE_RATE)
    return wav_path


def wav_to_mp3(wav_path: str, mp3_path: str) -> str:
    """Transcode a generated WAV file into MP3."""
    command = [
        "ffmpeg",
        "-y",
        "-i",
        wav_path,
        "-b:a",
        "320k",
        "-ar",
        str(DECODER_SAMPLE_RATE),
        mp3_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed with return code {result.returncode}:\n{result.stderr}"
        )
    return mp3_path


def write_outputs(
    request: GenerateRequest,
    backbone_name: str,
    superres_name: str,
    backbone_tokens_count: int,
    audio_tokens: torch.Tensor,
) -> dict:
    """Decode audio, write output files, and persist request metadata."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    wav_filename = f"{file_id}.wav"
    mp3_filename = f"{file_id}.mp3"
    meta_filename = f"{file_id}.json"

    wav_path = os.path.join(OUTPUT_DIR, wav_filename)
    mp3_path = os.path.join(OUTPUT_DIR, mp3_filename)
    meta_path = os.path.join(OUTPUT_DIR, meta_filename)

    decode_to_wav(audio_tokens, wav_path)
    try:
        wav_to_mp3(wav_path, mp3_path)
    except RuntimeError as exc:
        print(f"[Worker] MP3 conversion failed, falling back to WAV: {exc}")
        mp3_filename = wav_filename

    duration_sec = round(audio_tokens.size(2) / CODEC_FPS, 1)
    metadata = {
        "file_id": file_id,
        "superres_text_mode": request.superres_text_mode,
        "raw_user_input": request.raw_user_input,
        "raw_mode": request.raw_mode,
        "raw_prompt_mode": request.raw_prompt_mode,
        "model_genre": request.genre,
        "model_language": request.language,
        "model_tags": request.tags,
        "model_description": request.description,
        "duration_min": request.duration,
        "lyrics": request.lyrics,
        "backbone_name": backbone_name,
        "superres_name": superres_name,
        "top_k_bb": request.top_k_bb,
        "top_k_sr": request.top_k_sr,
        "temperature": request.temperature,
        "gpu_id": STATE.gpu_id,
        "seed": STATE.seed,
        "backbone_tokens": backbone_tokens_count,
        "duration_sec": duration_sec,
        "wav_file": wav_filename,
        "mp3_file": mp3_filename,
        "created_at": timestamp,
    }

    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "mp3_filename": mp3_filename,
        "wav_filename": wav_filename,
        "duration_sec": duration_sec,
        "backbone_tokens": backbone_tokens_count,
        "gpu_id": STATE.gpu_id,
        "seed": STATE.seed,
    }


# ============================================================
# Main generation flow
# ============================================================


def reset_request_state() -> None:
    set_status("idle")
    set_phase("idle", 0, "")
    clear_cuda_memory()


def release_request_models() -> None:
    """Release per-request GPU models in one-shot mode only."""
    if RUNTIME_MODE == "keep_loaded":
        return
    if RESOURCES["superres"] is not None:
        free_resource("superres")
        RESOURCES["superres_name"] = None
        print("[Worker] Released superres after request.")
    if RESOURCES["decoder"] is not None:
        free_resource("decoder")
        print("[Worker] Released decoder after request.")


def release_superres_for_stage_transition() -> None:
    """Release superres right after q0..q63 generation to free space for decoder."""
    if RESOURCES["superres"] is not None:
        free_resource("superres")
        RESOURCES["superres_name"] = None
        print("[Worker] Released superres before decoder.")


def release_backbone_for_stage_transition() -> None:
    """Release backbone right after q0/q1 generation when the process is one-shot."""
    if RESOURCES["backbone_engine"] is not None:
        free_resource("backbone_engine")
    if RESOURCES["backbone"] is not None:
        free_resource("backbone")
        RESOURCES["backbone_name"] = None
        print("[Worker] Released backbone before superres.")


def filtered_child_args(argv: list[str]) -> list[str]:
    """Forward Megatron/model args to a one-shot child while dropping service-only flags."""
    skip_with_value = {
        "--worker-port",
        "--runtime-mode",
        "--child-stage",
        "--artifact-dir",
        "--request-json",
        "--result-json",
        "--status-json",
        "--seed",
    }
    skip_flags = {"--child-once"}

    forwarded: list[str] = []
    idx = 1
    while idx < len(argv):
        token = argv[idx]
        if token in skip_flags:
            idx += 1
            continue
        if token in skip_with_value:
            idx += 2
            continue
        forwarded.append(token)
        idx += 1
    return forwarded


def sync_state_from_file(path: str) -> None:
    """Mirror child progress back into the parent worker state."""
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return

    STATE.status = data.get("status", STATE.status)
    STATE.phase = data.get("phase", STATE.phase)
    STATE.progress = data.get("progress", STATE.progress)
    STATE.progress_detail = data.get("progress_detail", STATE.progress_detail)
    STATE.seed = data.get("seed", STATE.seed)
    STATE.gpu_id = data.get("gpu_id", STATE.gpu_id)


def build_one_shot_command(
    stage: str,
    artifact_dir: str,
    request_path: str,
    result_path: str,
    status_path: str,
    seed_override: int,
) -> list[str]:
    """Construct the subprocess command used by one-shot workers."""
    return [
        sys.executable,
        os.path.abspath(__file__),
        *filtered_child_args(sys.argv),
        "--seed",
        str(seed_override),
        "--runtime-mode",
        "one_shot",
        "--child-once",
        "--child-stage",
        stage,
        "--artifact-dir",
        artifact_dir,
        "--request-json",
        request_path,
        "--result-json",
        result_path,
        "--status-json",
        status_path,
    ]


def one_shot_stage_paths(artifact_dir: str) -> dict[str, str]:
    return {
        "backbone_result": os.path.join(artifact_dir, "backbone_result.json"),
        "superres_result": os.path.join(artifact_dir, "superres_result.json"),
        "superres_tokens_npy": os.path.join(artifact_dir, "superres_tokens.npy"),
        "decoder_result": os.path.join(artifact_dir, "decoder_result.json"),
    }


def run_backbone_stage(request: GenerateRequest) -> dict:
    """Run only the backbone stage and persist q0/q1 tokens for the next child."""
    set_status("busy")
    set_phase("loading_backbone", 0, "")
    load_text_tokenizer()

    backbone_name = default_backbone_name(request.backbone_name)
    superres_name = default_superres_name(request.superres_name)
    backbone_prompt_text, superres_prompt_text = prepare_prompt_texts(
        genre=request.genre,
        language=request.language,
        tags=request.tags,
        description=request.description,
        duration=int(request.duration),
        lyrics=request.lyrics,
        superres_text_mode=request.superres_text_mode,
    )
    backbone_prompt_ids, superres_prompt_ids = prepare_prompt_ids(
        genre=request.genre,
        language=request.language,
        tags=request.tags,
        description=request.description,
        duration=int(request.duration),
        lyrics=request.lyrics,
        superres_text_mode=request.superres_text_mode,
    )

    load_backbone(backbone_name)
    set_phase("backbone", 0, "")
    _, backbone_tokens = generate_backbone(
        prompt_ids=backbone_prompt_ids,
        top_k=int(request.top_k_bb),
        temperature=float(request.temperature),
        duration=int(request.duration),
    )
    return {
        "status": "ok",
        "backbone_name": backbone_name,
        "superres_name": superres_name,
        "backbone_tokens": backbone_tokens,
        "backbone_tokens_count": len(backbone_tokens),
        "superres_prompt_ids": superres_prompt_ids,
    }


def run_superres_stage(request: GenerateRequest, artifact_dir: str) -> dict:
    """Run only the superres stage and persist q0..q63 tokens for decoder."""
    paths = one_shot_stage_paths(artifact_dir)
    with open(paths["backbone_result"], "r", encoding="utf-8") as file:
        backbone_result = json.load(file)

    set_status("busy")
    set_phase("loading_superres", 0, "")
    superres_name = backbone_result["superres_name"]
    superres_prompt_ids = backbone_result["superres_prompt_ids"]
    backbone_tokens = backbone_result["backbone_tokens"]

    load_superres(superres_name)
    set_phase("superres", 0, "")
    audio_tokens = generate_superres(
        superres_prompt_ids=superres_prompt_ids,
        backbone_tokens=backbone_tokens,
        top_k=int(request.top_k_sr),
    )
    audio_tokens_cpu = audio_tokens.cpu()
    del audio_tokens
    clear_cuda_memory()

    audio_tokens_np = audio_tokens_cpu.numpy()
    np.save(paths["superres_tokens_npy"], audio_tokens_np)
    return {
        "status": "ok",
        "superres_name": superres_name,
        "audio_q0_63_npy": paths["superres_tokens_npy"],
        "audio_q0_63_shape": list(audio_tokens_np.shape),
        "audio_q0_63_dtype": str(audio_tokens_np.dtype),
    }


def run_decoder_stage(request: GenerateRequest, artifact_dir: str) -> dict:
    """Run only the decoder stage from persisted q0..q63 tokens."""
    paths = one_shot_stage_paths(artifact_dir)
    with open(paths["backbone_result"], "r", encoding="utf-8") as file:
        backbone_result = json.load(file)
    with open(paths["superres_result"], "r", encoding="utf-8") as file:
        superres_result = json.load(file)

    set_status("busy")
    set_phase("loading_decoder", 0, "")
    load_decoder()
    set_phase("decoding", 0, "decoding audio")

    audio_tokens_np = np.load(superres_result["audio_q0_63_npy"])
    audio_tokens = torch.from_numpy(audio_tokens_np)
    return write_outputs(
        request=request,
        backbone_name=backbone_result["backbone_name"],
        superres_name=backbone_result["superres_name"],
        backbone_tokens_count=int(backbone_result["backbone_tokens_count"]),
        audio_tokens=audio_tokens,
    )


def run_child_stage(
    stage: str,
    request_json: str,
    result_json: str,
    status_json: str,
    artifact_dir: str,
) -> int:
    """Child-process entrypoint for one one-shot stage."""
    stop_event, status_thread = start_status_writer(status_json)
    try:
        with open(request_json, "r", encoding="utf-8") as file:
            request_payload = json.load(file)
        request = GenerateRequest(**request_payload)

        if stage == "backbone":
            result = run_backbone_stage(request)
        elif stage == "superres":
            result = run_superres_stage(request, artifact_dir)
        elif stage == "decoder":
            result = run_decoder_stage(request, artifact_dir)
        else:
            raise ValueError(f"Unsupported child stage: {stage}")

        with open(result_json, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        return 0 if result.get("status") == "ok" else 1
    except Exception:
        error_text = traceback.format_exc()
        with open(result_json, "w", encoding="utf-8") as file:
            json.dump({"status": "error", "error": error_text}, file, ensure_ascii=False, indent=2)
        return 1
    finally:
        if stop_event is not None:
            stop_event.set()
        if status_thread is not None:
            status_thread.join(timeout=2.0)


def run_generation_one_shot(request: GenerateRequest) -> dict:
    """Execute one request in a short-lived child process for process-level VRAM cleanup."""
    with STATE_LOCK:
        set_status("busy")
        set_phase("loading_backbone", 0, "")

        with tempfile.TemporaryDirectory(prefix="khala_worker_", dir=BACKEND_DIR) as temp_dir:
            request_path = os.path.join(temp_dir, "request.json")
            paths = one_shot_stage_paths(temp_dir)

            with open(request_path, "w", encoding="utf-8") as file:
                json.dump(request.model_dump(), file, ensure_ascii=False, indent=2)

            child_seed = int(request.seed_override) if int(request.seed_override) > 0 else int(STATE.seed)
            stages = [
                ("backbone", paths["backbone_result"]),
                ("superres", paths["superres_result"]),
                ("decoder", paths["decoder_result"]),
            ]
            loading_phase_by_stage = {
                "backbone": "loading_backbone",
                "superres": "loading_superres",
                "decoder": "loading_decoder",
            }

            try:
                for stage_name, result_path in stages:
                    set_phase(loading_phase_by_stage[stage_name], 0, "")
                    status_path = os.path.join(temp_dir, f"{stage_name}_status.json")
                    log_path = os.path.join(temp_dir, f"{stage_name}.log")
                    command = build_one_shot_command(
                        stage_name,
                        temp_dir,
                        request_path,
                        result_path,
                        status_path,
                        child_seed,
                    )
                    print(f"[Worker] Launching one-shot {stage_name} child: {' '.join(command)}")

                    env = os.environ.copy()
                    with open(log_path, "w", encoding="utf-8") as log_file:
                        proc = subprocess.Popen(
                            command,
                            cwd=BACKEND_DIR,
                            env=env,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )

                        while proc.poll() is None:
                            sync_state_from_file(status_path)
                            time.sleep(0.5)

                        sync_state_from_file(status_path)

                    if os.path.isfile(result_path):
                        with open(result_path, "r", encoding="utf-8") as file:
                            result = json.load(file)
                    else:
                        with open(log_path, "r", encoding="utf-8") as file:
                            log_text = file.read()[-4000:]
                        return {
                            "status": "error",
                            "error": (
                                f"one_shot {stage_name} child exited with code {proc.returncode} "
                                f"without a result file.\n{log_text}"
                            ),
                        }

                    if result.get("status") != "ok":
                        return result

                with open(paths["decoder_result"], "r", encoding="utf-8") as file:
                    return json.load(file)
            finally:
                set_status("idle")
                set_phase("idle", 0, "")
                STATE.progress_detail = ""
                STATE.progress = 0


def run_generation(request: GenerateRequest) -> dict:
    """Main request path executed by the worker's /generate endpoint."""
    with STATE_LOCK:
        set_status("busy")
        set_phase("loading_backbone", 0, "")
        request_started_at = time.perf_counter()

        try:
            print(
                "[Worker] Starting request "
                f"mode={request.raw_mode or 'unknown'} "
                f"prompt_mode={request.raw_prompt_mode or 'unknown'} "
                f"duration={request.duration}min"
            )

            load_text_tokenizer()

            backbone_name = default_backbone_name(request.backbone_name)
            superres_name = default_superres_name(request.superres_name)
            backbone_prompt_text, superres_prompt_text = prepare_prompt_texts(
                genre=request.genre,
                language=request.language,
                tags=request.tags,
                description=request.description,
                duration=int(request.duration),
                lyrics=request.lyrics,
                superres_text_mode=request.superres_text_mode,
            )
            backbone_prompt_ids, superres_prompt_ids = prepare_prompt_ids(
                genre=request.genre,
                language=request.language,
                tags=request.tags,
                description=request.description,
                duration=int(request.duration),
                lyrics=request.lyrics,
                superres_text_mode=request.superres_text_mode,
            )

            load_backbone(backbone_name)
            set_phase("backbone", 0, "")
            _, backbone_tokens = generate_backbone(
                prompt_ids=backbone_prompt_ids,
                top_k=int(request.top_k_bb),
                temperature=float(request.temperature),
                duration=int(request.duration),
            )
            if RUNTIME_MODE == "one_shot":
                release_backbone_for_stage_transition()

            set_phase("loading_superres", 0, "")
            load_superres(superres_name)
            set_phase("superres", 0, "")
            audio_tokens = generate_superres(
                superres_prompt_ids=superres_prompt_ids,
                backbone_tokens=backbone_tokens,
                top_k=int(request.top_k_sr),
            )
            audio_tokens_cpu = audio_tokens.cpu()
            del audio_tokens
            clear_cuda_memory()
            if RUNTIME_MODE == "one_shot":
                release_superres_for_stage_transition()

            set_phase("loading_decoder", 0, "")
            load_decoder()
            set_phase("decoding", 0, "decoding audio")
            return write_outputs(
                request=request,
                backbone_name=backbone_name,
                superres_name=superres_name,
                backbone_tokens_count=len(backbone_tokens),
                audio_tokens=audio_tokens_cpu,
            )
        except Exception:
            error_text = traceback.format_exc()
            print(f"[Worker] Generation failed:\n{error_text}")
            return {"status": "error", "error": error_text}
        finally:
            elapsed = time.perf_counter() - request_started_at
            release_request_models()
            reset_request_state()
            print(f"[Worker] Request finished in {elapsed:.3f}s.")


# ============================================================
# HTTP endpoints
# ============================================================


@app.get("/health")
def health():
    return state_snapshot()


@app.get("/config")
def config():
    return {
        "backbone_models": list(BACKBONE_MODELS.keys()),
        "superres_models": list(SUPERRES_MODELS.keys()),
        "genre_options": GENRE_OPTIONS,
        "language_options": LANGUAGE_OPTIONS,
        "runtime_mode": RUNTIME_MODE,
    }


@app.post("/generate")
def api_generate(request: GenerateRequest):
    if STATE.status == "busy":
        return {"status": "busy", "error": "Worker is currently busy."}
    if RUNTIME_MODE == "one_shot":
        return run_generation_one_shot(request)
    return run_generation(request)


@app.get("/download/{filename}")
def download(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(filepath):
        return {"status": "error", "error": f"File not found: {filename}"}
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return FileResponse(filepath, media_type=media_type, filename=filename)


# ============================================================
# Bootstrap
# ============================================================


def preload_runtime() -> None:
    """Warm all long-lived runtime pieces during process startup."""
    load_text_tokenizer()
    load_backbone(default_backbone_name(""))
    load_superres(default_superres_name(""))
    load_decoder()


def cli_value(argv: list[str], flag: str, default: str = "") -> str:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
    return default


def cli_has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def main() -> None:
    import sys as _sys
    import uvicorn
    from megatron.training import get_args
    from megatron.training.initialize import initialize_megatron

    global RUNTIME_MODE

    worker_port = int(cli_value(_sys.argv, "--worker-port", "8001"))
    runtime_mode = cli_value(_sys.argv, "--runtime-mode", "one_shot") or "one_shot"
    child_stage = cli_value(_sys.argv, "--child-stage", "")
    artifact_dir = cli_value(_sys.argv, "--artifact-dir", "")
    request_json = cli_value(_sys.argv, "--request-json", "")
    result_json = cli_value(_sys.argv, "--result-json", "")
    status_json = cli_value(_sys.argv, "--status-json", "")
    child_once = cli_has_flag(_sys.argv, "--child-once")
    RUNTIME_MODE = runtime_mode

    STATE.gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    STATE.seed = int(cli_value(_sys.argv, "--seed", "0") or 0)

    if runtime_mode == "one_shot" and not child_once:
        set_status("idle")
        set_phase("idle", 0, "")
        print(f"[Worker GPU {STATE.gpu_id}] Starting one-shot shell on http://0.0.0.0:{worker_port}")
        uvicorn.run(app, host="0.0.0.0", port=worker_port, log_level="warning")
        return

    if child_once and child_stage == "decoder":
        exit_code = run_child_stage(child_stage, request_json, result_json, status_json, artifact_dir)
        raise SystemExit(exit_code)

    patch_language_model_embedding()

    first_backbone = next(iter(BACKBONE_MODELS.values()))
    if "--load" not in _sys.argv:
        _sys.argv.extend(["--load", first_backbone["path"]])
    if "--vocab-size" not in _sys.argv:
        _sys.argv.extend(["--vocab-size", str(first_backbone["vocab_size"])])
    if "--use-checkpoint-args" not in _sys.argv:
        _sys.argv.append("--use-checkpoint-args")

    print(
        f"[Worker GPU {STATE.gpu_id}] bootstrap args: "
        f"--load {first_backbone['path']} "
        f"--vocab-size {first_backbone['vocab_size']} --use-checkpoint-args"
    )

    initialize_megatron(
        extra_args_provider=add_worker_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
        },
    )

    args = get_args()
    STATE.seed = args.seed
    print(f"[Worker GPU {STATE.gpu_id}] Megatron initialized. seed={args.seed}")

    if child_once:
        exit_code = run_child_stage(child_stage, request_json, result_json, status_json, artifact_dir)
        raise SystemExit(exit_code)

    try:
        if runtime_mode == "keep_loaded":
            preload_runtime()
    except Exception as exc:
        print(f"[Worker GPU {STATE.gpu_id}] WARNING: preload failed: {exc}")

    set_status("idle")
    set_phase("idle", 0, "")
    print(f"[Worker GPU {STATE.gpu_id}] Starting on http://0.0.0.0:{worker_port}")
    uvicorn.run(app, host="0.0.0.0", port=worker_port, log_level="warning")


if __name__ == "__main__":
    main()
