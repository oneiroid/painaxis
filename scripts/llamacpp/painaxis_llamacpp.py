"""Shared paths and helpers for the llama.cpp steering scripts.

Every path can be overridden with an environment variable, so the scripts run
on another machine without edits.
"""

import os
import struct
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]

# The llama.cpp checkout and its build. The CUDA build is separate from the
# CPU one so both stay usable.
LLAMA_CPP = Path(os.environ.get(
    "LLAMA_CPP", "/media/oneiroid/sub/workspace/llmfinetune/vendor/llama.cpp"))
LLAMA_BUILD = Path(os.environ.get("LLAMA_BUILD", LLAMA_CPP / "build-cuda"))
LLAMA_CLI = LLAMA_BUILD / "bin" / "llama-cli"

MODEL_GGUF = Path(os.environ.get(
    "MODEL_GGUF", "/media/oneiroid/sub/workspace/llmfinetune/models/Qwen3-8B-Q8_0.gguf"))

# Name of the folder under results/3.2_pain_vectors/pain_vectors/ whose vectors
# match MODEL_GGUF. "Qwen_3_8B_base" is the paper's name for Qwen/Qwen3-8B.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen_3_8B_base")

# How many decoder layers to put on the GPU. 8 GB of Q8_0 weights do not fit in
# 6 GB of VRAM, so this is a partial offload.
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "20"))

VECTORS_PT = REPO / "results" / "3.2_pain_vectors" / "pain_vectors" / MODEL_NAME / "pain_vectors.pt"
DATASET = REPO / "datasets" / "3.1_pain_and_control_datasets.json"
OUT_DIR = REPO / "results" / "llamacpp" / MODEL_NAME
DUMPER = Path(__file__).resolve().parent / "tools" / "dump_activations"

PAIN_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]

CATEGORY_LABELS = {
    "A1": "Physical Pain", "A2": "Psychological", "A3": "Social Pain",
    "A4": "Moral Injury", "A5": "Cognitive Pain",
    "B": "Fear", "C1": "Neg Emotion", "C2": "Neg World",
    "D": "Neutral", "E": "Body Sensation",
}


def load_vectors():
    """The paper's pain vectors for MODEL_NAME, as float32 numpy arrays.

    weights_only=False because the file stores the layer index as a numpy
    scalar; the file is part of this repository.
    """
    d = torch.load(VECTORS_PT, map_location="cpu", weights_only=False)
    return {
        "s1": d["s1_pain_vector"].float().numpy(),
        "s2": d["s2_pain_vector"].float().numpy(),
        "layer": int(d["layer"]),
        "extraction": d["extraction"],
    }


def read_activations(path):
    """Read a dump_activations file -> (array[n_prompts, n_layers, n_embd], is_mean)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"PXACT001":
            raise ValueError(f"{path}: bad magic {magic!r}")
        n_prompts, n_layers, n_embd, is_mean = struct.unpack("<4i", f.read(16))
        data = np.fromfile(f, dtype=np.float32, count=n_prompts * n_layers * n_embd)
    return data.reshape(n_prompts, n_layers, n_embd), bool(is_mean)
