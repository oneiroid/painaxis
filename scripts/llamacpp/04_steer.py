"""Steering ladder against the pain direction, run through llama.cpp.

The paper's ladder sweeps -2 to +3. This one defaults to the negative half: the
question here is what the direction opposite the pain vector does to the output,
and there is no reason to push a model along the pain axis to find that out.
--include-positive adds small positive rungs for anyone who wants the full
ladder.

Two controls run at the same negative coefficients, so a change at coeff -2 can
be told apart from what any perturbation of that size would do:
  shuffled  the pain vector's components permuted -- same norm, same marginal
            distribution of component sizes, no direction
  random    an isotropic gaussian direction scaled to the pain vector's norm

Reads the .bin files from 03. Writes results/llamacpp/<model>/steering/.
"""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from painaxis_llamacpp import (  # noqa: E402
    MODEL_GGUF, MODEL_NAME, N_GPU_LAYERS, OUT_DIR, REPO)

TOOL = Path(__file__).resolve().parent / "tools" / "steer_generate"
CV_DIR = OUT_DIR / "control_vectors"
OUT = OUT_DIR / "steering"
LADDER_SCRIPT = REPO / "scripts" / "4.2_steering" / "01_steering_ladder.py"

NEGATIVE_LADDER = [0, -0.5, -1, -1.5, -2, -3]
POSITIVE_RUNGS = [0.5, 1.0]
SEED = 42


def neutral_50():
    tree = ast.parse(LADDER_SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "NEUTRAL_50" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"NEUTRAL_50 not found in {LADDER_SCRIPT}")


def read_bin(path):
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    return np.frombuffer(raw[4:4 + 4 * n], dtype=np.float32).copy()


def write_bin(path, v):
    path.write_bytes(np.int32(v.size).tobytes() + v.astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", default="s2", choices=["s1", "s2"])
    ap.add_argument("--n-prompts", type=int, default=20,
                    help="how many of the 50 neutral prompts to use")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--include-positive", action="store_true",
                    help="also run small positive rungs, i.e. toward pain")
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((CV_DIR / "manifest.json").read_text())
    spec = manifest["vectors"][args.vector]
    layer = spec["layer"]

    OUT.mkdir(parents=True, exist_ok=True)
    prompts = neutral_50()[:args.n_prompts]
    prompts_txt = OUT / f"neutral{args.n_prompts}.txt"
    prompts_txt.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    coeffs = list(NEGATIVE_LADDER)
    if args.include_positive:
        coeffs += POSITIVE_RUNGS

    pain = read_bin(CV_DIR / spec["bin"])
    conditions = [("pain", CV_DIR / spec["bin"], coeffs)]

    if not args.no_controls:
        rng = np.random.default_rng(SEED)
        neg_only = [c for c in coeffs if c < 0]

        shuffled = pain[rng.permutation(pain.size)]
        p = OUT / f"control_shuffled_{args.vector}_L{layer}.bin"
        write_bin(p, shuffled)
        conditions.append(("shuffled", p, neg_only))

        rand = rng.standard_normal(pain.size).astype(np.float32)
        rand *= np.linalg.norm(pain) / np.linalg.norm(rand)
        p = OUT / f"control_random_{args.vector}_L{layer}.bin"
        write_bin(p, rand)
        conditions.append(("random", p, neg_only))

    print(f"model:   {MODEL_GGUF.name}")
    print(f"vector:  {args.vector.upper()}, layer {layer}, norm {np.linalg.norm(pain):.1f}")
    print(f"prompts: {len(prompts)}   tokens: {args.max_tokens}   greedy")
    print(f"ladder:  {coeffs}")
    if not args.include_positive:
        print("         (negative half only; pass --include-positive for the rest)")

    for cond, vec_path, cond_coeffs in conditions:
        out_jsonl = OUT / f"{MODEL_NAME}_{args.vector}_L{layer}_{cond}.jsonl"
        print(f"\n[{cond}] {len(cond_coeffs)} coefficients x {len(prompts)} prompts")
        subprocess.run([
            str(TOOL), "-m", str(MODEL_GGUF), "-f", str(prompts_txt),
            "--vector", str(vec_path), "--layer", str(layer),
            "--coeffs", ",".join(str(c) for c in cond_coeffs),
            "-o", str(out_jsonl), "-n", str(args.max_tokens),
            "-ngl", str(N_GPU_LAYERS), "-c", "1024",
        ], check=True)

    (OUT / "run.json").write_text(json.dumps({
        "model": MODEL_NAME, "gguf": str(MODEL_GGUF), "vector": args.vector,
        "layer": layer, "coeffs": coeffs, "n_prompts": len(prompts),
        "max_tokens": args.max_tokens, "seed": SEED,
        "conditions": [c for c, _, _ in conditions],
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
