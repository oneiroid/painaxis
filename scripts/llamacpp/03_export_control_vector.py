"""Turn a pain vector into something llama.cpp can steer with.

Picks the steering layer the way scripts/4.2_steering/01 does -- the candidate
layer whose ratio of vector norm to mean final-token residual norm is closest to
0.6 -- but measures the residual norms through llama.cpp instead of HuggingFace,
and prints the layer the paper recorded for this model next to it.

Writes, per vector:
  <name>_L<layer>.bin    raw direction for tools/steer_generate
  <name>_L<layer>.gguf   control vector for llama-cli --control-vector-scaled

Writes results/llamacpp/<model>/control_vectors/.
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from painaxis_llamacpp import (  # noqa: E402
    DUMPER, MODEL_GGUF, MODEL_NAME, N_GPU_LAYERS, OUT_DIR, REPO, load_vectors,
    read_activations)

RATIO_TARGET = 0.6
LADDER_SCRIPT = REPO / "scripts" / "4.2_steering" / "01_steering_ladder.py"
STEER_RESULTS = REPO / "results" / "4.2_steering"
OUT = OUT_DIR / "control_vectors"


def neutral_50():
    """The 50 neutral prompts, read out of the paper's steering script."""
    tree = ast.parse(LADDER_SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "NEUTRAL_50" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"NEUTRAL_50 not found in {LADDER_SCRIPT}")


def paper_layers():
    """Steering layer per vector tag, from the shipped result file names."""
    out = {}
    for tag in ("S1", "S2"):
        for p in (STEER_RESULTS / tag).glob(f"{MODEL_NAME}_steering_{tag}_*_L*.csv"):
            m = re.search(r"_L(\d+)\.csv$", p.name)
            if m:
                out[tag.lower()] = int(m.group(1))
    return out


def residual_norms():
    """Mean final-token residual norm per layer over the 50 neutral prompts."""
    OUT.mkdir(parents=True, exist_ok=True)
    bin_path = OUT / "neutral50_acts.bin"
    if not bin_path.exists():
        txt = OUT / "neutral50.txt"
        txt.write_text("\n".join(neutral_50()) + "\n", encoding="utf-8")
        subprocess.run([str(DUMPER), "-m", str(MODEL_GGUF), "-f", str(txt),
                        "-o", str(bin_path), "-ngl", str(N_GPU_LAYERS)], check=True)
    acts, _ = read_activations(bin_path)
    return np.linalg.norm(acts, axis=2).mean(axis=0)   # [n_layers]


def write_gguf(path, direction, layer, model_hint):
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "controlvector")
    w.add_string("controlvector.model_hint", model_hint)
    w.add_int32("controlvector.layer_count", layer)
    # llama.cpp maps direction.N onto the residual stream after block N.
    w.add_tensor(f"direction.{layer}", direction.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vecs = load_vectors()
    norms = residual_norms()
    n_layers = len(norms)
    recorded = paper_layers()

    # Same candidate set as the paper: fixed fractions of depth, plus the
    # extraction layer and the last layer.
    candidates = sorted({int(n_layers * f) for f in (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)}
                        | {vecs["layer"], n_layers - 1})
    candidates = [L for L in candidates if 1 <= L < n_layers]

    manifest = {"model": MODEL_NAME, "gguf": str(MODEL_GGUF),
                "extraction_layer": vecs["layer"], "n_layers": n_layers,
                "ratio_target": RATIO_TARGET, "vectors": {}}

    for name in ("s1", "s2"):
        v = vecs[name]
        vnorm = float(np.linalg.norm(v))
        ratios = {L: vnorm / float(norms[L]) for L in candidates}
        picked = min(candidates, key=lambda L: abs(ratios[L] - RATIO_TARGET))

        print(f"\n{name.upper()} vector, norm {vnorm:.1f}, extracted at layer {vecs['layer']}")
        print(f"  {'layer':>6} {'depth':>7} {'|v|/|resid|':>13}")
        for L in candidates:
            mark = "  <- picked" if L == picked else ""
            print(f"  {L:>6} {L / n_layers:>7.2f} {ratios[L]:>13.3f}{mark}")
        if name in recorded:
            print(f"  paper used layer {recorded[name]}"
                  f" (ratio here {ratios.get(recorded[name], float('nan')):.3f})")

        # Prefer the layer the paper actually ran, so the ladder is comparable.
        layer = recorded.get(name, picked)
        if layer != picked:
            print(f"  using the paper's layer {layer}, not the auto-picked {picked}")

        stem = f"{MODEL_NAME}_{name}_L{layer}"
        (OUT / f"{stem}.bin").write_bytes(
            np.int32(v.size).tobytes() + v.astype(np.float32).tobytes())
        write_gguf(OUT / f"{stem}.gguf", v, layer, "qwen3")

        manifest["vectors"][name] = {
            "layer": layer, "auto_picked_layer": picked,
            "paper_layer": recorded.get(name), "norm": vnorm,
            "ratio_at_layer": ratios.get(layer),
            "bin": f"{stem}.bin", "gguf": f"{stem}.gguf",
        }
        print(f"  wrote {stem}.bin and {stem}.gguf")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUT}")
    print("\nSteer with stock llama-cli, for example one step against the pain direction:")
    g = manifest["vectors"]["s2"]
    print(f"  {MODEL_GGUF.name}  --control-vector-scaled {g['gguf']}:-1.0")


if __name__ == "__main__":
    main()
