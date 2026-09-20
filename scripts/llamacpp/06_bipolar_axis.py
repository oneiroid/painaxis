"""Build an axis whose negative end means something, and compare it to the
negated pain vector.

The pain vectors are mean(pain) - mean(controls), where the controls are fear,
negative emotion, negative world states, neutral statements and non-painful body
sensations. Negating that does not produce the opposite of pain, because the
control pole was never a pole -- it was an absence.

This script builds mean(pain) - mean(arousal) at the same layer from the
llama.cpp activations that 02 already dumped. Arousal_1P is the dataset's
high-intensity positive material ("I cross the finish line first"), so the two
ends of this axis are an affective pair the way sadness and happiness are in a
face-attribute direction.

Then it reads out both axes at both poles through the unembedding, and writes
the new axis as a control vector so 04 can steer with it.

Requires 02 to have run. Writes results/llamacpp/<model>/bipolar/.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from painaxis_llamacpp import (  # noqa: E402
    DATASET, MODEL_NAME, OUT_DIR, PAIN_CATEGORIES, load_vectors, read_activations)

import importlib  # noqa: E402
_inspect = importlib.import_module("01_inspect_vector")

DENOISE_VARIANCE = 0.5   # same as scripts/3.2_pain_vectors/01
TOP_N = 25
OUT = OUT_DIR / "bipolar"
VALIDATION = OUT_DIR / "validation"


def diff_in_means(pos, neg, denoise=True):
    """mean(pos) - mean(neg), with the top principal components of neg removed.

    Same construction as compute_pain_vector in scripts/3.2_pain_vectors/01,
    which is what makes the result comparable to the shipped vectors.
    """
    neg_mean = neg.mean(axis=0)
    vec = pos.mean(axis=0) - neg_mean
    if denoise and len(neg) > 1:
        pca = PCA().fit(neg - neg_mean)
        cum = np.cumsum(pca.explained_variance_ratio_)
        n_comp = min(int(np.searchsorted(cum, DENOISE_VARIANCE)) + 1, len(pca.components_))
        for d in pca.components_[:n_comp]:
            vec = vec - np.dot(vec, d) * d
    return vec


def acts_at(name, layer, tag="nobos"):
    path = VALIDATION / f"{name}_{tag}.bin"
    if not path.exists():
        raise SystemExit(f"missing {path}; run 02_validate_vector.py first")
    acts, _ = read_activations(path)
    return acts[:, layer, :]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vecs = load_vectors()
    layer = vecs["layer"]
    data = json.loads(DATASET.read_text(encoding="utf-8"))["datasets"]

    tag = "nobos"
    if not (VALIDATION / f"S2_1P_{tag}.bin").exists():
        tag = "bos"

    s2_acts = acts_at("S2_1P", layer, tag)
    s2_cats = np.array([s["category"] for s in data["S2_1P"]["sentences"]])
    arousal = acts_at("Arousal_1P", layer, tag)
    pain_acts = s2_acts[np.isin(s2_cats, PAIN_CATEGORIES)]

    bipolar = diff_in_means(pain_acts, arousal)
    s2 = vecs["s2"]

    cos = float(bipolar @ s2 / (np.linalg.norm(bipolar) * np.linalg.norm(s2)))
    print(f"pain - arousal axis at layer {layer} ({tag}, read through llama.cpp)")
    print(f"  norm {np.linalg.norm(bipolar):.1f}   "
          f"shipped S2 vector norm {np.linalg.norm(s2):.1f}")
    print(f"  cos(pain-arousal, S2 pain vector) = {cos:.4f}")

    vocab, w_u, gain, _ = _inspect.load_unembedding()
    if gain is None:
        gain = np.ones(w_u.shape[1], dtype=np.float32)

    readout = {}
    for name, v in [("s2_pain_minus_controls", s2), ("pain_minus_arousal", bipolar)]:
        unit = (v / np.linalg.norm(v)).astype(np.float32)
        logits = w_u @ (unit * gain)
        order = np.argsort(logits)
        top = [_inspect.clean(vocab[i]) for i in order[::-1][:TOP_N]]
        bot = [_inspect.clean(vocab[i]) for i in order[:TOP_N]]
        readout[name] = {"positive": top, "negative": bot}
        print(f"\n  {name}")
        print(f"    + : {' | '.join(top)}")
        print(f"    - : {' | '.join(bot)}")

    # Export so 04 can steer with it. Layer choice follows the shipped S2 vector,
    # scaled so one unit of coefficient is the same push as one unit of S2.
    cv_dir = OUT_DIR / "control_vectors"
    manifest_path = cv_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        steer_layer = manifest["vectors"]["s2"]["layer"]
        scaled = bipolar * (np.linalg.norm(s2) / np.linalg.norm(bipolar))

        stem = f"{MODEL_NAME}_bipolar_L{steer_layer}"
        (cv_dir / f"{stem}.bin").write_bytes(
            np.int32(scaled.size).tobytes() + scaled.astype(np.float32).tobytes())
        _write_gguf = importlib.import_module("03_export_control_vector").write_gguf
        _write_gguf(cv_dir / f"{stem}.gguf", scaled, steer_layer, "qwen3")

        manifest["vectors"]["bipolar"] = {
            "layer": steer_layer, "auto_picked_layer": None, "paper_layer": None,
            "norm": float(np.linalg.norm(scaled)),
            "ratio_at_layer": manifest["vectors"]["s2"].get("ratio_at_layer"),
            "bin": f"{stem}.bin", "gguf": f"{stem}.gguf",
            "note": "pain - arousal, built from llama.cpp activations; "
                    "rescaled to the S2 vector's norm",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n  wrote {stem}.bin and {stem}.gguf at layer {steer_layer}")
    else:
        print("\n  run 03_export_control_vector.py to get a steerable copy")

    np.save(OUT / "pain_minus_arousal.npy", bipolar)
    (OUT / "readout.json").write_text(json.dumps({
        "model": MODEL_NAME, "layer": layer, "tokenization": tag,
        "cos_with_s2_pain_vector": cos,
        "norm": float(np.linalg.norm(bipolar)),
        "readout": readout,
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
