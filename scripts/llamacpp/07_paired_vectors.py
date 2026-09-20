"""Paired affect vectors, and whether the model's affect space is bipolar.

The paper's vectors are mean(pain sentences) - mean(control sentences) over
different sentences, so the difference carries the topic gap along with the
affect. The GAN attribute vectors that negate cleanly are built the other way:
same face, expression off and on, so identity cancels and only the attribute is
left.

This script builds the paired version in three forms:

  nociception   mean(S2_1P A1 - Numb_1P) over the 20 items where this repository
                already ships both halves: same injury, nociception removed by a
                clause. The closest thing in the data to "same face, expression
                off".
  pain-relief   mean(positive - neutral) and mean(negative - neutral) over the
                40 matched stems in datasets/llamacpp_matched_affect_triples.json
  sadness-joy   the same, over the other 40 stems

Then it answers the question the GAN result raises. Negating a neutral-to-sad
vector and getting a happy face implies v_positive = -v_negative measured from a
common neutral origin. That is one number:

  cos(v_pos, v_neg)   -1 means the two poles are one axis and negation works;
                      0 means they are separate directions and negation cannot
                      work however clean the pairing is
  midpoint t          where the neutral mean falls between the two poles along
                      the axis joining them; 0.5 means neutral is the origin
  split-half cos      the same vector built on half the stems against the other
                      half, which is the ceiling any of the above can reach

Everything here is read-only.

Writes results/llamacpp/<model>/paired/.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from painaxis_llamacpp import (  # noqa: E402
    DATASET, DUMPER, MODEL_GGUF, MODEL_NAME, N_GPU_LAYERS, OUT_DIR, REPO,
    load_vectors, read_activations)

import importlib  # noqa: E402
_inspect = importlib.import_module("01_inspect_vector")

TRIPLES = REPO / "datasets" / "llamacpp_matched_affect_triples.json"
OUT = OUT_DIR / "paired"
VALIDATION = OUT_DIR / "validation"
TOP_N = 20
SEED = 42


def dump(prompts, out_bin):
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    if out_bin.exists():
        try:
            acts, _ = read_activations(out_bin)
            if acts.shape[0] == len(prompts):
                return acts
        except Exception:
            pass
    txt = out_bin.with_suffix(".txt")
    txt.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    subprocess.run([str(DUMPER), "-m", str(MODEL_GGUF), "-f", str(txt),
                    "-o", str(out_bin), "-ngl", str(N_GPU_LAYERS)], check=True)
    return read_activations(out_bin)[0]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bipolarity(neutral, positive, negative):
    """Per-layer cos(v_pos, v_neg), pole norms, and where neutral sits between them.

    All three arrays are [n_stems, n_layers, n_embd] and aligned stem by stem, so
    the differences are taken pair by pair before averaging.
    """
    n_layers = neutral.shape[1]
    rows = []
    for L in range(n_layers):
        v_pos = (positive[:, L, :] - neutral[:, L, :]).mean(axis=0)
        v_neg = (negative[:, L, :] - neutral[:, L, :]).mean(axis=0)

        # Where the neutral mean falls on the axis joining the two poles.
        axis = positive[:, L, :].mean(axis=0) - negative[:, L, :].mean(axis=0)
        u = axis / (np.linalg.norm(axis) + 1e-12)
        p_p = positive[:, L, :].mean(axis=0) @ u
        p_g = negative[:, L, :].mean(axis=0) @ u
        p_n = neutral[:, L, :].mean(axis=0) @ u
        t = float((p_n - p_g) / (p_p - p_g)) if abs(p_p - p_g) > 1e-9 else float("nan")

        rows.append({
            "layer": L,
            "cos_pos_neg": cos(v_pos, v_neg),
            "norm_pos": float(np.linalg.norm(v_pos)),
            "norm_neg": float(np.linalg.norm(v_neg)),
            "neutral_midpoint": t,
        })
    return rows


def split_half(neutral, other, layer, seed=SEED):
    """cos between the paired vector built on one half of the stems and the other.

    This is the reliability ceiling: cos(v_pos, v_neg) cannot be read as
    meaningful beyond the precision this number reports.
    """
    n = neutral.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    a, b = idx[: n // 2], idx[n // 2:]
    va = (other[a, layer, :] - neutral[a, layer, :]).mean(axis=0)
    vb = (other[b, layer, :] - neutral[b, layer, :]).mean(axis=0)
    return cos(va, vb)


def decompose(neutral, positive, negative, layer):
    """Split each pole vector into the part the two poles share and the part that
    separates them.

    Sentences that carry an emotion are also sentences where something happened,
    and the neutral stems are sentences where nothing did. That "an event
    occurred" component is common to both poles and is what keeps
    cos(v_pos, v_neg) away from -1. A neutral face and the same face smiling do
    not have this problem, which is why the GAN case negates cleanly.

      v_common   (v_pos + v_neg) / 2   -- salience, shared by both poles
      v_valence  mean(positive - negative), taken pair by pair -- what is left
    """
    n, p, g = (a[:, layer, :] for a in (neutral, positive, negative))
    v_pos = (p - n).mean(axis=0)
    v_neg = (g - n).mean(axis=0)
    v_valence = (p - g).mean(axis=0)
    v_common = 0.5 * (v_pos + v_neg)

    # How much of each pole is salience and how much is valence.
    def share(v, basis):
        u = basis / (np.linalg.norm(basis) + 1e-12)
        return float((v @ u) ** 2 / (v @ v))

    return {
        "v_pos": v_pos, "v_neg": v_neg,
        "v_valence": v_valence, "v_common": v_common,
        "stats": {
            "norm_valence": float(np.linalg.norm(v_valence)),
            "norm_common": float(np.linalg.norm(v_common)),
            "cos_valence_common": cos(v_valence, v_common),
            "pos_share_common": share(v_pos, v_common),
            "pos_share_valence": share(v_pos, v_valence),
            "neg_share_common": share(v_neg, v_common),
            "neg_share_valence": share(v_neg, v_valence),
        },
    }


def readout(vocab, w_u, gain, v, n=TOP_N):
    unit = (v / np.linalg.norm(v)).astype(np.float32)
    logits = w_u @ (unit * gain)
    order = np.argsort(logits)
    return ([_inspect.clean(vocab[i]) for i in order[::-1][:n]],
            [_inspect.clean(vocab[i]) for i in order[:n]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shipped = load_vectors()
    layer = shipped["layer"]

    triples = json.loads(TRIPLES.read_text(encoding="utf-8"))["datasets"]
    core = json.loads(DATASET.read_text(encoding="utf-8"))["datasets"]

    print(f"model: {MODEL_GGUF.name}   analysis layer: {layer}\n")

    # ---- the three paired constructions --------------------------------
    acts = {}
    for name, block in triples.items():
        by_cat = {}
        for cat in ("neutral", "positive", "negative"):
            sents = sorted([s for s in block["sentences"] if s["category"] == cat],
                           key=lambda s: s["set"])
            by_cat[cat] = dump([s["prompt"] for s in sents], OUT / f"{name}_{cat}.bin")
        acts[name] = by_cat

    # Numb_1P sets 1-20 are the same injuries as S2_1P category A1 sets 1-20,
    # with the nociception removed by a clause.
    numb_sents = sorted([s for s in core["Numb_1P"]["sentences"] if s["set"] <= 20],
                        key=lambda s: s["set"])
    s2_a1 = sorted([s for s in core["S2_1P"]["sentences"] if s["category"] == "A1"],
                   key=lambda s: s["set"])
    assert len(numb_sents) == len(s2_a1) == 20
    numb_acts = dump([s["prompt"] for s in numb_sents], OUT / "Numb_paired.bin")
    pain_acts = dump([s["prompt"] for s in s2_a1], OUT / "S2_A1_paired.bin")

    summary = {"model": MODEL_NAME, "gguf": str(MODEL_GGUF), "layer": layer,
               "sets": {}, "nociception": {}}

    # ---- nociception: same event, pain on and off ----------------------
    v_noci_by_layer = [(pain_acts[:, L, :] - numb_acts[:, L, :]).mean(axis=0)
                       for L in range(pain_acts.shape[1])]
    v_noci = v_noci_by_layer[layer]
    unpaired = pain_acts[:, layer, :].mean(axis=0) - numb_acts[:, layer, :].mean(axis=0)

    print("nociception vector (S2 A1 minus Numb, 20 matched injuries)")
    print(f"  norm {np.linalg.norm(v_noci):.1f}")
    print(f"  cos with the shipped S2 pain vector: {cos(v_noci, shipped['s2']):+.4f}")
    print(f"  cos with the shipped S1 pain vector: {cos(v_noci, shipped['s1']):+.4f}")
    print(f"  split-half reliability: "
          f"{cos((pain_acts[:10, layer, :] - numb_acts[:10, layer, :]).mean(axis=0), (pain_acts[10:, layer, :] - numb_acts[10:, layer, :]).mean(axis=0)):+.4f}")
    summary["nociception"] = {
        "n_pairs": 20, "norm": float(np.linalg.norm(v_noci)),
        "cos_shipped_s2": cos(v_noci, shipped["s2"]),
        "cos_shipped_s1": cos(v_noci, shipped["s1"]),
        # Paired and unpaired coincide for a balanced design; kept as a check.
        "cos_paired_vs_unpaired": cos(v_noci, unpaired),
    }
    np.save(OUT / "v_nociception.npy", np.stack(v_noci_by_layer))

    # ---- the bipolarity question ---------------------------------------
    curves = {}
    for name, by_cat in acts.items():
        rows = bipolarity(by_cat["neutral"], by_cat["positive"], by_cat["negative"])
        curves[name] = rows
        at = rows[layer]
        sh_pos = split_half(by_cat["neutral"], by_cat["positive"], layer)
        sh_neg = split_half(by_cat["neutral"], by_cat["negative"], layer)

        print(f"\n{name}  ({by_cat['neutral'].shape[0]} matched stems, layer {layer})")
        print(f"  cos(v_positive, v_negative) = {at['cos_pos_neg']:+.4f}")
        print(f"  neutral sits at t = {at['neutral_midpoint']:.3f} between the poles"
              f"  (0.5 = exactly between)")
        print(f"  |v_positive| {at['norm_pos']:.1f}   |v_negative| {at['norm_neg']:.1f}")
        print(f"  split-half reliability: positive {sh_pos:+.3f}, negative {sh_neg:+.3f}")

        best = min(rows[1:], key=lambda r: r["cos_pos_neg"])
        print(f"  most bipolar layer: {best['layer']} "
              f"(cos {best['cos_pos_neg']:+.4f}, t {best['neutral_midpoint']:.3f})")

        summary["sets"][name] = {
            "n_stems": int(by_cat["neutral"].shape[0]),
            "at_layer": at,
            "split_half_pos": sh_pos, "split_half_neg": sh_neg,
            "most_bipolar_layer": best,
        }

    # ---- salience against valence ---------------------------------------
    dec = {name: decompose(b["neutral"], b["positive"], b["negative"], layer)
           for name, b in acts.items()}

    print("\nsplitting each pole into the part both poles share and the part that separates them")
    for name, d in dec.items():
        st = d["stats"]
        print(f"\n  {name}")
        print(f"    |v_common| (salience) {st['norm_common']:.1f}   "
              f"|v_valence| {st['norm_valence']:.1f}")
        print(f"    cos(valence, common) = {st['cos_valence_common']:+.4f}")
        print(f"    v_negative is {100 * st['neg_share_common']:.0f}% salience, "
              f"{100 * st['neg_share_valence']:.0f}% valence")
        print(f"    v_positive is {100 * st['pos_share_common']:.0f}% salience, "
              f"{100 * st['pos_share_valence']:.0f}% valence")
        summary["sets"][name]["decomposition"] = st

    names = list(dec)
    if len(names) == 2:
        a, b = names
        c_val = cos(dec[a]["v_valence"], dec[b]["v_valence"])
        c_com = cos(dec[a]["v_common"], dec[b]["v_common"])
        print(f"\n  across the two domains:")
        print(f"    cos(valence_{a}, valence_{b}) = {c_val:+.4f}")
        print(f"    cos(common_{a},  common_{b})  = {c_com:+.4f}")
        summary["across_domains"] = {"cos_valence": c_val, "cos_common": c_com}
    for name, d in dec.items():
        summary["sets"][name]["cos_valence_shipped_s2"] = cos(d["v_valence"], shipped["s2"])

    # ---- what the poles read as ----------------------------------------
    print(f"\nreading the unembedding from {MODEL_GGUF.name} ...")
    vocab, w_u, gain, _ = _inspect.load_unembedding()
    if gain is None:
        gain = np.ones(w_u.shape[1], dtype=np.float32)

    readouts = {}
    named = {"nociception (pain minus numb)": v_noci}
    for name, d in dec.items():
        named[f"{name}: v_negative"] = d["v_neg"]
        named[f"{name}: v_positive"] = d["v_pos"]
        named[f"{name}: v_valence (poles only)"] = d["v_valence"]
        named[f"{name}: v_common (salience)"] = d["v_common"]

    for name, v in named.items():
        top, bot = readout(vocab, w_u, gain, v)
        readouts[name] = {"positive_end": top, "negative_end": bot}
        print(f"\n  {name}")
        print(f"    + : {' | '.join(top)}")
        print(f"    - : {' | '.join(bot)}")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, rows in curves.items():
        layers = [r["layer"] for r in rows]
        axes[0].plot(layers, [r["cos_pos_neg"] for r in rows], linewidth=2, label=name)
        axes[1].plot(layers, [r["neutral_midpoint"] for r in rows], linewidth=2, label=name)
    axes[0].axhline(-1, color="green", linestyle=":", label="perfectly bipolar")
    axes[0].axhline(0, color="gray", linestyle=":", label="independent directions")
    axes[0].axvline(layer, color="red", linestyle="--", alpha=0.5)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("cos(v_positive, v_negative)")
    axes[0].set_title("Is the affect axis bipolar around neutral?")
    axes[0].legend(fontsize=8)

    axes[1].axhline(0.5, color="green", linestyle=":", label="neutral at the origin")
    axes[1].axvline(layer, color="red", linestyle="--", alpha=0.5)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("position of neutral between the poles")
    axes[1].set_title("Where the neutral sentences sit")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "bipolarity_by_layer.png", dpi=150)
    plt.close()

    summary["readouts"] = readouts
    summary["curves"] = curves
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for name, d in dec.items():
        for key in ("v_pos", "v_neg", "v_valence", "v_common"):
            np.save(OUT / f"{name}_{key}.npy", d[key])

    # Export the paired valence axes so 04 can steer with them. They are the
    # matched-stem version of what 06 builds from unpaired sets.
    cv_dir = OUT_DIR / "control_vectors"
    manifest_path = cv_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        steer_layer = manifest["vectors"]["s2"]["layer"]
        ref_norm = np.linalg.norm(shipped["s2"])
        write_gguf = importlib.import_module("03_export_control_vector").write_gguf
        for name, d in dec.items():
            tag = "valence_pain" if "Pain" in name else "valence_mood"
            # Rebuild at the steering layer, not the analysis layer.
            b = acts[name]
            v = (b["positive"][:, steer_layer, :] - b["negative"][:, steer_layer, :]).mean(axis=0)
            v = v * (ref_norm / np.linalg.norm(v))   # one coefficient unit = one S2 unit
            stem = f"{MODEL_NAME}_{tag}_L{steer_layer}"
            (cv_dir / f"{stem}.bin").write_bytes(
                np.int32(v.size).tobytes() + v.astype(np.float32).tobytes())
            write_gguf(cv_dir / f"{stem}.gguf", v, steer_layer, "qwen3")
            manifest["vectors"][tag] = {
                "layer": steer_layer, "auto_picked_layer": None, "paper_layer": None,
                "norm": float(np.linalg.norm(v)),
                "ratio_at_layer": manifest["vectors"]["s2"].get("ratio_at_layer"),
                "bin": f"{stem}.bin", "gguf": f"{stem}.gguf",
                "note": f"paired positive minus negative over the {name} matched stems, "
                        f"rescaled to the S2 vector norm; positive coefficients move "
                        f"toward the pleasant pole",
            }
            print(f"  wrote {stem}.bin and {stem}.gguf")
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
