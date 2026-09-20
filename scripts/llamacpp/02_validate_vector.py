"""Does the paper's pain vector still separate pain from control inside the
quantized GGUF?

The vectors were fitted on bf16 HuggingFace activations through TransformerLens.
This script re-reads the residual stream through llama.cpp instead, projects the
S1 and S2 sentences onto the shipped vector, and reports the AUC -- the same
numbers the paper's in-sample table gives (for Qwen3-8B at layer 34, final
token: 0.948 for the S1 vector on S1_1P and 0.99 for the S2 vector on S2_1P).

This is read-only. It measures the axis without steering the model along it.

Writes results/llamacpp/<model>/validation/.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from painaxis_llamacpp import (  # noqa: E402
    CONTROL_CATEGORIES, DATASET, DUMPER, MODEL_GGUF, MODEL_NAME, N_GPU_LAYERS,
    OUT_DIR, PAIN_CATEGORIES, CATEGORY_LABELS, load_vectors, read_activations)

# The paper fits on S2_1P and S1_1P; the rest are scored, not fitted.
SETS = ["S1_1P", "S2_1P", "Random_1P", "Arousal_1P", "Numb_1P"]

# In-sample AUCs from results/3.2_pain_vectors/auc_tables/s1_auc_final_token_TABLE.csv,
# the row for MODEL_NAME. These are the numbers the GGUF has to reproduce.
PAPER_AUC = {("s1", "S1_1P"): 0.948, ("s2", "S2_1P"): 0.990}
OUT = OUT_DIR / "validation"


def auc(acts, cats, vec):
    """AUC of the projection onto vec, pain sentences vs control sentences."""
    unit = vec / np.linalg.norm(vec)
    proj = acts @ unit
    pain = np.isin(cats, PAIN_CATEGORIES)
    ctrl = np.isin(cats, CONTROL_CATEGORIES)
    if pain.sum() == 0 or ctrl.sum() == 0:
        return float("nan")
    labels = np.concatenate([np.ones(pain.sum()), np.zeros(ctrl.sum())])
    scores = np.concatenate([proj[pain], proj[ctrl]])
    return float(roc_auc_score(labels, scores))


def dump(prompts, out_bin, add_bos):
    """Run the dumper, reusing an existing file when it already has the right shape."""
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    txt = out_bin.with_suffix(".txt")
    txt.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    if out_bin.exists():
        try:
            acts, _ = read_activations(out_bin)
            if acts.shape[0] == len(prompts):
                print(f"  reusing {out_bin.name}")
                return acts
        except Exception:
            pass

    cmd = [str(DUMPER), "-m", str(MODEL_GGUF), "-f", str(txt), "-o", str(out_bin),
           "-ngl", str(N_GPU_LAYERS)]
    if add_bos:
        cmd.append("--bos")
    subprocess.run(cmd, check=True)
    return read_activations(out_bin)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vecs = load_vectors()
    layer = vecs["layer"]
    data = json.loads(DATASET.read_text(encoding="utf-8"))["datasets"]

    print(f"model:  {MODEL_GGUF.name}")
    print(f"vector: {MODEL_NAME}, {vecs['extraction']}, layer {layer}\n")

    # TransformerLens prepends a BOS token by default; llama.cpp's Qwen3 vocab
    # has none. Try both and keep whichever reproduces the paper.
    results, acts_by_bos = {}, {}
    for add_bos in (False, True):
        tag = "bos" if add_bos else "nobos"
        acts_by_bos[tag] = {}
        for name in SETS:
            sents = data[name]["sentences"]
            acts = dump([s["prompt"] for s in sents], OUT / f"{name}_{tag}.bin", add_bos)
            acts_by_bos[tag][name] = (acts, np.array([s["category"] for s in sents]))

        for vname in ("s1", "s2"):
            for name in ("S1_1P", "S2_1P"):
                acts, cats = acts_by_bos[tag][name]
                results[(tag, vname, name)] = auc(acts[:, layer, :], cats, vecs[vname])

    print(f"{'AUC pain vs control':<28}{'no BOS':>10}{'with BOS':>10}{'paper':>10}")
    for vname in ("s1", "s2"):
        for name in ("S1_1P", "S2_1P"):
            paper = PAPER_AUC.get((vname, name))
            print(f"  {vname.upper()} vector on {name:<11}"
                  f"{results[('nobos', vname, name)]:>10.3f}"
                  f"{results[('bos', vname, name)]:>10.3f}"
                  f"{(f'{paper:.3f}' if paper else '-'):>10}")

    # Whichever tokenization matches the paper better is the one to steer with.
    def gap(t):
        return sum(abs(results[(t, v, s)] - a) for (v, s), a in PAPER_AUC.items())

    best = min(("nobos", "bos"), key=gap)
    print(f"\ntotal distance from the paper: "
          f"nobos {gap('nobos'):.4f}, bos {gap('bos'):.4f}")
    print(f"\nclosest to the paper: {best}")
    acts_all = acts_by_bos[best]

    # AUC at every layer, to check the peak is still where the paper put it.
    n_layers = acts_all["S2_1P"][0].shape[1]
    curves = {}
    for vname in ("s1", "s2"):
        for name in ("S1_1P", "S2_1P"):
            acts, cats = acts_all[name]
            curves[f"{vname}_{name}"] = [auc(acts[:, L, :], cats, vecs[vname])
                                         for L in range(n_layers)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, ys in curves.items():
        ax.plot(range(n_layers), ys, label=key, linewidth=1.8)
    ax.axvline(layer, color="red", linestyle="--", alpha=0.6, label=f"extraction layer {layer}")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xlabel("layer (residual stream after the block)")
    ax.set_ylabel("AUC, pain vs control")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"{MODEL_NAME} through llama.cpp ({MODEL_GGUF.stem.split('-')[-1]}), {best}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "auc_by_layer.png", dpi=150)
    plt.close()

    # Where each condition lands on the axis, z-scored against S2_1P.
    unit = vecs["s2"] / np.linalg.norm(vecs["s2"])
    ref = acts_all["S2_1P"][0][:, layer, :] @ unit
    mu, sd = ref.mean(), ref.std()
    z_rows = []
    for name, (acts, cats) in acts_all.items():
        z = (acts[:, layer, :] @ unit - mu) / sd
        if name in ("S1_1P", "S2_1P"):
            z_rows.append({"set": name, "group": "pain (A1-A5)",
                           "z": float(z[np.isin(cats, PAIN_CATEGORIES)].mean())})
            z_rows.append({"set": name, "group": "control (B-E)",
                           "z": float(z[np.isin(cats, CONTROL_CATEGORIES)].mean())})
        else:
            z_rows.append({"set": name, "group": "all", "z": float(z.mean())})

    print("\nmean z on the S2 axis (reference: S2_1P):")
    for r in z_rows:
        print(f"  {r['set']:<12} {r['group']:<15} {r['z']:+.3f}")

    # Per-category strip plot on S2_1P: the picture the paper's Figure 2 shows.
    acts, cats = acts_all["S2_1P"]
    proj = acts[:, layer, :] @ unit
    groups = []
    for c in PAIN_CATEGORIES + CONTROL_CATEGORIES:
        m = cats == c
        if m.sum():
            groups.append((f"{c} ({CATEGORY_LABELS[c]})", proj[m], c in PAIN_CATEGORIES))
    groups.sort(key=lambda g: g[1].mean(), reverse=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rng = np.random.default_rng(42)
    for i, (label, vals, is_pain) in enumerate(groups):
        color = "#d62728" if is_pain else "#1f77b4"
        ax.scatter(vals, i + rng.uniform(-0.15, 0.15, len(vals)),
                   c=color, alpha=0.55, s=22, edgecolors="none")
        ax.scatter([vals.mean()], [i], c=color, s=130,
                   marker="o" if is_pain else "s", edgecolors="white", linewidths=2, zorder=5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups])
    ax.set_xlabel(f"projection onto the S2 pain vector (layer {layer})")
    ax.set_title(f"S2_1P through llama.cpp, {MODEL_GGUF.name}")
    plt.tight_layout()
    plt.savefig(OUT / "strip_S2_1P.png", dpi=150)
    plt.close()

    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "model": MODEL_NAME, "gguf": str(MODEL_GGUF), "layer": layer,
            "tokenization": best, "n_gpu_layers": N_GPU_LAYERS,
            "auc": {f"{t}/{v}/{s}": a for (t, v, s), a in results.items()},
            "paper_auc": {f"{v}/{s}": a for (v, s), a in PAPER_AUC.items()},
            "z_scores": z_rows,
        }, f, indent=2)

    np.save(OUT / "auc_by_layer.npy", np.array([curves[k] for k in sorted(curves)]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
