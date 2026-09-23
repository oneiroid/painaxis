"""Does the " I feel:" ending shape the vectors?

Every sentence in the paper's section 3 sets, in the matched triples, and in the
steering prompts ends with " I feel:", and activations are read at the final
token. A vector fitted and tested on one template can look cleaner than the
thing it is meant to measure. This script re-reads the same sentences with four
endings and compares what comes out:

  feel       "<sentence> I feel:"                  the paper's template
  none       "<sentence>"                          ends on the full stop
  then       "<sentence> Then:"                    a continuation cue with no affect
  dialogue   "[User]: <sentence>\\n[Assistant]:"   the format of the paper's 4.1 set

Three measurements, at the analysis layer:

  1. The S2 pain vector (denoised difference in means, as in
     scripts/3.2_pain_vectors/01) fitted separately under each ending: cosine
     between the endings, and AUC when it is fitted on the odd sentence sets of
     one ending and scored on the even sets of another. The diagonal is held-out
     too, so every cell is comparable.
  2. The paired valence axes from the triples under each ending: cosine between
     endings, and cos(v_pos, v_neg) per ending.
  3. Each ending's S2 vector against the vector the paper shipped.

Read-only. Writes results/llamacpp/<model>/template/.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from painaxis_llamacpp import (  # noqa: E402
    CONTROL_CATEGORIES, DATASET, DUMPER, MODEL_GGUF, N_GPU_LAYERS, OUT_DIR,
    PAIN_CATEGORIES, REPO, load_vectors, read_activations)

import importlib  # noqa: E402
diff_in_means = importlib.import_module("06_bipolar_axis").diff_in_means

TRIPLES = REPO / "datasets" / "llamacpp_matched_affect_triples.json"
OUT = OUT_DIR / "template"
SUFFIX = " I feel:"

TEMPLATES = {
    "feel":     lambda s: f"{s} I feel:",
    "none":     lambda s: s,
    "then":     lambda s: f"{s} Then:",
    "dialogue": lambda s: f"[User]: {s}\\n[Assistant]:",   # \\n is unescaped by the dumper
}


def base(prompt):
    assert prompt.endswith(SUFFIX), prompt
    return prompt[: -len(SUFFIX)]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def auc(acts, cats, vec):
    proj = acts @ (vec / np.linalg.norm(vec))
    pain, ctrl = np.isin(cats, PAIN_CATEGORIES), np.isin(cats, CONTROL_CATEGORIES)
    labels = np.r_[np.ones(pain.sum()), np.zeros(ctrl.sum())]
    return float(roc_auc_score(labels, np.r_[proj[pain], proj[ctrl]]))


def fit_pain(acts, cats):
    return diff_in_means(acts[np.isin(cats, PAIN_CATEGORIES)],
                         acts[np.isin(cats, CONTROL_CATEGORIES)])


def matrix_str(names, m, fmt="{:+.3f}"):
    head = " " * 10 + "".join(f"{n:>10}" for n in names)
    rows = [f"{a:<10}" + "".join(f"{fmt.format(m[i][j]):>10}" for j in range(len(names)))
            for i, a in enumerate(names)]
    return "\n".join([head] + rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shipped = load_vectors()
    layer = shipped["layer"]

    s2 = json.loads(DATASET.read_text(encoding="utf-8"))["datasets"]["S2_1P"]["sentences"]
    triples = json.loads(TRIPLES.read_text(encoding="utf-8"))["datasets"]

    # One list of sentences, dumped once per ending, so each ending costs one model load.
    items = [("S2", s["category"], s["set"], base(s["prompt"])) for s in s2]
    for name, block in triples.items():
        items += [(name, s["category"], s["set"], base(s["prompt"])) for s in block["sentences"]]

    acts = {}
    for t, fmt in TEMPLATES.items():
        out_bin = OUT / f"{t}.bin"
        if out_bin.exists() and read_activations(out_bin)[0].shape[0] == len(items):
            acts[t] = read_activations(out_bin)[0][:, layer, :]
            continue
        txt = out_bin.with_suffix(".txt")
        txt.write_text("\n".join(fmt(x[3]) for x in items) + "\n", encoding="utf-8")
        subprocess.run([str(DUMPER), "-m", str(MODEL_GGUF), "-f", str(txt),
                        "-o", str(out_bin), "-ngl", str(N_GPU_LAYERS)], check=True)
        acts[t] = read_activations(out_bin)[0][:, layer, :]

    names = list(TEMPLATES)
    src = np.array([x[0] for x in items])
    cat = np.array([x[1] for x in items])
    sets = np.array([x[2] for x in items])

    # ---- 1. the S2 pain vector under each ending -------------------------
    is_s2 = src == "S2"
    odd, even = is_s2 & (sets % 2 == 1), is_s2 & (sets % 2 == 0)

    full = {t: fit_pain(acts[t][is_s2], cat[is_s2]) for t in names}
    half = {t: fit_pain(acts[t][odd], cat[odd]) for t in names}

    cos_m = [[cos(full[a], full[b]) for b in names] for a in names]
    auc_m = [[auc(acts[b][even], cat[even], half[a]) for b in names] for a in names]

    print(f"model: {MODEL_GGUF.name}   layer {layer}\n")
    print("S2 pain vector fitted under each ending -- cosine between them")
    print(matrix_str(names, cos_m))
    print("\nAUC pain vs control: fitted on odd sets of the row ending, "
          "scored on even sets of the column ending")
    print(matrix_str(names, auc_m, "{:.3f}"))
    print("\ncos with the vector the paper shipped (fitted under 'feel' on bf16):")
    for t in names:
        print(f"  {t:<10}{cos(full[t], shipped['s2']):+.3f}")

    # ---- 2. the paired valence axes under each ending ---------------------
    triple_out = {}
    for name in triples:
        m = src == name
        val, bip = {}, {}
        for t in names:
            a = acts[t][m]
            c, st = cat[m], sets[m]
            order = lambda k: a[c == k][np.argsort(st[c == k])]
            n, p, g = order("neutral"), order("positive"), order("negative")
            val[t] = (p - g).mean(axis=0)
            bip[t] = cos((p - n).mean(axis=0), (g - n).mean(axis=0))
        vm = [[cos(val[a], val[b]) for b in names] for a in names]
        print(f"\n{name}: valence axis (positive minus negative, paired) -- "
              f"cosine between endings")
        print(matrix_str(names, vm))
        print("  cos(v_pos, v_neg) under each ending: "
              + "   ".join(f"{t} {bip[t]:+.3f}" for t in names))
        triple_out[name] = {"valence_cos": vm, "cos_pos_neg": bip}

    (OUT / "summary.json").write_text(json.dumps({
        "layer": layer, "templates": names,
        "s2_cos": cos_m, "s2_auc_heldout": auc_m,
        "s2_cos_shipped": {t: cos(full[t], shipped["s2"]) for t in names},
        "triples": triple_out,
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
