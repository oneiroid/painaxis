"""Bipolarity on the length-matched triples, with confidence intervals, and the
self-directed set.

Version 1 of the triples had neutral sentences about 3 words shorter than
their poles, which put a shared length component into both pole vectors and
pushed cos(v_pos, v_neg) up. datasets/llamacpp_matched_affect_triples_v2.json
matches lengths stem by stem and adds SelfDirected: user turns addressed to the
assistant where the unpleasant outcome is harm to the assistant itself.

Sets and endings:
  PainRelief_1P, SadnessJoy_1P   under " I feel:" and under the dialogue frame
  SelfDirected                   under the dialogue frame only

Per set, at the analysis layer, with 95% intervals from resampling stems:
  cos(v_pos, v_neg)       -1 = one bipolar axis around neutral, as in the GAN case
  neutral position t      0.5 = neutral halfway between the poles
Across sets, in the dialogue frame:
  cosines between the valence axes (mean of positive minus negative, paired)
  each axis against the S2 pain vector refitted under the dialogue frame (from
  08) and against the vector the paper shipped
  AUC of the S2 vector at telling each set's unpleasant sentences from its
  neutral ones -- does the model's axis for a person's pain also fire for harm
  aimed at the model?

Read-only. Writes results/llamacpp/<model>/bipolarity_v2/.
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
_inspect = importlib.import_module("01_inspect_vector")

TRIPLES = REPO / "datasets" / "llamacpp_matched_affect_triples_v2.json"
TEMPLATE_DIR = OUT_DIR / "template"
OUT = OUT_DIR / "bipolarity_v2"
N_BOOT = 2000
SEED = 42
TOP_N = 20

ENDINGS = {
    "feel":     lambda s: f"{s} I feel:",
    "dialogue": lambda s: f"[User]: {s}\n[Assistant]:",
}
RUNS = [("PainRelief_1P", "feel"), ("SadnessJoy_1P", "feel"),
        ("PainRelief_1P", "dialogue"), ("SadnessJoy_1P", "dialogue"),
        ("SelfDirected", "dialogue")]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def dump(prompts, out_bin):
    if out_bin.exists() and read_activations(out_bin)[0].shape[0] == len(prompts):
        return read_activations(out_bin)[0]
    txt = out_bin.with_suffix(".txt")
    # The dumper reads one prompt per line and turns a literal \n back into a newline.
    txt.write_text("\n".join(p.replace("\n", "\\n") for p in prompts) + "\n", encoding="utf-8")
    subprocess.run([str(DUMPER), "-m", str(MODEL_GGUF), "-f", str(txt),
                    "-o", str(out_bin), "-ngl", str(N_GPU_LAYERS)], check=True)
    return read_activations(out_bin)[0]


def stats(n, p, g):
    """cos(v_pos, v_neg) and neutral position t for aligned [stems, d] arrays."""
    v_pos, v_neg = (p - n).mean(0), (g - n).mean(0)
    pm, gm, nm = p.mean(0), g.mean(0), n.mean(0)
    u = (pm - gm) / (np.linalg.norm(pm - gm) + 1e-12)
    t = float((nm @ u - gm @ u) / (pm @ u - gm @ u))
    return cos(v_pos, v_neg), t


def boot(n, p, g, rng):
    idx = rng.integers(0, len(n), size=(N_BOOT, len(n)))
    vals = np.array([stats(n[i], p[i], g[i]) for i in idx])
    return np.percentile(vals, [2.5, 97.5], axis=0)   # rows: lo, hi; cols: cos, t


def boot_cos(a_diff, b, rng):
    """CI of cos(mean over stems of a_diff, b), resampling stems of a_diff."""
    idx = rng.integers(0, len(a_diff), size=(N_BOOT, len(a_diff)))
    vals = np.array([cos(a_diff[i].mean(0), b) for i in idx])
    return np.percentile(vals, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    shipped = load_vectors()
    layer = shipped["layer"]
    data = json.loads(TRIPLES.read_text(encoding="utf-8"))["datasets"]

    # One dump per ending, covering every set that uses it.
    acts = {}
    for ending, fmt in ENDINGS.items():
        items = [(name, s) for name, e in RUNS if e == ending
                 for s in data[name]["sentences"]]
        a = dump([fmt(s["text"]) for _, s in items], OUT / f"{ending}.bin")[:, layer, :]
        for name in {n for n, _ in items}:
            rows = [i for i, (n, _) in enumerate(items) if n == name]
            sub = [items[i][1] for i in rows]
            get = lambda cat: a[[rows[j] for j, s in enumerate(sub) if s["category"] == cat]]
            order = lambda cat: np.argsort([s["set"] for s in sub if s["category"] == cat])
            acts[(name, ending)] = {c: get(c)[order(c)] for c in ("neutral", "positive", "negative")}

    # S2 pain vector refitted under the dialogue frame, from 08's dump.
    s2 = json.loads(DATASET.read_text(encoding="utf-8"))["datasets"]["S2_1P"]["sentences"]
    tdia = read_activations(TEMPLATE_DIR / "dialogue.bin")[0][: len(s2), layer, :]
    cats = np.array([s["category"] for s in s2])
    s2_dialogue = diff_in_means(tdia[np.isin(cats, PAIN_CATEGORIES)],
                                tdia[np.isin(cats, CONTROL_CATEGORIES)])
    refs = {"S2 dialogue": s2_dialogue, "S2 shipped": shipped["s2"]}

    print(f"model: {MODEL_GGUF.name}   layer {layer}   {N_BOOT} resamples of stems\n")
    print(f"{'set':<15}{'ending':<10}{'cos(v_pos, v_neg)':>28}{'neutral position t':>28}")
    summary = {"layer": layer, "n_boot": N_BOOT, "sets": {}, "cross": {}}
    for name, ending in RUNS:
        d = acts[(name, ending)]
        c, t = stats(d["neutral"], d["positive"], d["negative"])
        ci = boot(d["neutral"], d["positive"], d["negative"], rng)
        print(f"{name:<15}{ending:<10}{c:>+10.3f}  [{ci[0,0]:+.3f}, {ci[1,0]:+.3f}]"
              f"{t:>12.3f}  [{ci[0,1]:.3f}, {ci[1,1]:.3f}]")
        summary["sets"][f"{name}/{ending}"] = {
            "cos_pos_neg": c, "cos_ci": ci[:, 0].tolist(),
            "neutral_t": t, "t_ci": ci[:, 1].tolist()}

    # Valence axes in the dialogue frame, and how they relate.
    val = {name: acts[(name, "dialogue")] for name, e in RUNS if e == "dialogue"}
    vdiff = {k: v["positive"] - v["negative"] for k, v in val.items()}
    ndiff = {k: v["negative"] - v["neutral"] for k, v in val.items()}
    names = list(val)

    print("\nvalence axes (positive minus negative, paired), dialogue frame -- cosines")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = cos(vdiff[a].mean(0), vdiff[b].mean(0))
            print(f"  {a} vs {b}: {c:+.3f}")
            summary["cross"][f"valence {a} vs {b}"] = c

    print("\nagainst the S2 pain vector (sign: + means the unpleasant pole points the pain way)")
    for k in names:
        for rname, r in refs.items():
            c_neg = cos(ndiff[k].mean(0), r)
            ci = boot_cos(ndiff[k], r, rng)
            c_val = cos(-vdiff[k].mean(0), r)
            print(f"  {k:<15} v_neg vs {rname:<12} {c_neg:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                  f"   unpleasant-minus-pleasant vs {rname:<12} {c_val:+.3f}")
            summary["cross"][f"{k} v_neg vs {rname}"] = {"cos": c_neg, "ci": ci.tolist()}
            summary["cross"][f"{k} neg-pos vs {rname}"] = c_val

    print("\nAUC of the S2 vector: unpleasant vs neutral sentences of each set")
    for k in names:
        for rname, r in refs.items():
            u = r / np.linalg.norm(r)
            neg, neu = val[k]["negative"] @ u, val[k]["neutral"] @ u
            a = roc_auc_score(np.r_[np.ones(len(neg)), np.zeros(len(neu))], np.r_[neg, neu])
            print(f"  {k:<15} {rname:<12} {a:.3f}")
            summary["cross"][f"{k} AUC neg vs neutral on {rname}"] = float(a)

    # The shipped vector was fitted under " I feel:", so score it there too; under
    # the dialogue frame alone a low AUC could be the ending, not the vector.
    print("\nAUC of the shipped vectors under their own ending ' I feel:'")
    for k in ("PainRelief_1P", "SadnessJoy_1P"):
        d = acts[(k, "feel")]
        for vn in ("s2", "s1"):
            u = shipped[vn] / np.linalg.norm(shipped[vn])
            neg, neu, pos = d["negative"] @ u, d["neutral"] @ u, d["positive"] @ u
            f = lambda x, y: float(roc_auc_score(np.r_[np.ones(len(x)), np.zeros(len(y))], np.r_[x, y]))
            print(f"  {k:<15} {vn.upper()} shipped   unpleasant vs neutral {f(neg, neu):.3f}"
                  f"   unpleasant vs pleasant {f(neg, pos):.3f}")
            summary["cross"][f"{k} feel AUC neg vs neutral on {vn} shipped"] = f(neg, neu)
            summary["cross"][f"{k} feel AUC neg vs pos on {vn} shipped"] = f(neg, pos)

    print(f"\nreading the unembedding from {MODEL_GGUF.name} ...")
    vocab, w_u, gain, _ = _inspect.load_unembedding()
    gain = gain if gain is not None else np.ones(w_u.shape[1], dtype=np.float32)
    summary["readouts"] = {}
    for k in names:
        for label, v in (("valence (pleasant end +)", vdiff[k].mean(0)),
                         ("v_neg (neutral to unpleasant)", ndiff[k].mean(0))):
            logits = w_u @ ((v / np.linalg.norm(v)).astype(np.float32) * gain)
            o = np.argsort(logits)
            top = [_inspect.clean(vocab[i]) for i in o[::-1][:TOP_N]]
            bot = [_inspect.clean(vocab[i]) for i in o[:TOP_N]]
            print(f"\n  {k}: {label}\n    + : {' | '.join(top)}\n    - : {' | '.join(bot)}")
            summary["readouts"][f"{k}: {label}"] = {"plus": top, "minus": bot}

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
