"""Read the steering ladder and say what moved.

Four things are measured per coefficient:

  pain rate      share of generations containing pain/hurt as a whole word,
                 the paper's keyword measure from scripts/4.2_steering/02
  relief rate    the words the negated S2 vector points at through the
                 unembedding: calm, at ease, comfortable, fine, settled
  joy rate       the words the negated pain-minus-arousal axis points at:
                 joy, happiness, celebration, delight, rejoice, triumph
  degeneration   share of generations that collapse into a repeated 4-gram,
                 which is how a residual stream that has been pushed too far
                 usually fails

Both lexicons come from 01 and 06's unembedding readouts, so they were fixed
before any of these generations existed rather than written to fit them.

The control conditions from 04 are reported next to the pain vector. A change
that shows up equally under "shuffled" and "random" is a perturbation of that
size doing it, not the pain axis.

Writes results/llamacpp/<model>/steering/report/<vector>/.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from painaxis_llamacpp import MODEL_NAME, OUT_DIR  # noqa: E402

STEER = OUT_DIR / "steering"
OUT = STEER / "report"

PAIN_RE = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)
RELIEF_RE = re.compile(
    r"\b(?:relief|relieved|calm|calmly|calmness|at ease|content|contented|"
    r"comfortable|comfort|fine|okay|ok|peaceful|peace|settled|satisfied|"
    r"relaxed|reassured|untroubled|nothing)\b", re.IGNORECASE)
# From the negative pole of the pain-minus-arousal readout in 06.
JOY_RE = re.compile(
    r"\b(?:joy|joyful|joyous|happy|happiness|delight|delighted|delights|"
    r"celebration|celebrations|celebrate|jubilant|rejoice|rejoicing|"
    r"excitement|excited|triumph|fulfilled|fulfilment|fulfillment|"
    r"inspired|grateful|gratitude|bliss|elated)\b", re.IGNORECASE)
SAMPLE_PROMPTS = 3


def degenerate(text, n=4):
    """True when some n-gram of words repeats three or more times."""
    words = text.split()
    if len(words) < n * 3:
        return False
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return grams.most_common(1)[0][1] >= 3


def load(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    df = pd.DataFrame(rows)
    df["pain"] = df["completion"].apply(lambda t: bool(PAIN_RE.search(t)))
    df["relief"] = df["completion"].apply(lambda t: bool(RELIEF_RE.search(t)))
    df["joy"] = df["completion"].apply(lambda t: bool(JOY_RE.search(t)))
    df["degenerate"] = df["completion"].apply(degenerate)
    df["n_words"] = df["completion"].str.split().str.len()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", default="s2",
                    help="which ladder to report, matching run_<vector>.json from 04")
    args = ap.parse_args()

    run_path = STEER / f"run_{args.vector}.json"
    if not run_path.exists():
        raise SystemExit(f"no {run_path.name} in {STEER}; run 04_steer.py first")
    run = json.loads(run_path.read_text())

    out = OUT / args.vector
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for cond in run["conditions"]:
        suffix = "" if run.get("mode", "add") == "add" else f"_{run['mode']}"
        p = STEER / f"{MODEL_NAME}_{run['vector']}_L{run['layer']}{suffix}_{cond}.jsonl"
        if not p.exists():
            print(f"missing {p.name}, skipping")
            continue
        frames.append(load(p).assign(condition=cond))
    if not frames:
        raise SystemExit(f"no steering output in {STEER}")
    df = pd.concat(frames, ignore_index=True)

    table = (df.groupby(["condition", "coeff"])
               .agg(n=("completion", "size"),
                    pain_rate=("pain", "mean"),
                    relief_rate=("relief", "mean"),
                    joy_rate=("joy", "mean"),
                    degenerate_rate=("degenerate", "mean"),
                    mean_words=("n_words", "mean"))
               .reset_index())
    for c in ("pain_rate", "relief_rate", "joy_rate", "degenerate_rate"):
        table[c] = (table[c] * 100).round(1)
    table["mean_words"] = table["mean_words"].round(1)
    table.to_csv(out / "rates_by_coeff.csv", index=False)

    print(f"{MODEL_NAME}   {run['vector'].upper()} vector at layer {run['layer']}   "
          f"{run['n_prompts']} neutral prompts, greedy\n")
    print(f"{'cond':<10}{'coeff':>7}{'pain %':>9}{'relief %':>10}{'joy %':>8}"
          f"{'degen %':>9}{'words':>8}")
    for _, r in table.iterrows():
        print(f"{r['condition']:<10}{r['coeff']:>7g}{r['pain_rate']:>9}"
              f"{r['relief_rate']:>10}{r['joy_rate']:>8}"
              f"{r['degenerate_rate']:>9}{r['mean_words']:>8}")

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), sharex=True)
    for ax, col, title in zip(axes,
                              ["pain_rate", "relief_rate", "joy_rate", "degenerate_rate"],
                              ["pain or hurt word", "relief or calm word",
                               "joy or celebration word", "repeats a 4-gram"]):
        for cond, g in table.groupby("condition"):
            g = g.sort_values("coeff")
            ax.plot(g["coeff"], g[col], marker="o",
                    linewidth=2.2 if cond == "pain" else 1.3,
                    linestyle="-" if cond == "pain" else "--",
                    label=cond)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("coefficient on the pain vector")
        ax.set_ylabel("% of generations")
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.suptitle(f"{MODEL_NAME}, {run['vector'].upper()} vector at layer {run['layer']}")
    plt.tight_layout()
    plt.savefig(out / "rates_by_coeff.png", dpi=150)
    plt.close()

    # Same prompt down the whole ladder, so the drift is readable.
    lines = [f"# Steering {MODEL_NAME} against the {run['vector'].upper()} direction", "",
             f"- vector: {run['vector'].upper()}, layer {run['layer']}",
             f"- prompts: {run['n_prompts']} neutral, greedy, {run['max_tokens']} tokens", "",
             "## Rates", "", table.to_markdown(index=False), "", "## Generations", ""]

    pain_df = df[df["condition"] == "pain"]
    for idx in sorted(pain_df["prompt_idx"].unique())[:SAMPLE_PROMPTS]:
        sub = pain_df[pain_df["prompt_idx"] == idx].sort_values("coeff", ascending=False)
        lines.append(f"### `{sub.iloc[0]['prompt']}`")
        lines.append("")
        for _, r in sub.iterrows():
            text = " ".join(r["completion"].split())[:400]
            lines.append(f"**{r['coeff']:+g}** — {text}")
            lines.append("")

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
