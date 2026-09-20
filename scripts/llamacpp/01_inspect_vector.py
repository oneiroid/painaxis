"""What the pain direction looks like: component statistics, shape of the
distribution, and the vocabulary it points at.

The vocabulary readout is a logit lens on the direction: the dot product of the
unit pain vector with every row of the model's unembedding matrix, read out of
the GGUF so it is the same weights that will be steered. Two variants are
reported -- the raw dot product, and the dot product after the final RMSNorm
gain, which is what the residual stream actually passes through on its way to
the logits.

The bottom of the list is the readout of the negated vector, so the same table
shows both poles of the axis.

Writes results/llamacpp/<model>/vector/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from painaxis_llamacpp import MODEL_GGUF, MODEL_NAME, OUT_DIR, load_vectors  # noqa: E402

TOP_N = 40
OUT = OUT_DIR / "vector"


def participation_ratio(v):
    """Effective number of dimensions the direction lives in.

    1 means a single coordinate carries everything, len(v) means the energy is
    spread evenly.
    """
    p = v ** 2
    return float(p.sum() ** 2 / (p ** 2).sum())


def describe(name, v):
    unit = v / np.linalg.norm(v)
    energy = np.sort(unit ** 2)[::-1]
    cum = np.cumsum(energy)
    return {
        "vector": name,
        "d_model": int(v.size),
        "norm": float(np.linalg.norm(v)),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "kurtosis": float(((v - v.mean()) ** 4).mean() / v.std() ** 4),
        "participation_ratio": participation_ratio(v),
        "dims_for_50pct_energy": int(np.searchsorted(cum, 0.50) + 1),
        "dims_for_90pct_energy": int(np.searchsorted(cum, 0.90) + 1),
        "top_dim": int(np.argmax(np.abs(v))),
        "top_dim_share_of_energy": float(energy[0]),
    }


def load_unembedding():
    """(vocab, W_U, final_norm_gain) from the GGUF.

    W_U is dequantized to float32: for Qwen3-8B that is 151936 x 4096.
    """
    from gguf import GGUFReader, quants

    reader = GGUFReader(str(MODEL_GGUF))
    tensors = {t.name: t for t in reader.tensors}

    # Models with untied embeddings store output.weight; the rest reuse token_embd.
    name = "output.weight" if "output.weight" in tensors else "token_embd.weight"
    t = tensors[name]
    w = quants.dequantize(t.data, t.tensor_type).astype(np.float32)
    # GGUF stores shape reversed relative to the logical [n_vocab, n_embd].
    w = w.reshape(int(t.shape[1]), int(t.shape[0]))

    gain = None
    if "output_norm.weight" in tensors:
        g = tensors["output_norm.weight"]
        gain = quants.dequantize(g.data, g.tensor_type).astype(np.float32).reshape(-1)

    vocab_field = reader.get_field("tokenizer.ggml.tokens")
    vocab = [str(bytes(vocab_field.parts[idx]), encoding="utf-8", errors="replace")
             for idx in vocab_field.data]

    return vocab, w, gain, name


def _byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode, used by Qwen's byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _byte_decoder()


def clean(tok):
    """GGUF stores tokens byte-level encoded; map them back to real text."""
    try:
        raw = bytes(BYTE_DECODER[c] for c in tok)
    except KeyError:
        return tok
    return raw.decode("utf-8", errors="replace").replace("\n", "\\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vecs = load_vectors()
    layer = vecs["layer"]

    s1, s2 = vecs["s1"], vecs["s2"]
    cos12 = float(s1 @ s2 / (np.linalg.norm(s1) * np.linalg.norm(s2)))

    stats = [describe("s1_pain_vector", s1), describe("s2_pain_vector", s2)]
    print(f"model: {MODEL_NAME}   extraction: {vecs['extraction']} at layer {layer}")
    print(f"cos(S1, S2) = {cos12:.4f}\n")
    for s in stats:
        print(f"  {s['vector']}")
        print(f"    norm {s['norm']:.2f}   std {s['std']:.4f}   range [{s['min']:.2f}, {s['max']:.2f}]")
        print(f"    kurtosis {s['kurtosis']:.1f}   participation ratio {s['participation_ratio']:.1f} "
              f"of {s['d_model']} dims")
        print(f"    50% of its length sits in {s['dims_for_50pct_energy']} dims, "
              f"90% in {s['dims_for_90pct_energy']}")
        print(f"    largest single dim: {s['top_dim']} "
              f"({100 * s['top_dim_share_of_energy']:.1f}% of the energy)")

    # ---- plots -----------------------------------------------------------
    for name, v in [("s1", s1), ("s2", s2)]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

        axes[0].hist(v, bins=200, color="#d62728", alpha=0.85)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("component value")
        axes[0].set_ylabel("dimensions (log)")
        axes[0].set_title(f"{name.upper()} pain vector: component distribution")

        order = np.argsort(np.abs(v))[::-1]
        axes[1].plot(np.abs(v)[order], color="#d62728")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("dimension, sorted by magnitude")
        axes[1].set_ylabel("|component| (log)")
        axes[1].set_title("magnitude profile")

        top = order[:TOP_N]
        colors = ["#d62728" if v[i] > 0 else "#1f77b4" for i in top]
        axes[2].bar(range(TOP_N), v[top], color=colors)
        axes[2].set_xticks(range(0, TOP_N, 4))
        axes[2].set_xticklabels([str(top[i]) for i in range(0, TOP_N, 4)], rotation=60, fontsize=7)
        axes[2].axhline(0, color="gray", linewidth=0.8)
        axes[2].set_xlabel("dimension index")
        axes[2].set_title(f"{TOP_N} largest components")

        plt.tight_layout()
        plt.savefig(OUT / f"{name}_components.png", dpi=150)
        plt.close()

    # S1 against S2, component by component.
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(s1, s2, s=4, alpha=0.3, color="#d62728", edgecolors="none")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("S1 pain vector")
    ax.set_ylabel("S2 pain vector")
    ax.set_title(f"S1 vs S2, layer {layer}  (cos = {cos12:.3f})")
    plt.tight_layout()
    plt.savefig(OUT / "s1_vs_s2.png", dpi=150)
    plt.close()

    # ---- vocabulary readout ---------------------------------------------
    print(f"\nreading the unembedding matrix from {MODEL_GGUF.name} ...")
    vocab, w_u, gain, w_name = load_unembedding()
    print(f"  {w_name}: {w_u.shape}, vocab {len(vocab)}, "
          f"final norm gain: {'yes' if gain is not None else 'absent'}")

    readout = {}
    for name, v in [("s1", s1), ("s2", s2)]:
        unit = (v / np.linalg.norm(v)).astype(np.float32)
        logits_raw = w_u @ unit
        logits_norm = w_u @ (unit * gain) if gain is not None else logits_raw

        rows = []
        for tag, logits in [("raw", logits_raw), ("rmsnorm_gain", logits_norm)]:
            order = np.argsort(logits)
            for rank, idx in enumerate(order[::-1][:TOP_N]):
                rows.append({"variant": tag, "pole": "+pain", "rank": rank + 1,
                             "token": clean(vocab[idx]), "score": float(logits[idx])})
            for rank, idx in enumerate(order[:TOP_N]):
                rows.append({"variant": tag, "pole": "-pain", "rank": rank + 1,
                             "token": clean(vocab[idx]), "score": float(logits[idx])})
        readout[name] = rows

        top = [r["token"] for r in rows if r["variant"] == "rmsnorm_gain" and r["pole"] == "+pain"][:25]
        bot = [r["token"] for r in rows if r["variant"] == "rmsnorm_gain" and r["pole"] == "-pain"][:25]
        print(f"\n  {name.upper()} vector, layer {layer}, through the final norm:")
        print(f"    +direction: {' | '.join(top)}")
        print(f"    -direction: {' | '.join(bot)}")

    import csv
    with open(OUT / "unembedding_readout.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["vector", "variant", "pole", "rank", "token", "score"])
        wr.writeheader()
        for name, rows in readout.items():
            for r in rows:
                wr.writerow({"vector": name, **r})

    with open(OUT / "stats.json", "w") as f:
        json.dump({"model": MODEL_NAME, "gguf": str(MODEL_GGUF), "layer": layer,
                   "extraction": vecs["extraction"], "cos_s1_s2": cos12,
                   "stats": stats}, f, indent=2)

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
