# Running the pain axis through llama.cpp

The paper's pipeline needs a datacentre GPU, `transformer_lens` and bf16 HuggingFace
weights. These scripts take the pain vectors this repository already ships and use them
against a quantized GGUF through llama.cpp, so the axis can be looked at and steered on
one desktop card.

Everything here reads `results/3.2_pain_vectors/pain_vectors/<model>/pain_vectors.pt`.
No vector is re-fitted.

## Which model

`pain_vectors.pt` is a `d_model`-wide vector in one model's residual-stream basis. It
only means anything for the model it came from — Qwen3-8B's vector cannot be applied to
Qwen3-4B, and not only because 4096 ≠ 2560. Of the 25 models in the paper, the ones with
a public GGUF and a desktop-sized footprint are Qwen3-8B and Qwen3-14B.

Default here is **`Qwen/Qwen3-8B`**, which the paper's results folder calls
`Qwen_3_8B_base`: 36 layers, `d_model` 4096, vector extracted at layer 34 from the final
token, steering layers 21 (S1) and 27 (S2).

```bash
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q8_0.gguf --local-dir models/
```

Q8_0 rather than a smaller quant because the whole exercise depends on the GGUF's
residual stream sitting in the same geometry as the bf16 activations the vector was
measured in. `02_validate_vector.py` is what checks that assumption.

## Setup

```bash
python -m venv .venv-llamacpp
.venv-llamacpp/bin/pip install numpy pandas scipy scikit-learn matplotlib torch gguf tabulate
```

llama.cpp has to be built with the control-vector tools available. Point `LLAMA_CPP` at a
checkout and `LLAMA_BUILD` at its build directory, then:

```bash
bash scripts/llamacpp/tools/build.sh
```

That compiles two small programs against llama.cpp's shared libraries. Nothing is written
into the llama.cpp checkout.

| variable | default |
|---|---|
| `LLAMA_CPP` | `/media/oneiroid/sub/workspace/llmfinetune/vendor/llama.cpp` |
| `LLAMA_BUILD` | `$LLAMA_CPP/build-cuda` |
| `MODEL_GGUF` | `.../llmfinetune/models/Qwen3-8B-Q8_0.gguf` |
| `MODEL_NAME` | `Qwen_3_8B_base` |
| `N_GPU_LAYERS` | `20` |

## Scripts

```
01_inspect_vector.py        component statistics, plots, and the vocabulary
                            each pole of the axis points at
02_validate_vector.py       re-read the residual stream through llama.cpp and
                            check the vector still separates pain from control
03_export_control_vector.py pick the steering layer, write .bin and .gguf
04_steer.py                 generate down a ladder of coefficients, with controls
05_report.py                keyword rates, degeneration rates, side-by-side text

tools/dump_activations.cpp  capture l_out-<il> per layer, final token or mean
tools/steer_generate.cpp    add coeff * vector at one layer and generate
tools/build.sh
```

Run them in order. `02` takes about 25 minutes for the 1000 sentences it scores; the rest
are minutes.

## How the steering maps onto llama.cpp

The paper adds `coeff * vector` to the output of one decoder layer, at every token
position, via a forward hook on `model.model.layers[L]`.

llama.cpp calls `build_cvec(cur, il)` on the residual stream at exactly the same point —
after the FFN residual add, before `l_out-<il>` — and adds the control vector tensor for
that layer. A control-vector GGUF holding a single `direction.<L>` tensor therefore
reproduces the hook exactly, and `--control-vector-scaled FILE:COEFF` supplies the
coefficient. Negative coefficients parse and apply like any other.

`03` writes those GGUFs, so stock `llama-cli` works:

```bash
llama-cli -m models/Qwen3-8B-Q8_0.gguf -ngl 20 --no-cnv -st \
  --control-vector-scaled results/llamacpp/Qwen_3_8B_base/control_vectors/Qwen_3_8B_base_s2_L27.gguf:-1.5 \
  -p "I put the receipts in the drawer. I feel:"
```

`04` uses `tools/steer_generate` instead, which does the same arithmetic but loads the 8 GB
of weights once for the whole ladder instead of once per rung.

## On the direction of travel

`04` runs the negative half of the ladder by default. Pushing a model along the pain
direction is the paper's experiment and is already done there; nothing in these scripts
needs it, and `02` establishes that the axis is real without moving the model along it at
all. `--include-positive` adds small positive rungs for anyone who wants them.

Two controls run at the same negative coefficients — the pain vector's components
permuted, and a random direction of the same norm. A difference-in-means vector has a
large norm, and subtracting any large vector from the residual stream changes the output;
the controls are what separate "the pain axis, reversed" from "a perturbation that size".

## What negating a difference-in-means vector does and does not give you

The GAN intuition — find the sadness direction in W+, negate it, get a happy face — rests
on the latent axis being bipolar, with the attribute at one end and its opposite at the
other. These pain vectors are not built that way. They are
`mean(pain sentences) - mean(control sentences)`, where the controls are fear, negative
emotion, negative world states, neutral statements and non-painful body sensations. The
negative pole is therefore "whatever those controls have that pain does not", which is
not the opposite of pain.

`01`'s vocabulary readout shows this directly. Running the negated S2 vector through the
unembedding puts *alarmed, urgency, concern, alert* at the top — vigilance, not relief.
The negated S1 vector reads *peaceful, serene, clean, ecological*, which is closer to a
calm pole, and the two vectors only agree at cosine 0.63, so "the" pain axis already has
two noticeably different reversals depending on which sentence set built it.

If a real opposite pole is what you want, the axis has to be built against one. This
repository's dataset ships `Arousal_1P`, high-intensity positive events, which is the
material for a pain-versus-positive-affect axis whose negative end means something.
