# Laya on the QCS6490 NPU

Running **Laya** — Convai's 421M-parameter ModernBERT **typed-decision** model
([`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)) — on a Qualcomm
**QCS6490** Hexagon **V68** NPU at **w8a16**, with real benchmarks and a reproducible conversion recipe.

Laya returns **calibrated typed decisions** (choice / yes-no / score with probabilities), not text —
ideal for an edge decision engine: fast, cheap, private, and it knows when it's unsure.

## Results (measured on the physical board)

| | QCS6490 NPU (V68) | Tesla T4 GPU (Laya published) |
|---|---|---|
| **Latency / decision** | ~84 ms | 39.5 ms |
| **Power** | ~6 W (whole SoC) | ~70 W |
| **Energy / decision** | ~0.5 J | ~2.8 J |

| measure (NPU w8a16 vs float) | @128 | @256 min-max | @256 **sqnr** |
|---|---|---|---|
| encoder hidden-state fidelity (Pearson) | **0.87** | — | **0.88** (0.82 at markers) |
| decision agreement | **100%** | — | — |
| marker 16↔20 separation (must be ≠0) | ok | **0.00 — collapse** | **23.4 ✅** |

→ **~2× a datacenter GPU's latency at ~1/10 the power** (~5× better energy/decision), **near-lossless
w8a16 at 128**, and — with the right calibration — a **faithful 256-token encoder on the edge NPU**
(hidden-state r≈0.88) where naïve quantization collapses outright.

## The one thing that matters most: calibration method

Typed decisions read **marker tokens** whose meaning is carried by their **RoPE positional signal**
(two adjacent markers differ *only* by their positional rotation). Under w8a16:

- **Short context (≤128):** use **min-max** — near-lossless (hidden-state r 0.87, 100% decision
  agreement). sqnr does *not* help here (logit fidelity 0.77 vs min-max 0.80).
- **Long context (256+):** min-max's ranges (set by activation outliers) make the 16-bit step too
  coarse to preserve the positional signal — the two markers quantize to the **same** value (separation
  0.00) and every decision collapses to 0.5 (AUROC 0.00). **`--act_quantizer_calibration sqnr`** keeps a
  fine enough step: the markers separate (23.4 on-board) and the encoder tracks float again
  (hidden-state r 0.88), **entirely within w8a16** — no FP16, no new silicon.

### On the phishing benchmark

We also ran PhishNChips end-to-end (float AUROC 0.69 @256, matching Laya's published 0.678 base; NPU
0.80). **We don't headline these**: on this core subset *email length alone* scores AUROC 0.76, so the
task is length-confounded and is not a clean measure of model or quantization quality. The clean
fidelity signal is the **direct hidden-state comparison** above. See [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

See [`docs/RECIPE.md`](docs/RECIPE.md) for the full conversion and [`docs/BENCHMARK.md`](docs/BENCHMARK.md)
for the numbers and method.

## Architecture (the edge split)

```
text → tokenize + embedding lookup (host)
     → encoder  ── Hexagon V68 NPU · w8a16 · context binary ──►  hidden states
     → decision head + temperature calibration (host, float)
     → calibrated typed decision
```

The **encoder runs on the NPU**; embedding lookup, the marker/scorer **head**, and temperature
calibration run in **float on the host CPU**. (V68 ignores runtime Gather indices on-chip, so the
lookups must stay on the host — the standard edge split for outlier-heavy encoders. The split is
exact at float level.)

## Layout
```
convert/     ONNX surgery + calibration generation for the 256-token encoder
benchmark/   phishing accuracy, quant-fidelity correlation, and the QNN-emulator eval
docs/        RECIPE.md (reproducible conversion), BENCHMARK.md (results + method)
```

## Models
The quantized encoder DLCs are published on the Hugging Face Hub:
**https://huggingface.co/nameissakthi/laya-qcs6490-encoder**
(`laya_encoder_mm.dlc` for 128-token min-max, `laya_encoder_256_sqnr.dlc` for 256-token sqnr, plus the
`mask_only_*.onnx` host helpers). The HTP **context binaries** are board- and QAIRT-version-specific;
regenerate them from the DLCs with `qnn-context-binary-generator` (see [`docs/RECIPE.md`](docs/RECIPE.md)).

## Credits
Model: [Convai Innovations — Laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0).
Benchmark task: [AreLit/PhishNChips](https://huggingface.co/datasets/AreLit/PhishNChips) and the
[Luni/laya-jev-benchmark](https://huggingface.co/datasets/Luni/laya-jev-benchmark) suite.
Tooling: Qualcomm QAIRT / QNN 2.37.
