# BENCHMARK — Laya on the QCS6490 NPU

All NPU numbers are measured on the **physical board** (Advantech QCS6490, Hexagon V68 HTP, w8a16),
through the host-split pipeline in [`RECIPE.md`](RECIPE.md). "Float" is the unquantized model on CPU.

Scripts: `benchmark/pb256.py` (phishing @256), `benchmark/corr_compare.py` (quant fidelity),
`benchmark/qeval.py` (QNN HTP x86-emulator eval, for fast calibration sweeps without the board).

**TL;DR of what is and isn't a clean result:** the solid, defensible numbers are **latency/energy**,
the **@128 near-lossless fidelity**, the **256 min-max collapse → sqnr fix**, and the **direct 256
encoder fidelity (hidden-state r≈0.88)**. The end-to-end phishing AUROC is **length-confounded** on this
dataset and is reported only for completeness, not as a headline.

---

## 1. Latency & energy — one typed decision

| | QCS6490 NPU (V68) | Tesla T4 GPU (Laya published) |
|---|---|---|
| per decision | **~84 ms** (73 ms NPU encoder + ~11 ms host head) | 39.5 ms |
| context load (one-time) | ~0.4–1.2 s (cached context binary) | — |
| board power | ~6 W (whole SoC) | ~70 W |
| **energy / decision** | **~0.5 J** | ~2.8 J |

Method: `qnn-net-run --retrieve_context`; the wall-clock delta between a 1-input and a 10-input list
isolates ~73 ms/inference from the one-time context load. Head + tokenize timed on the host CPU.

→ **~2× a datacenter GPU's latency at ~1/10 the power** (~5× better energy per decision).

---

## 2. Quantization fidelity — the clean signal (NPU/emulator vs float, same inputs)

This is the honest measure of what quantization costs: compare the model's own outputs, NPU vs float,
on identical inputs. No labels, no task.

| measure | @128 | @256 **sqnr** |
|---|---|---|
| **encoder hidden states, marker positions** (Pearson) | — | **0.82** |
| **encoder hidden states, all real tokens** (Pearson) | — | **0.88** (per-email mean 0.88, worst 0.80) |
| phishing decision (noul) vs float | r **0.87**, **100%** agreement | see note ▼ |
| marker-logit corr, min-max (emulator, 10 inp) | r **0.80**, 7/10 | — |
| marker-logit corr, sqnr (emulator, 10 inp) | r **0.77**, 7/10 | — |

- **@128 is near-lossless**: the NPU decision agrees with float on **100%** of phishing cases (r 0.87).
- **@256 with sqnr the encoder is faithful**: hidden-state correlation **0.82 at markers, 0.88
  overall** — a clean, task-independent fidelity number measured directly on the board
  (`fid256`, 60 emails, 120 marker positions).
- **sqnr does not lift the ~0.77 ceiling at 128** (0.77 vs min-max's 0.80) — the ceiling is intrinsic
  to w8a16 on this graph, not a calibration choice. Use min-max at ≤128, sqnr at 256+.

▼ *Note on the @256 noul correlation.* End-to-end, `corr(NPU noul, float noul)` over 2000 phishing
emails is only **0.36 Pearson** — but this is a **downstream artifact, not encoder infidelity**. On
this task the markers softmax to near-zero noul for almost every email (the model rarely fires), so
the probabilities cluster near 0 and tiny differences destroy Pearson. The *encoder* that produces
those logits tracks float at r 0.82–0.88 (above). Fidelity should be judged on the hidden states, not
on squashed near-zero probabilities.

---

## 3. The 256 marker-collapse — root cause & fix (the core result)

Two adjacent `[MASK]` marker tokens differ only by their RoPE positional rotation. We measured their
separation (`max|hidden[16] − hidden[20]|`) through each path:

| path | marker 16 vs 20 separation |
|---|---|
| full float model | 5.48 (differentiated) |
| float ONNX encoder | 5.48 (exact, Δ 0.0001 vs full) |
| min-max w8a16 — emulator | 0.00 (collapsed) |
| min-max w8a16 — **board** | 0.00 (collapsed) |
| **sqnr w8a16 — emulator** | **23.07 (differentiated)** |
| **sqnr w8a16 — board** | **23.37 (differentiated)** |

min-max quantizes the two markers to the *same* value → the head cannot tell them apart → every
decision is 0.5 (AUROC 0.00). sqnr preserves the separation → decisions recover, and the full encoder
output stays faithful to float (§2). The emulator reproduces the board on both (collapse under min-max,
recovery under sqnr), which is what made the fast calibration sweep possible.

---

## 4. Phishing (PhishNChips core) — reported with its caveat

| config | CPU-float | QCS6490 NPU (w8a16) | note |
|---|---|---|---|
| @128 tokens | AUROC 0.36 | AUROC 0.39 | 128 too short for this task |
| @256, min-max | 0.69 | **0.00 — collapse** | marker collapse |
| @256, **sqnr** | 0.69 | 0.80 | **both beaten by a length baseline ↓** |
| **email token-length alone** | **0.76** | — | the confound |
| Laya published base (T4, full ctx) | 0.678 | — | matches our float 0.69 |

**Read this carefully.** On this core subset, *email length alone* is a **0.76** phishing predictor —
higher than float's 0.69 and near the NPU's 0.80. So the end-to-end AUROC is **confounded by length**
and is **not** a reliable measure of model capability or NPU fidelity. Our full-set float (0.69) matches
Laya's published base (0.678), confirming the number is real; the NPU's apparent 0.80 largely reflects
it tracking length-correlated signal slightly more than float does. **We do not claim the NPU "beats"
float.** For fidelity, see §2; for the capability result, see §3.

(An N=40 sample of this task read AUROC 0.83 for both float and NPU — small-sample noise that the full
2000-email run corrected. A cautionary tale for trusting small evals.)

---

## Reproduce

```bash
python benchmark/pb256.py                      # phishing @256 on the board; N via env; result256.json
python benchmark/corr_compare.py               # quant fidelity, float vs NPU
python benchmark/qeval.py laya_encoder_sqnr.dlc SQNR-128   # calibration sweep via x86 HTP emulator
```
