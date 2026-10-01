# RECIPE — converting Laya's encoder for the QCS6490 NPU (w8a16)

This is the **successful path** only: the exact steps that produce a working quantized encoder for
the Hexagon **V68** HTP at **w8a16** (int8 weights, int16 activations), for both the 128-token
(min-max) and 256-token (**sqnr**) configurations.

Tooling: **Qualcomm QAIRT / QNN 2.37.1.250807**. Model: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).

---

## 0. Why a split, and what runs where

The V68 HTP **ignores runtime integer Gather indices on-chip** (it silently returns the wrong rows),
so the token-embedding lookup and the marker-position gather cannot run on the NPU. The working design
is the standard edge split for an outlier-heavy encoder:

```
text ─► tokenize + embedding lookup            (host CPU, float)
      ─► transformer encoder                     (Hexagon V68 NPU, w8a16, context binary)
      ─► marker gather + scorer head + temp cal  (host CPU, float)
      ─► calibrated typed decision
```

So we only quantize and deploy the **encoder body** (embeddings in → final LayerNorm out). The encoder
takes `inputs_embeds` (already looked up on the host), not `input_ids`.

---

## 1. Export the encoder to ONNX with an embeddings input

`convert/make256.py` does the ONNX surgery:

1. Trace the ModernBERT encoder to ONNX.
2. **Embed-surgery**: replace the `input_ids → Gather(embedding_table)` front with a direct
   `inputs_embeds` input (the lookup moves to the host). The attention-mask → additive-bias path is
   kept and exposed as two mask inputs (`gmask`, `smask`) so the host controls padding.
3. Cut the graph at the encoder output `ENC_OUT = /encoder/final_norm/LayerNormalization_output_0`.
4. Fix all shapes to the target context length `L` (128 or 256) — the HTP wants static shapes.

```bash
python convert/make256.py --ctx 256 --out laya_s256_clean.onnx   # 256-token encoder
python convert/make256.py --ctx 128 --out laya_s128_clean.onnx   # 128-token encoder
```

The head (`marker gather → scorer → temperature`) stays in PyTorch/float on the host — it is never
quantized. `convert/mask_only_128.onnx` (and its 256 twin, produced by the script) is the tiny
attention-mask→bias helper the host runs to build `gmask`/`smask` from an `attention_mask`.

---

## 2. Convert ONNX → DLC

```bash
qairt-converter --input_network laya_s256_clean.onnx --output_path laya_encoder_256.dlc
```

This produces a float DLC. Quantization is the next, decisive step.

---

## 3. Quantize to w8a16 — **the one setting that matters**

Run the quantizer from the `qenv` virtualenv (needs `onnx==1.16.1`):

```bash
# 128-token: min-max is near-lossless
~/qenv/bin/python $(which qairt-quantizer) \
  --input_dlc laya_encoder_128.dlc \
  --output_dlc laya_encoder_mm.dlc \
  --input_list calib128/input_list.txt \
  --act_bitwidth 16 --bias_bitwidth 32 --weight_bitwidth 8

# 256-token: min-max COLLAPSES. You MUST use sqnr calibration:
~/qenv/bin/python $(which qairt-quantizer) \
  --input_dlc laya_encoder_256.dlc \
  --output_dlc laya_encoder_256_sqnr.dlc \
  --input_list calib256/input_list.txt \
  --act_bitwidth 16 --bias_bitwidth 32 --weight_bitwidth 8 \
  --act_quantizer_calibration sqnr          # <<< the fix
```

**Why `sqnr` for long context.** Typed decisions are read off **marker tokens** whose meaning is
carried by a *small RoPE positional signal* (two adjacent markers differ only by their positional
rotation). At 256 tokens, min-max sets activation ranges from distant outliers, which makes the 16-bit
step too coarse to preserve that fine signal — the two markers quantize to the **same** value, the head
sees no difference, and every decision collapses to 0.5 (AUROC 0.00). **SQNR** chooses ranges that
minimise quantization noise on the signal that actually carries information, so the markers stay
distinct (separation 0.00 → 23.37 on-board) and the encoder output tracks float again (hidden-state
Pearson **0.88**), entirely within w8a16 — no FP16, no new silicon.

Calibration-method results on this graph (256-token, marker 16 vs 20 separation):

| calibration | marker separation | usable? |
|---|---|---|
| min-max | 0.00 (collapsed) | ❌ |
| entropy | collapsed | ❌ |
| mse | quantizer segfaults | ❌ |
| **sqnr** | **23.4 (board)** | ✅ |

Generate the calibration set with `convert/calib256.py` (it writes `inputs_embeds` / `gmask` / `smask`
raw tensors plus the `input_list.txt` the quantizer reads). A few dozen representative inputs suffice.

---

## 4. Build the HTP context binary (one-time finalize)

Loading a DLC cold on the board finalizes the HTP graph (~33 s). Do that **once** and cache the result
as a context binary so every later load is ~1 s:

```bash
# on the board
qnn-context-binary-generator \
  --backend /usr/lib/libQnnHtp.so \
  --model   /usr/lib/libQnnModelDlc.so \
  --dlc_path laya_encoder_256_sqnr.dlc \
  --binary_file laya_encoder_256_sqnr_ctx \
  --output_dir ctx256s
```

Then run inference by **retrieving** the context (no re-finalize):

```bash
export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp
qnn-net-run --backend /usr/lib/libQnnHtp.so \
  --retrieve_context ctx256s/laya_encoder_256_sqnr_ctx.bin \
  --input_list il.txt --output_dir out
```

Each line of `il.txt` is one inference: `inputs_embeds:=ie_0.raw gmask:=gm_0.raw smask:=sm_0.raw`.
**Batch many inputs into one `qnn-net-run`** — each `--retrieve_context` invocation creates and destroys
an HTP context, and doing that per-inference in a tight loop can exhaust the DSP (`Device Creation
failure`), which needs a board reboot. One `qnn-net-run` per ~100–200 inputs is safe and fast.

---

## 5. Host side

On the host (see `benchmark/pb256.py` and `benchmark/corr_compare.py`): tokenize, look up embeddings,
build `gmask`/`smask` from the attention mask, send the batch to the board, then run the float head
(`type embedding → head layers → marker gather → scorer → temperature`) on the returned hidden states.
The split is **exact at float level** (parity 0.000000 vs the unsplit model).

---

## Gotchas worth knowing

- **V68 has no FP16 path** — w8a16 is the only option; this whole recipe lives inside it.
- Run the quantizer in the `qenv` venv (`onnx==1.16.1`); newer onnx breaks the importer.
- `--act_quantizer_calibration mse` segfaults the quantizer on this graph; `entropy` collapses like
  min-max. Only `sqnr` works at 256.
- The Adreno GPU backend can't compile the attention subgraph (`OpPackage validation failure` on
  `.../attn/Where`) — the NPU/HTP path is the one that works.
- torch 2.14's `TransformerEncoderLayer` fast-path rejects a batched `src_key_padding_mask`; run the
  **head at batch size 1** (the encoder still batches fine on the NPU).
