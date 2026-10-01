#!/usr/bin/env python3
"""Build the 256-token encoder ONNX + 256 mask model, mirroring the 128 pipeline."""
import onnx
from onnx import helper, TensorProto
L, H = 256, 1024

# --- 1) embedding surgery: laya_s256_htp.onnx -> laya_s256_emb.onnx (input_ids Gather -> inputs_embeds) ---
m = onnx.load("laya_s256_htp.onnx"); g = m.graph
GE = [n for n in g.node if n.op_type == "Gather" and "tok_embeddings/Gather" in n.name]
assert len(GE) == 1, [n.name for n in g.node if n.op_type == "Gather"]
gath = GE[0]; out = gath.output[0]
g.node.remove(gath)
for n in g.node:
    n.input[:] = ["inputs_embeds" if i == out else i for i in n.input]
g.input.extend([helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, L, H])])
used = set(o.name for o in g.output)
for n in g.node:
    for i in n.input: used.add(i)
keepin = [i for i in g.input if i.name in used or i.name == "inputs_embeds"]
del g.input[:]; g.input.extend(keepin)
keepi = [i for i in g.initializer if i.name in used]
del g.initializer[:]; g.initializer.extend(keepi)
onnx.save(m, "laya_s256_emb.onnx")
print("laya_s256_emb.onnx inputs:", [i.name for i in g.input], flush=True)

# --- 2) split out the encoder: inputs_embeds/attention_mask/gmask/smask -> encoder final norm ---
m2 = onnx.load("laya_s256_emb.onnx"); g2 = m2.graph
ENC_OUT = "/encoder/final_norm/LayerNormalization_output_0"
names = {vi.name for vi in list(g2.value_info) + list(g2.output)} | {o for n in g2.node for o in n.output}
assert ENC_OUT in names, "encoder final-norm tensor not found; candidates: " + \
    str([x for x in names if "final_norm" in x][:5])
onnx.utils.extract_model("laya_s256_emb.onnx", "laya_encoder_256.onnx",
    ["inputs_embeds", "attention_mask", "gmask", "smask"], [ENC_OUT])
enc = onnx.load("laya_encoder_256.onnx")
print("laya_encoder_256.onnx  in:", [i.name for i in enc.graph.input],
      "out:", [o.name for o in enc.graph.output], flush=True)

# --- 3) 256 mask builder from the clean (pre-mask-injection) model ---
onnx.utils.extract_model("laya_s256_clean.onnx", "mask_only_256.onnx",
    ["attention_mask"], ["/encoder/Expand_output_0", "/encoder/Expand_1_output_0"])
print("mask_only_256.onnx written", flush=True)
