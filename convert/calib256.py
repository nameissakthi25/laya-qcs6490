#!/usr/bin/env python3
"""Generate 256-token calibration raws for the encoder DLC quantizer."""
import os, numpy as np, onnxruntime as ort
from laya.agent import Agent
from laya.common import build_sequence, collate_items, QTYPES
os.makedirs("calib256", exist_ok=True)
ag = Agent("convaiinnovations/laya", compile=False, device="cpu"); tok = ag.tok
L, M, H = 256, 64, 1024
W = np.load("embed_table.npy")
STATES = ["My subscription renewed for $40 after the service was down. I want a refund today.",
 "The transfer is still pending and I used the wrong sort code, please stop it.",
 "Battery dies before lunch but the keyboard and screen are excellent.",
 "This is the third time explaining the same missing refund. Get me a person.",
 "Please reset the card PIN, the new one never arrived and the old one is locked.",
 "The deploy left staging tags inconsistent. Production checkout is unaffected.",
 "Guest in 1408 says the AC is out since yesterday and wants to move tonight.",
 "Payroll file must be corrected before the 5pm cutoff or everyone is paid late.",
 "Your mailbox is almost full, click here in the next hour or we delete everything.",
 "I need to move my Friday flight to Saturday morning, same cabin, aisle seat."]
QS = [{"t":"choice","ins":"Which department should handle this request?","crit":{"billing":"payments, refunds","technical":"bugs, outages","other":"everything else"}},
 {"t":"noul","ins":"Does the user request a refund?","crit":{"true":"yes","false":"no"}},
 {"t":"score","ins":"How urgent is this request?","crit":["low","medium","high"]}]
ex = ort.InferenceSession("mask_only_256.onnx", providers=["CPUExecutionProvider"])
eye = np.eye(L, dtype=bool)[None, None]
items = []
for s in STATES:
    for q in QS:
        seq, mk = build_sequence(tok, s, q, L, 64, truncate_left=False)
        items.append({"ids": seq, "markers": mk, "qtype": QTYPES[q["t"]]})
def fix(x, n):
    x = x.numpy(); o = np.zeros((1, n), x.dtype); k = min(x.shape[-1], n); o[0, :k] = x.reshape(-1)[:k]; return o
lines = []
for i, it in enumerate(items[:30]):
    b = collate_items([[it]], tok.pad_token_id)
    ii = fix(b["input_ids"], L).astype(np.int64); am = fix(b["attention_mask"], L)
    g, s = ex.run(None, {"attention_mask": am.astype(np.int64)})
    g = (np.asarray(g).astype(bool) | eye); s = (np.asarray(s).astype(bool) | eye)
    emb = W[ii[0]].reshape(1, L, H).astype(np.float32)
    for n, ar in [("inputs_embeds", emb), ("attention_mask", am.astype(np.int32)),
                  ("gmask", g.astype(np.int32)), ("smask", s.astype(np.int32))]:
        ar.tofile(f"calib256/{n}_emb_{i:03d}.raw")
    lines.append(" ".join(f"{n}:=calib256/{n}_emb_{i:03d}.raw"
                          for n in ["inputs_embeds", "attention_mask", "gmask", "smask"]))
open("calib256/input_list_enc256.txt", "w").write("\n".join(lines) + "\n")
print("calib256 done: 30 samples, gmask bytes =", os.path.getsize("calib256/gmask_emb_000.raw"), flush=True)
