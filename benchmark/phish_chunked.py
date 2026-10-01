#!/usr/bin/env python3
"""Full PhishNChips @256 on the NPU (sqnr), WEDGE-SAFE: one qnn-net-run per CHUNK emails
(few context creations), head run through ag.predict via capture→replay (exact). Writes resultfull.json."""
import os, time, json, numpy as np, onnxruntime as ort, torch, urllib.request
from types import SimpleNamespace
import pyarrow.parquet as pq
from laya.agent import Agent
from laya.common import ece_score
import pb256   # board_encode (sqnr ctx256s), MASK

L, H = 256, 1024
N = int(os.environ.get("N", "2000")); CHUNK = int(os.environ.get("CHUNK", "100"))
Qpub = {"is_phishing": {"type": "noul", "instructions": "Is this email a phishing or scam attempt?"}}
ag = Agent("convaiinnovations/laya", compile=False, device="cpu"); ag.model.eval()
emb = ag.model.encoder.get_input_embeddings().weight; cfg = ag.model.encoder.config
W = emb.detach().numpy().astype(np.float32)
mask = ort.InferenceSession(pb256.MASK, providers=["CPUExecutionProvider"]); eye = np.eye(L, dtype=bool)[None, None]

cap = []          # capture ag.predict's exact (input_ids, attention_mask) per encoder call
class CaptureEnc(torch.nn.Module):
    def __init__(s): super().__init__(); s.config = cfg
    def forward(s, input_ids=None, attention_mask=None, **kw):
        cap.append((input_ids.cpu().numpy()[0].copy(), attention_mask.cpu().numpy()[0].copy()))
        return SimpleNamespace(last_hidden_state=torch.zeros(input_ids.shape[0], input_ids.shape[1], H))
class ReplayEnc(torch.nn.Module):
    def __init__(s): super().__init__(); s.config = cfg; s.hidden = None
    def forward(s, input_ids=None, attention_mask=None, **kw):
        seq = input_ids.shape[1]
        return SimpleNamespace(last_hidden_state=torch.from_numpy(s.hidden[:seq])[None].to(input_ids.device))

def load_core():
    p = "/tmp/phish_core.parquet"
    if not os.path.exists(p): urllib.request.urlretrieve("https://huggingface.co/datasets/AreLit/PhishNChips/resolve/refs%2Fconvert%2Fparquet/emails/core/0000.parquet", p)
    t = pq.read_table(p).to_pydict(); return list(zip(t["email_content"], [int(x) for x in t["phish_label"]]))

data = load_core(); ph = [e for e, l in data if l == 1][:N // 2]; lg = [e for e, l in data if l == 0][:N // 2]
emails = ph + lg; labels = [1] * len(ph) + [0] * len(lg)
print(f"{len(emails)} emails, CHUNK={CHUNK} (one qnn-net-run per chunk, capture→replay head)", flush=True)

# pass 1: capture ag.predict's exact encoder inputs (no board, fast)
ag.model.encoder = CaptureEnc()
with torch.no_grad():
    for e in emails: ag.predict(e, Qpub, max_len=256)
assert len(cap) == len(emails), (len(cap), len(emails))
print(f"captured {len(cap)} encoder inputs", flush=True)

# build board inputs from the captured ids/masks
def masks_for(am):
    g, s = mask.run(None, {"attention_mask": am[None].astype(np.int64)})
    return (np.asarray(g).astype(bool) | eye).astype(np.int32)[0], (np.asarray(s).astype(bool) | eye).astype(np.int32)[0]

replay = ReplayEnc(); ag.model.encoder = replay
npu = []; t0 = time.time()
for i in range(0, len(emails), CHUNK):
    idxs = range(i, min(i + CHUNK, len(emails)))
    E = []; G = []; S = []
    for k in idxs:
        ids, am = cap[k]
        idp = np.zeros(L, np.int64); n = min(len(ids), L); idp[:n] = ids[:n]
        amp = np.zeros(L, np.int64); amp[:n] = am[:n]
        g, s = masks_for(amp); E.append(W[idp]); G.append(g); S.append(s)
    hidden = pb256.board_encode(np.stack(E), np.stack(G), np.stack(S))   # ONE context creation for the chunk
    for jj, k in enumerate(idxs):
        replay.hidden = hidden[jj]
        with torch.no_grad(): r = ag.predict(emails[k], Qpub, max_len=256)
        npu.append(float(r["answers"]["is_phishing"]["noul"]))
    done = min(i + CHUNK, len(emails)); print(f"  {done}/{len(emails)} ({(time.time()-t0)/done:.2f}s/email)", flush=True)
    np.save(os.path.expanduser("~/robotgate/ckpt_chunked.npy"), np.array([npu, labels[:len(npu)]]))

probs = np.array(npu); la = np.array(labels); preds = (probs >= .5).astype(int)
acc = float((preds == la).mean()); rec = float(preds[la == 1].mean())
ece = float(ece_score(np.maximum(probs, 1 - probs), (preds == la).astype(float)))
o = np.argsort(probs); rk = np.empty(len(probs)); rk[o] = np.arange(1, len(probs) + 1); npo, nn = la.sum(), len(la) - la.sum()
au = float((rk[la == 1].sum() - npo * (npo + 1) / 2) / (npo * nn))
print(f"\n=== PhishNChips FULL @256 NPU (sqnr) N={len(emails)} ===\nacc={acc:.3f} AUROC={au:.3f} recall={rec:.3f} ECE={ece:.3f}", flush=True)
json.dump({"n": len(emails), "ctx": 256, "calibration": "sqnr", "acc": acc, "auroc": au, "recall": rec, "ece": ece},
          open(os.path.expanduser("~/robotgate/resultfull.json"), "w"), indent=1)
print("wrote resultfull.json", flush=True)
