#!/usr/bin/env python3
"""Direct encoder-fidelity @256: NPU (sqnr) hidden states vs float, at marker positions and overall.
No task, no labels, no softmax — a clean quantization-fidelity number. Writes resultfid.json."""
import os, json, numpy as np, onnxruntime as ort, torch, urllib.request
import pyarrow.parquet as pq
from laya.agent import Agent
from laya.common import build_sequence
import pb256

L, H = 256, 1024
N = int(os.environ.get("N", "60")); CHUNK = int(os.environ.get("CHUNK", "60"))
qdef = {"t": "noul", "ins": "Is this email a phishing or scam attempt?", "crit": {"true": "yes", "false": "no"}}
ag = Agent("convaiinnovations/laya", compile=False, device="cpu"); ag.model.eval()
tok = ag.tok; enc = ag.model.encoder
W = enc.get_input_embeddings().weight.detach().numpy().astype(np.float32)
mask = ort.InferenceSession(pb256.MASK, providers=["CPUExecutionProvider"]); eye = np.eye(L, dtype=bool)[None, None]
MASKID = tok.mask_token_id

def load_core():
    p = "/tmp/phish_core.parquet"
    if not os.path.exists(p): urllib.request.urlretrieve("https://huggingface.co/datasets/AreLit/PhishNChips/resolve/refs%2Fconvert%2Fparquet/emails/core/0000.parquet", p)
    t = pq.read_table(p).to_pydict(); return list(zip(t["email_content"], [int(x) for x in t["phish_label"]]))

data = load_core(); emails = ([e for e, l in data if l == 1][:N // 2] + [e for e, l in data if l == 0][:N // 2])
print(f"{len(emails)} emails — direct encoder fidelity @256 (sqnr)", flush=True)

def pearson(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
def cos(a, b):
    a = a.ravel(); b = b.ravel(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

npu_mk = []; fl_mk = []; npu_tok = []; fl_tok = []; per_email_r = []
for i in range(0, len(emails), CHUNK):
    chunk = emails[i:i + CHUNK]; E = []; G = []; S = []; META = []
    for e in chunk:
        seq, markers = build_sequence(tok, e, qdef, L, 192, truncate_left=False)
        n = min(len(seq), L); ids = np.zeros(L, np.int64); ids[:n] = seq[:n]
        am = np.zeros(L, np.int64); am[:n] = 1
        g, s = mask.run(None, {"attention_mask": am[None]})
        E.append(W[ids]); G.append((np.asarray(g).astype(bool) | eye).astype(np.int32)[0])
        S.append((np.asarray(s).astype(bool) | eye).astype(np.int32)[0])
        mk = [m for m in markers if m < n]
        META.append((ids, am, n, mk))
    npu_hid = pb256.board_encode(np.stack(E), np.stack(G), np.stack(S))  # [B,256,H]
    for j, (ids, am, n, mk) in enumerate(META):
        with torch.no_grad():
            fh = enc(input_ids=torch.from_numpy(ids[:n])[None], attention_mask=torch.from_numpy(am[:n])[None]).last_hidden_state[0].numpy()
        nh = npu_hid[j][:n]
        per_email_r.append(pearson(nh, fh))
        npu_tok.append(nh); fl_tok.append(fh)
        if mk:
            npu_mk.append(npu_hid[j][mk]); fl_mk.append(fh[mk])
    print(f"  {min(i+CHUNK,len(emails))}/{len(emails)} done", flush=True)

NT = np.concatenate(npu_tok, 0); FT = np.concatenate(fl_tok, 0)
NM = np.concatenate(npu_mk, 0); FM = np.concatenate(fl_mk, 0)
res = {
    "n": len(emails),
    "marker_hidden_pearson": pearson(NM, FM),
    "marker_hidden_cosine": cos(NM, FM),
    "alltoken_hidden_pearson": pearson(NT, FT),
    "alltoken_hidden_cosine": cos(NT, FT),
    "per_email_pearson_mean": float(np.mean(per_email_r)),
    "per_email_pearson_min": float(np.min(per_email_r)),
    "n_markers": int(NM.shape[0]),
}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(os.path.expanduser("~/robotgate/resultfid.json"), "w"), indent=1)
print("wrote resultfid.json", flush=True)
