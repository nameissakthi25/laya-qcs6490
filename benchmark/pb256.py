#!/usr/bin/env python3
"""PhishNChips on QCS6490 NPU at 256-token context (w8a16) vs CPU-float. Writes result256.json."""
import os, sys, time, json, glob, tempfile, subprocess, urllib.request, numpy as np, onnxruntime as ort, torch
from types import SimpleNamespace
import pyarrow.parquet as pq
from laya.agent import Agent
from laya.common import ece_score

L, H = 256, 1024
N = int(os.environ.get("N", "80"))
BOARD_HOST="192.168.31.41"; BOARD_USER="root"; BOARD_PASS="oelinux123"; BOARD_DIR="/home/denc"
CTX_BIN = os.environ.get("CTX_BIN", "/home/denc/ctx256s/laya_encoder_256_sqnr_ctx.bin")
MASK = os.path.expanduser("~/robotgate/mask_only_256.onnx")
Q = {"is_phishing": {"type": "noul", "instructions": "Is this email a phishing or scam attempt?"}}
URL = ("https://huggingface.co/datasets/AreLit/PhishNChips/resolve/refs%2Fconvert%2Fparquet/emails/core/0000.parquet")

def _ssh(*c): return ["sshpass","-p",BOARD_PASS,"ssh","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null",
    "-o","ConnectTimeout=20","-o","PreferredAuthentications=password","-o","PubkeyAuthentication=no",f"{BOARD_USER}@{BOARD_HOST}",*c]
def _scp(*a): return ["sshpass","-p",BOARD_PASS,"scp","-r","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null",
    "-o","ConnectTimeout=20","-o","PreferredAuthentications=password","-o","PubkeyAuthentication=no",*a]

def board_encode(embeds, gmask, smask):
    B = embeds.shape[0]
    with tempfile.TemporaryDirectory() as td:
        lines=[]
        for i in range(B):
            embeds[i].astype(np.float32).tofile(f"{td}/ie_{i}.raw"); gmask[i].astype(np.int32).tofile(f"{td}/gm_{i}.raw")
            smask[i].astype(np.int32).tofile(f"{td}/sm_{i}.raw")
            lines.append(f"inputs_embeds:=ie_{i}.raw gmask:=gm_{i}.raw smask:=sm_{i}.raw")
        open(f"{td}/il.txt","w").write("\n".join(lines)+"\n")
        rdir=f"{BOARD_DIR}/req_{int(time.time()*1000)}"
        subprocess.run(_ssh(f"mkdir -p {rdir}"),check=True,timeout=30)
        subprocess.run(_scp(*[f"{td}/{f}" for f in os.listdir(td)],f"{BOARD_USER}@{BOARD_HOST}:{rdir}/"),check=True,timeout=120)
        env="export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp && export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH"
        cmd=f"cd {rdir} && {env} && qnn-net-run --backend /usr/lib/libQnnHtp.so --retrieve_context {CTX_BIN} --input_list il.txt --output_dir out"
        r=subprocess.run(_ssh(cmd),timeout=300,capture_output=True,text=True)
        if r.returncode!=0: subprocess.run(_ssh(f"rm -rf {rdir}"),timeout=30); raise RuntimeError("qnn:"+(r.stderr or r.stdout)[-300:])
        subprocess.run(_scp(f"{BOARD_USER}@{BOARD_HOST}:{rdir}/out",td),check=True,timeout=120)
        subprocess.run(_ssh(f"rm -rf {rdir}"),timeout=30)
        out=np.zeros((B,L,H),np.float32)
        for i in range(B): out[i]=np.fromfile(glob.glob(f"{td}/out/Result_{i}/*.raw")[0],np.float32).reshape(L,H)
        return out

class BoardEncoder(torch.nn.Module):
    def __init__(self, emb, cfg, mask, pad_id):
        super().__init__(); self.config=cfg; self.emb=emb.detach().cpu().numpy().astype(np.float32)
        self.mask=mask; self.pad_id=int(pad_id); self.eye=np.eye(L,dtype=bool)[None,None]
    def forward(self, input_ids=None, attention_mask=None, **kw):
        ids=input_ids.cpu().numpy().astype(np.int64); am=attention_mask.cpu().numpy().astype(np.int64)
        B,seq=ids.shape
        if seq<L: ids=np.pad(ids,((0,0),(0,L-seq)),constant_values=self.pad_id); am=np.pad(am,((0,0),(0,L-seq)),constant_values=0)
        elif seq>L: ids=ids[:,:L]; am=am[:,:L]
        embeds=self.emb[ids].reshape(B,L,H); gs=[];ss=[]
        for i in range(B):
            g1,s1=self.mask.run(None,{"attention_mask":am[i:i+1]}); gs.append(np.asarray(g1)); ss.append(np.asarray(s1))
        g=(np.concatenate(gs,0).astype(bool)|self.eye).astype(np.int32); s=(np.concatenate(ss,0).astype(bool)|self.eye).astype(np.int32)
        hid=board_encode(embeds,g,s)
        return SimpleNamespace(last_hidden_state=torch.from_numpy(hid[:,:seq,:]).to(input_ids.device))

def load_core():
    p="/tmp/phish_core.parquet"
    if not os.path.exists(p): urllib.request.urlretrieve(URL,p)
    t=pq.read_table(p).to_pydict(); return list(zip(t["email_content"],[int(x) for x in t["phish_label"]]))

def metrics(name,probs,labels):
    probs=np.array(probs); labels=np.array(labels); preds=(probs>=.5).astype(int)
    acc=float((preds==labels).mean()); rec=float(preds[labels==1].mean())
    conf=np.maximum(probs,1-probs); ece=float(ece_score(conf,(preds==labels).astype(float)))
    order=np.argsort(probs); ranks=np.empty(len(probs)); ranks[order]=np.arange(1,len(probs)+1)
    npos=labels.sum(); nneg=len(labels)-npos
    au=float((ranks[labels==1].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else float("nan")
    print(f"{name:<14} acc={acc:.3f} AUROC={au:.3f} recall={rec:.3f} ECE={ece:.3f}",flush=True)
    return {"acc":acc,"auroc":au,"recall":rec,"ece":ece}

def main():
    data=load_core(); ph=[e for e in data if e[1]==1][:N//2]; lg=[e for e in data if e[1]==0][:N//2]
    rows=ph+lg; emails=[e[0] for e in rows]; labels=[e[1] for e in rows]
    print(f"{len(emails)} emails @ 256-token context | CTX_BIN={CTX_BIN}",flush=True)
    ag=Agent("convaiinnovations/laya",compile=False,device="cpu"); ag.model.eval()
    orig=ag.model.encoder; emb=orig.get_input_embeddings().weight; cfg=orig.config
    board=BoardEncoder(emb,cfg,ort.InferenceSession(MASK,providers=["CPUExecutionProvider"]),ag.tok.pad_token_id)
    def run(enc,tag):
        ag.model.encoder=enc; out=[]; t0=time.perf_counter()
        for i,e in enumerate(emails):
            with torch.no_grad(): r=ag.predict(e,Q,max_len=256)
            out.append(float(r["answers"]["is_phishing"]["noul"]))
            if (i+1)%10==0: print(f"  {tag} {i+1}/{len(emails)} ({(time.perf_counter()-t0)/(i+1):.2f}s/email)",flush=True)
        return out
    print("[CPU-float 256]",flush=True); fp=run(orig,"float")
    print("[QCS6490 NPU 256]",flush=True); npu=run(board,"npu")
    print("  npu noul samples:", [round(x,4) for x in npu[:6]], flush=True)
    print("\n=== PhishNChips core @ 256 tokens — Laya base ===")
    mf=metrics("CPU-float",fp,labels); mn=metrics("QCS6490 NPU",npu,labels)
    print("published (T4 base, full ctx): acc=0.505 AUROC=0.678")
    corr=float(np.corrcoef(fp,npu)[0,1]); agr=float(((np.array(fp)>=.5)==(np.array(npu)>=.5)).mean())
    print(f"NPU vs float: prob r={corr:.3f}, agreement={agr:.3f}")
    json.dump({"n":len(emails),"ctx":256,"float":mf,"npu":mn,"corr":corr,"agreement":agr},
              open(os.path.expanduser("~/robotgate/result256.json"),"w"),indent=1)
    print("wrote result256.json",flush=True)

if __name__=="__main__": main()
