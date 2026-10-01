#!/usr/bin/env python3
"""
Correlation comparison: NPU-quantised encoder (w8a16) vs exact CPU-float, per DLC variant.
Runs the SAME 10 calibration states x 3 questions through:
  - reference: original torch ModernBERT encoder + laya head (all float, CPU)
  - each board DLC: encoder on QCS6490 Hexagon NPU + the SAME float head on host
and reports Pearson r + decision agreement over all valid marker slots.
"""
import os, sys, time, glob, tempfile, subprocess
from types import SimpleNamespace
import numpy as np, torch, onnxruntime as ort
from laya.agent import Agent
from laya.common import build_sequence, collate_items, QTYPES

BOARD_HOST="192.168.31.41"; BOARD_USER="root"; BOARD_PASS="oelinux123"; BOARD_DIR="/home/denc"
BACKEND="libQnnHtp.so"; L,H=128,1024
MASK_ONNX=os.path.expanduser("~/robotgate/mask_only_128.onnx")
VARIANTS=[("min-max (orig)","laya_encoder_mm.dlc"),("min-max (rebuilt)","laya_encoder_mm2.dlc"),
          ("smoothquant","laya_encoder_sq.dlc")]

# --- the exact calibration set (from calib128.py) ---
STATES=["My subscription renewed for $40 after the service was down. I want a refund today.",
 "The transfer is still pending and I used the wrong sort code, please stop it.",
 "Battery dies before lunch but the keyboard and screen are excellent.",
 "This is the third time explaining the same missing refund. Get me a person.",
 "Please reset the card PIN, the new one never arrived and the old one is locked.",
 "The deploy left staging tags inconsistent. Production checkout is unaffected.",
 "Guest in 1408 says the AC is out since yesterday and wants to move tonight.",
 "Payroll file must be corrected before the 5pm cutoff or everyone is paid late.",
 "Your mailbox is almost full, click here in the next hour or we delete everything.",
 "I need to move my Friday flight to Saturday morning, same cabin, aisle seat."]
QS=[{"t":"choice","ins":"Which department should handle this request?","crit":{"billing":"payments, refunds","technical":"bugs, outages","other":"everything else"}},
 {"t":"noul","ins":"Does the user request a refund?","crit":{"true":"yes","false":"no"}},
 {"t":"score","ins":"How urgent is this request?","crit":["low","medium","high"]}]

def ssh(*c):
    return ["sshpass","-p",BOARD_PASS,"ssh","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null",
            "-o","ConnectTimeout=20","-o","PreferredAuthentications=password","-o","PubkeyAuthentication=no",
            f"{BOARD_USER}@{BOARD_HOST}",*c]
def scp(*a):
    return ["sshpass","-p",BOARD_PASS,"scp","-r","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null",
            "-o","ConnectTimeout=20","-o","PreferredAuthentications=password","-o","PubkeyAuthentication=no",*a]

def run_board(dlc, embeds, gmask, smask):
    B=embeds.shape[0]
    with tempfile.TemporaryDirectory() as td:
        lines=[]
        for i in range(B):
            embeds[i].astype(np.float32).tofile(f"{td}/ie_{i}.raw")
            gmask[i].astype(np.int32).tofile(f"{td}/gm_{i}.raw")
            smask[i].astype(np.int32).tofile(f"{td}/sm_{i}.raw")
            lines.append(f"inputs_embeds:=ie_{i}.raw gmask:=gm_{i}.raw smask:=sm_{i}.raw")
        open(f"{td}/il.txt","w").write("\n".join(lines)+"\n")
        rdir=f"{BOARD_DIR}/cmp_{int(time.time()*1000)}"
        subprocess.run(ssh(f"mkdir -p {rdir}"),check=True,timeout=30)
        files=[os.path.join(td,f) for f in os.listdir(td)]
        subprocess.run(scp(*files,f"{BOARD_USER}@{BOARD_HOST}:{rdir}/"),check=True,timeout=300)
        cmd=(f"cd {rdir} && export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp && "
             f"export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH && "
             f"qnn-net-run --backend /usr/lib/libQnnHtp.so --model /usr/lib/libQnnModelDlc.so "
             f"--dlc_path {BOARD_DIR}/{dlc} --input_list il.txt --output_dir out")
        r=subprocess.run(ssh(cmd),timeout=600,capture_output=True,text=True)
        if r.returncode!=0:
            subprocess.run(ssh(f"rm -rf {rdir}"),timeout=30)
            raise RuntimeError("qnn-net-run failed:\n"+(r.stderr or r.stdout)[-600:])
        subprocess.run(scp(f"{BOARD_USER}@{BOARD_HOST}:{rdir}/out",td),check=True,timeout=180)
        subprocess.run(ssh(f"rm -rf {rdir}"),timeout=30)
        out=np.zeros((B,L,H),np.float32)
        for i in range(B):
            out[i]=np.fromfile(glob.glob(f"{td}/out/Result_{i}/*.raw")[0],np.float32).reshape(L,H)
        return out

class BoardEnc(torch.nn.Module):
    def __init__(self, emb, cfg, mask, dlc):
        super().__init__(); self.config=cfg; self.emb=emb.detach().cpu().numpy().astype(np.float32)
        self.mask=mask; self.dlc=dlc; self.eye=np.eye(L,dtype=bool)[None,None]
    def forward(self, input_ids=None, attention_mask=None, **kw):
        ids=input_ids.cpu().numpy().astype(np.int64); am=attention_mask.cpu().numpy().astype(np.int64)
        B=ids.shape[0]; embeds=self.emb[ids].reshape(B,L,H)
        gs=[]; ss=[]
        for i in range(B):                      # mask_only_128.onnx is batch-1
            g1,s1=self.mask.run(None,{"attention_mask":am[i:i+1]})
            gs.append(np.asarray(g1)); ss.append(np.asarray(s1))
        g=np.concatenate(gs,0); s=np.concatenate(ss,0)      # [B,1,L,L]
        g=(g.astype(bool)|self.eye).astype(np.int32); s=(s.astype(bool)|self.eye).astype(np.int32)
        hid=run_board(self.dlc,embeds,g,s)
        return SimpleNamespace(last_hidden_state=torch.from_numpy(hid).to(input_ids.device))

def build_batch(ag):
    tok=ag.tok; items=[]
    for s in STATES:
        for q in QS:
            seq,markers=build_sequence(tok,s,q,L,64,truncate_left=False)
            items.append({"ids":seq,"markers":markers,"qtype":QTYPES[q["t"]]})
    b=collate_items([[it] for it in items], tok.pad_token_id)
    # board encoder DLC has fixed seq length L=128; pad the batch to 128 (as calibration did)
    n,cur=b["input_ids"].shape
    if cur<L:
        pad=L-cur
        b["input_ids"]=torch.cat([b["input_ids"],torch.full((n,pad),tok.pad_token_id,dtype=torch.long)],1)
        b["attention_mask"]=torch.cat([b["attention_mask"],torch.zeros((n,pad),dtype=torch.long)],1)
    elif cur>L:
        b["input_ids"]=b["input_ids"][:,:L]; b["attention_mask"]=b["attention_mask"][:,:L]
    return b, items

@torch.no_grad()
def logits_of(ag,b):
    lg,_=ag.model(b["input_ids"],b["attention_mask"],b["marker_pos"],b["marker_mask"],b["qtype"])
    return lg.float().cpu().numpy()

def compare(ref,q,mm):
    A=[];B=[];agree=0;n=0
    for i in range(ref.shape[0]):
        m=mm[i].cpu().numpy().astype(bool)
        A.append(ref[i][m]);B.append(q[i][m])
        if ref[i][m].argmax()==q[i][m].argmax():agree+=1
        n+=1
    A=np.concatenate(A);B=np.concatenate(B)
    r=float(np.corrcoef(A,B)[0,1]); rmse=float(np.sqrt(np.mean((A-B)**2)))
    return r,rmse,agree,n

def main():
    print("loading laya (CPU float reference)...",flush=True)
    ag=Agent("convaiinnovations/laya",compile=False,device="cpu"); ag.model.eval()
    orig=ag.model.encoder
    emb=orig.get_input_embeddings().weight; cfg=orig.config
    mask=ort.InferenceSession(MASK_ONNX,providers=["CPUExecutionProvider"])
    b,items=build_batch(ag)
    print(f"built {len(items)} items (10 states x 3 questions)",flush=True)
    ref=logits_of(ag,b)
    present=subprocess.run(ssh("ls "+BOARD_DIR),capture_output=True,text=True,timeout=30).stdout
    print("\n=== NPU w8a16 encoder vs CPU-float (all valid marker slots) ===")
    print(f"{'variant':<12}{'pearson r':>11}{'RMSE':>9}{'decisions':>12}")
    for name,dlc in VARIANTS:
        if dlc not in present:
            print(f"{name:<12}{'(dlc not on board, skipped)':>32}"); continue
        t0=time.time()
        ag.model.encoder=BoardEnc(emb,cfg,mask,dlc)
        try:
            q=logits_of(ag,b)
            r,rmse,ag_ok,n=compare(ref,q,b["marker_mask"])
            print(f"{name:<12}{r:>11.4f}{rmse:>9.3f}{ag_ok:>7}/{n:<4}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"{name:<12} ERROR: {e}")
        finally:
            ag.model.encoder=orig
    print("\ndone",flush=True)

if __name__=="__main__":
    main()
