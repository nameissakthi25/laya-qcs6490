#!/usr/bin/env python3
"""
Evaluate a quantized encoder DLC on the A100 via the QNN CPU backend (no board),
through the host head, vs the saved float reference. Fast calibration-method sweep.
Usage: qeval.py <dlc_file> [tag]
"""
import os, sys, glob, subprocess, tempfile, numpy as np, onnxruntime as ort

L, H, M = 128, 1024, 64
ENC_OUT = "/encoder/final_norm/LayerNormalization_output_0"
QROOT = os.path.expanduser("~/qairt/2.37.1.250807")
BIN = f"{QROOT}/bin/x86_64-linux-clang"
LIB = f"{QROOT}/lib/x86_64-linux-clang"
DLC = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(DLC)

def run_cpu(dlc, outdir):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = BIN + ":" + env.get("PATH", "")
    cmd = [f"{BIN}/qnn-net-run", "--backend", f"{LIB}/libQnnHtp.so",
           "--model", f"{LIB}/libQnnModelDlc.so", "--dlc_path", dlc,
           "--input_list", "calib128/input_list_enc3.txt", "--output_dir", outdir]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0 or not glob.glob(f"{outdir}/Result_0/*.raw"):
        raise RuntimeError("qnn-net-run(cpu) failed:\n" + (r.stderr or r.stdout)[-800:])

head = ort.InferenceSession("laya_head.onnx", providers=["CPUExecutionProvider"])
hin = [i.name for i in head.get_inputs()]
def marker_inputs(i):
    r = lambda n, dt, sh: np.fromfile(f"calib128/{n}_emb_{i:03d}.raw", np.int32).astype(dt).reshape(sh)
    return {"marker_pos": r("marker_pos", np.int64, (1, M)), "qtype": r("qtype", np.int64, (1,)),
            "marker_mask": r("marker_mask", bool, (1, M)), "attention_mask": r("attention_mask", np.int64, (1, L))}

ref = np.load("calib128/refs_logits_emb.npy"); mm = np.load("calib128/refs_mm_emb.npy")
N = ref.shape[0]

with tempfile.TemporaryDirectory() as td:
    run_cpu(DLC, td)
    A = []; B = []; agree = 0
    for i in range(N):
        hid = np.fromfile(glob.glob(f"{td}/Result_{i}/*.raw")[0], np.float32).reshape(1, L, H)
        d = marker_inputs(i); feed = {ENC_OUT: hid}
        for k in hin:
            if k != ENC_OUT and k in d: feed[k] = d[k]
        lg = head.run(["logits"], feed)[0]
        m = mm[i]
        A.append(ref[i][m]); B.append(lg[0][m])
        if ref[i][m].argmax() == lg[0][m].argmax(): agree += 1
    A = np.concatenate(A); B = np.concatenate(B)
    r = float(np.corrcoef(A, B)[0, 1]); rmse = float(np.sqrt(np.mean((A - B) ** 2)))
    print(f"{TAG:<22} r={r:.4f}  RMSE={rmse:.3f}  decisions={agree}/{N}", flush=True)
