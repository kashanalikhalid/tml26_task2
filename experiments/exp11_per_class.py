#!/usr/bin/env python3
"""Experiment 11 — Per-class loss correlation + per-class top-1 agreement.

For each of the 100 classes, compute target's mean CE loss on samples
of that class, plus target's mean confidence on that class. Stolen
models share the target's "I'm bad at class 42 / great at class 17"
profile. Independents have different per-class weakness patterns.

Adds two feature vectors per suspect:
  per_class_loss_corr — Pearson correlation of 100-vector of mean losses.
  per_class_top1_corr — Pearson correlation of 100-vector of top-1 accuracy.
"""
from __future__ import annotations

import logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.behavioral import behavioral_features, forward_logits, label_aware_features, member_gap_features  # noqa: E402
from detect.model import load_state_dict, make_model  # noqa: E402
from detect.probe import build_member_probes, build_probe_set  # noqa: E402
from detect.weights import weight_features  # noqa: E402

OUT_DIR = Path("outputs/exp11")
SIGNS = {
    "behavioral_jsd": -1, "behavioral_wrong_agree_test": +1,
    "behavioral_loss_corr_member": +1, "behavioral_member_gap_diff_abs": -1,
    "weight_exact_tensor_frac": +1, "weight_l2_relative": -1,
    "per_class_loss_corr": +1, "per_class_top1_corr": +1,
}

def rn(v):
    n=len(v); o=v.argsort(); r=np.empty(n); r[o]=np.arange(n); return r/max(n-1,1)

def ensemble(df):
    used=[c for c in SIGNS if c in df.columns]
    R=np.stack([rn(df[c].to_numpy()*SIGNS[c]) for c in used], axis=1)
    return R.mean(axis=1)

def per_class_vectors(logits, y, num_classes=100):
    """Returns (loss_vec, acc_vec) each of shape (num_classes,)."""
    loss = F.cross_entropy(logits, y, reduction="none")
    top1 = (logits.argmax(1) == y).float()
    loss_vec = torch.zeros(num_classes)
    acc_vec = torch.zeros(num_classes)
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = (y == c)
        if mask.any():
            loss_vec[c] = loss[mask].mean()
            acc_vec[c] = top1[mask].mean()
            counts[c] = 1.0
    return loss_vec, acc_vec, counts.bool()

def pearson(a, b, mask=None):
    if mask is not None:
        a = a[mask]; b = b[mask]
    if a.numel() == 0: return 0.0
    ac = a - a.mean(); bc = b - b.mean()
    return float((ac * bc).sum() / (ac.norm() * bc.norm() + 1e-12))

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p=argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    p.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    p.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    p.add_argument("--features-out", type=Path, default=OUT_DIR/"features.csv")
    p.add_argument("--submission-out", type=Path, default=OUT_DIR/"submission.csv")
    p.add_argument("--device", default=None); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-members", type=int, default=4000); p.add_argument("--n-nonmembers", type=int, default=4000)
    args=p.parse_args()
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("[exp11] device=%s", device)

    target_sd=load_state_dict(args.target); target=make_model(); target.load_state_dict(target_sd,strict=True); target.eval().to(device)
    test_x,test_y=build_probe_set(args.cifar_root, train=False)
    (mx,my),(nx,ny)=build_member_probes(args.cifar_root, args.train_main_idx, n_members=args.n_members, n_nonmembers=args.n_nonmembers)
    t_test=forward_logits(target,test_x,args.batch_size,device)
    t_m=forward_logits(target,mx,args.batch_size,device); t_nm=forward_logits(target,nx,args.batch_size,device)
    t_loss_v, t_acc_v, mask = per_class_vectors(t_test, test_y)

    paths=sorted(args.suspects.glob("suspect_*.safetensors"))
    rows=[]
    for path in tqdm(paths, desc="exp11"):
        row={"id": int(path.stem.split("_")[-1]), "status":"ok"}
        try:
            sd=load_state_dict(path); m=make_model(); m.load_state_dict(sd,strict=True); m.eval().to(device)
            s_test=forward_logits(m,test_x,args.batch_size,device)
            row.update(behavioral_features(t_test, s_test))
            row.update(label_aware_features(t_test, s_test, test_y, prefix="test"))
            s_loss_v, s_acc_v, _ = per_class_vectors(s_test, test_y)
            row["per_class_loss_corr"] = pearson(t_loss_v, s_loss_v, mask)
            row["per_class_top1_corr"] = pearson(t_acc_v, s_acc_v, mask)
            s_m=forward_logits(m,mx,args.batch_size,device); s_nm=forward_logits(m,nx,args.batch_size,device)
            row.update(label_aware_features(t_m, s_m, my, prefix="member"))
            row.update(label_aware_features(t_nm, s_nm, ny, prefix="nonmember"))
            row.update(member_gap_features(t_m,t_nm,s_m,s_nm,my,ny))
            row.update(weight_features(target_sd, sd))
            del m
            if device.type=="cuda": torch.cuda.empty_cache()
        except Exception as e:
            row["status"]="error"; row["error"]=str(e); logging.exception("fail %s", path)
        rows.append(row)

    df=pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    args.features_out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(args.features_out, index=False)
    ok=df[df["status"]=="ok"] if "status" in df.columns else df
    sub=pd.DataFrame({"id": ok["id"].astype(int).to_numpy(), "score": ensemble(ok)}).sort_values("id").reset_index(drop=True)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True); sub.to_csv(args.submission_out, index=False)
    print("[exp11] top-15:"); print(sub.sort_values("score",ascending=False).head(15).to_string(index=False))
    return 0

if __name__ == "__main__": sys.exit(main())
