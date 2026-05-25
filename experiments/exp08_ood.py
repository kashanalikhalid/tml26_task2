#!/usr/bin/env python3
"""Experiment 08 — Out-of-distribution probes (Gaussian noise + CIFAR-10).

Hypothesis: target has model-specific responses to OOD inputs that stolen
models inherit. Independents trained on the same in-distribution data have
different decision-boundary behaviour outside the training manifold.
"""
from __future__ import annotations

import logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.behavioral import behavioral_features, forward_logits, label_aware_features, member_gap_features  # noqa: E402
from detect.model import load_state_dict, make_model  # noqa: E402
from detect.ood import build_gaussian_noise_probe, build_cifar10_probe, ood_features  # noqa: E402
from detect.probe import build_member_probes, build_probe_set  # noqa: E402
from detect.weights import weight_features  # noqa: E402

OUT_DIR = Path("outputs/exp08")
SIGNS = {
    "behavioral_jsd": -1, "behavioral_wrong_agree_test": +1,
    "behavioral_loss_corr_member": +1, "behavioral_loss_corr_nonmember": +1,
    "behavioral_member_gap_diff_abs": -1,
    "weight_exact_tensor_frac": +1, "weight_l2_relative": -1,
    "ood_noise_top1_agree": +1, "ood_noise_logit_pearson": +1, "ood_noise_jsd": -1,
    "ood_cifar10_top1_agree": +1, "ood_cifar10_logit_pearson": +1, "ood_cifar10_jsd": -1,
}

def rn(v):
    n=len(v); o=v.argsort(); r=np.empty(n); r[o]=np.arange(n); return r/max(n-1,1)

def ensemble(df):
    used=[c for c in SIGNS if c in df.columns]
    R=np.stack([rn(df[c].to_numpy()*SIGNS[c]) for c in used], axis=1)
    return R.mean(axis=1)

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p=argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    p.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    p.add_argument("--cifar10-root", type=Path, default=Path("data/cifar10"))
    p.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    p.add_argument("--features-out", type=Path, default=OUT_DIR/"features.csv")
    p.add_argument("--submission-out", type=Path, default=OUT_DIR/"submission.csv")
    p.add_argument("--device", default=None); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-members", type=int, default=4000); p.add_argument("--n-nonmembers", type=int, default=4000)
    p.add_argument("--n-noise", type=int, default=2000); p.add_argument("--n-cifar10", type=int, default=2000)
    args=p.parse_args()
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("[exp08] device=%s n_noise=%d n_cifar10=%d", device, args.n_noise, args.n_cifar10)

    target_sd=load_state_dict(args.target); target=make_model(); target.load_state_dict(target_sd, strict=True); target.eval().to(device)
    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    (mx,my),(nx,ny) = build_member_probes(args.cifar_root, args.train_main_idx, n_members=args.n_members, n_nonmembers=args.n_nonmembers)
    noise_x = build_gaussian_noise_probe(n=args.n_noise, seed=42)
    cifar10_x = build_cifar10_probe(args.cifar10_root, limit=args.n_cifar10)

    t0=time.time()
    t_test=forward_logits(target,test_x,args.batch_size,device)
    t_m=forward_logits(target,mx,args.batch_size,device)
    t_nm=forward_logits(target,nx,args.batch_size,device)
    t_noise=forward_logits(target,noise_x,args.batch_size,device)
    t_c10=forward_logits(target,cifar10_x,args.batch_size,device)
    logging.info("target forwards in %.1fs", time.time()-t0)

    paths=sorted(args.suspects.glob("suspect_*.safetensors"))
    rows=[]
    for path in tqdm(paths, desc="exp08"):
        row={"id": int(path.stem.split("_")[-1]), "status":"ok"}
        try:
            sd=load_state_dict(path); m=make_model(); m.load_state_dict(sd,strict=True); m.eval().to(device)
            s_test=forward_logits(m,test_x,args.batch_size,device)
            row.update(behavioral_features(t_test, s_test))
            row.update(label_aware_features(t_test, s_test, test_y, prefix="test"))
            s_m=forward_logits(m,mx,args.batch_size,device); s_nm=forward_logits(m,nx,args.batch_size,device)
            row.update(label_aware_features(t_m, s_m, my, prefix="member"))
            row.update(label_aware_features(t_nm, s_nm, ny, prefix="nonmember"))
            row.update(member_gap_features(t_m,t_nm,s_m,s_nm,my,ny))
            row.update(ood_features(t_noise, forward_logits(m,noise_x,args.batch_size,device), prefix="noise"))
            row.update(ood_features(t_c10,  forward_logits(m,cifar10_x,args.batch_size,device), prefix="cifar10"))
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
    print("[exp08] top-15:"); print(sub.sort_values("score",ascending=False).head(15).to_string(index=False))
    return 0

if __name__ == "__main__": sys.exit(main())
