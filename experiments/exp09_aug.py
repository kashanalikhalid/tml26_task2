#!/usr/bin/env python3
"""Experiment 09 — Augmentation-sensitive probe (target's biased crop).

The PDF tells us target was trained with biased random crop
(bias_x=0.5, bias_y=-0.25, jitter=0.25, pad 4 reflect). This is *target's
specific* training-time augmentation. Stolen / fine-tuned-from-target
models inherit sensitivity to crops from this region; independents trained
with standard center-or-uniform crops have a different sensitivity
profile.

We compute, for each suspect, the cosine and Pearson correlation of the
per-image delta (biased_crop_logits - center_crop_logits) with target's
delta. Stolen models have highly correlated deltas.
"""
from __future__ import annotations

import logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.augmentation import augmentation_features, make_biased_probe  # noqa: E402
from detect.behavioral import behavioral_features, forward_logits, label_aware_features, member_gap_features  # noqa: E402
from detect.model import load_state_dict, make_model  # noqa: E402
from detect.probe import build_member_probes, build_probe_set  # noqa: E402
from detect.weights import weight_features  # noqa: E402

OUT_DIR = Path("outputs/exp09")
SIGNS = {
    "behavioral_jsd": -1, "behavioral_wrong_agree_test": +1,
    "behavioral_loss_corr_member": +1, "behavioral_member_gap_diff_abs": -1,
    "weight_exact_tensor_frac": +1, "weight_l2_relative": -1,
    "aug_delta_cos_mean": +1, "aug_delta_pearson": +1, "aug_biased_top1_agree": +1, "aug_biased_jsd": -1,
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
    p.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    p.add_argument("--features-out", type=Path, default=OUT_DIR/"features.csv")
    p.add_argument("--submission-out", type=Path, default=OUT_DIR/"submission.csv")
    p.add_argument("--device", default=None); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-members", type=int, default=4000); p.add_argument("--n-nonmembers", type=int, default=4000)
    args=p.parse_args()
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("[exp09] device=%s (biased crop bias_x=0.5 bias_y=-0.25)", device)

    target_sd=load_state_dict(args.target); target=make_model(); target.load_state_dict(target_sd, strict=True); target.eval().to(device)
    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    (mx,my),(nx,ny) = build_member_probes(args.cifar_root, args.train_main_idx, n_members=args.n_members, n_nonmembers=args.n_nonmembers)
    # Apply target's biased crop to the same images for an aug-comparison signal
    test_biased = make_biased_probe(test_x, bias_x=0.5, bias_y=-0.25)

    t0=time.time()
    t_test=forward_logits(target,test_x,args.batch_size,device)
    t_biased=forward_logits(target,test_biased,args.batch_size,device)
    t_m=forward_logits(target,mx,args.batch_size,device); t_nm=forward_logits(target,nx,args.batch_size,device)
    logging.info("target forwards in %.1fs (incl biased)", time.time()-t0)

    paths=sorted(args.suspects.glob("suspect_*.safetensors"))
    rows=[]
    for path in tqdm(paths, desc="exp09"):
        row={"id": int(path.stem.split("_")[-1]), "status":"ok"}
        try:
            sd=load_state_dict(path); m=make_model(); m.load_state_dict(sd,strict=True); m.eval().to(device)
            s_test=forward_logits(m,test_x,args.batch_size,device)
            s_biased=forward_logits(m,test_biased,args.batch_size,device)
            row.update(behavioral_features(t_test, s_test))
            row.update(label_aware_features(t_test, s_test, test_y, prefix="test"))
            row.update(augmentation_features(t_test, t_biased, s_test, s_biased))
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
    print("[exp09] top-15:"); print(sub.sort_values("score",ascending=False).head(15).to_string(index=False))
    return 0

if __name__ == "__main__": sys.exit(main())
