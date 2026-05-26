#!/usr/bin/env python3
"""Soft LiRA: continuous per-suspect z-score of softmax probability on
target's wrong-prediction class, vs. the distribution induced by N shadow
models trained with target's recipe under different seeds.
"""
from __future__ import annotations

import argparse, glob, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.model import load_state_dict, make_model  # noqa: E402
from detect.probe import build_probe_set  # noqa: E402


OUT_DIR = Path("outputs/soft_lira")


@torch.no_grad()
def softmax_logits(model, x, batch_size, device):
    out = []
    for i in range(0, x.size(0), batch_size):
        chunk = x[i:i + batch_size].to(device, non_blocking=True)
        out.append(model(chunk).cpu())
    return torch.cat(out, dim=0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    p.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    p.add_argument("--shadow-stats-glob", default="outputs/shadow_stats/shadow_*.npz")
    p.add_argument("--features-out", type=Path, default=OUT_DIR / "features.csv")
    p.add_argument("--submission-out", type=Path, default=OUT_DIR / "submission.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-test-probe", type=int, default=1000)
    p.add_argument("--max-shadow-wrong-pct", type=float, default=0.15)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shadow_files = sorted(glob.glob(args.shadow_stats_glob))
    if not shadow_files:
        raise SystemExit(f"no shadow stats at {args.shadow_stats_glob}")
    shadows = [np.load(f) for f in shadow_files]
    logging.info("loaded %d shadow stats", len(shadows))

    target = make_model()
    target.load_state_dict(load_state_dict(args.target), strict=True)
    target.eval().to(device)

    test_x, test_y = build_probe_set(args.cifar_root, limit=args.n_test_probe, train=False)
    t_logits = softmax_logits(target, test_x, args.batch_size, device)
    t_top1 = t_logits.argmax(1).numpy()
    y_np = test_y.numpy()

    target_wrong = (t_top1 != y_np)
    shadow_test_top1 = np.stack([sh["test_top1"][:args.n_test_probe] for sh in shadows], axis=0)
    shadow_wrong_frac = (shadow_test_top1 != y_np[None, :]).mean(axis=0)
    vuln_mask = target_wrong & (shadow_wrong_frac <= args.max_shadow_wrong_pct)
    vuln_idx = np.where(vuln_mask)[0]
    logging.info("vulnerability samples: %d / %d", len(vuln_idx), args.n_test_probe)

    target_wrong_class = t_top1[vuln_idx]
    shadow_test_conf = np.stack(
        [sh["test_conf"][:args.n_test_probe][vuln_idx] for sh in shadows], axis=0
    )
    shadow_test_top1_v = shadow_test_top1[:, vuln_idx]
    # Shadow's prob on target's wrong class — we only have top1 + top1_conf, so
    # use shadow_conf when the shadow argmax matches target's wrong class,
    # otherwise distribute (1-conf) over the remaining 99 classes.
    match = (shadow_test_top1_v == target_wrong_class[None, :])
    shadow_prob = np.where(match, shadow_test_conf, (1.0 - shadow_test_conf) / 99.0)
    shadow_mean = shadow_prob.mean(axis=0)
    shadow_std = shadow_prob.std(axis=0) + 1e-6

    paths = sorted(args.suspects.glob("suspect_*.safetensors"))
    rows = []
    for path in tqdm(paths, desc="soft-lira"):
        row = {"id": int(path.stem.split("_")[-1]), "status": "ok"}
        try:
            m = make_model()
            m.load_state_dict(load_state_dict(path), strict=True)
            m.eval().to(device)
            s_probs = F.softmax(softmax_logits(m, test_x, args.batch_size, device), dim=1).numpy()
            suspect_prob = s_probs[vuln_idx, target_wrong_class]
            per_sample_z = (suspect_prob - shadow_mean) / shadow_std
            row["soft_lira_mean_z"] = float(per_sample_z.mean())
            row["soft_lira_median_z"] = float(np.median(per_sample_z))
            row["soft_lira_q75_z"] = float(np.quantile(per_sample_z, 0.75))
            row["soft_lira_mean_prob"] = float(suspect_prob.mean())
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            logging.exception("suspect %s failed", path)
            row["status"] = "error"
            row["error"] = str(e)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.features_out, index=False)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df

    sub = pd.DataFrame({
        "id": ok["id"].astype(int).to_numpy(),
        "score": ok["soft_lira_mean_z"].fillna(0).to_numpy(),
    }).sort_values("id").reset_index(drop=True)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.submission_out, index=False)
    print(sub.sort_values("score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
