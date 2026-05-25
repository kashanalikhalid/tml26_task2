#!/usr/bin/env python3
"""Experiment 14 — LiRA-style scoring using shadow models.

Loads M shadow ResNet-18s (trained by exp13 on the same train subset
target was trained on, with different seeds) and computes, for each
suspect, a likelihood-ratio-style score:

  z_i = (sim_to_target(i) - mean(sim_to_shadows(i))) / std(sim_to_shadows(i))

A suspect that is *much* more similar to target than to typical shadows
gets a large positive z -- statistical evidence of stealing rather than
just same-data-different-seed.

We compute the score with several base similarity measures (per-sample
loss correlation, top-1 agreement on member set, JSD on test set, ...)
and merge them with our existing ensemble.
"""
from __future__ import annotations

import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.behavioral import (  # noqa: E402
    behavioral_features, forward_logits, label_aware_features, member_gap_features,
)
from detect.model import load_state_dict, make_model  # noqa: E402
from detect.probe import build_member_probes, build_probe_set  # noqa: E402
from detect.weights import weight_features  # noqa: E402


OUT_DIR = Path("outputs/exp14")
SIGNS = {
    "behavioral_jsd": -1, "behavioral_wrong_agree_test": +1,
    "behavioral_loss_corr_member": +1, "behavioral_member_gap_diff_abs": -1,
    "weight_exact_tensor_frac": +1, "weight_l2_relative": -1,
    # Shadow-based LiRA features (high = stolen)
    "lira_z_loss_corr_member": +1,
    "lira_z_jsd_test": +1,
    "lira_z_top1_member": +1,
    "lira_z_wrong_agree_test": +1,
}


def rn(v):
    n = len(v)
    o = v.argsort()
    r = np.empty(n)
    r[o] = np.arange(n)
    return r / max(n - 1, 1)


def ensemble(df):
    used = [c for c in SIGNS if c in df.columns]
    R = np.stack([rn(df[c].to_numpy() * SIGNS[c]) for c in used], axis=1)
    return R.mean(axis=1)


def pearson(a, b):
    ac = a - a.mean()
    bc = b - b.mean()
    return float((ac * bc).sum() / (ac.norm() * bc.norm() + 1e-12))


def pairwise_sim(target_log, target_member_log, target_y_member, suspect_log, suspect_member_log):
    """Return dict of similarity measurements between target and suspect (per-test-set probe + per-member probe)."""
    # JSD on test logits
    tp = F.softmax(target_log, dim=1)
    sp = F.softmax(suspect_log, dim=1)
    eps = 1e-30
    mid = 0.5 * (tp + sp)
    midlog = mid.clamp_min(eps).log()
    jsd = 0.5 * ((tp * ((tp + eps).log() - midlog)).sum(1)
                 + (sp * ((sp + eps).log() - midlog)).sum(1))
    jsd_mean = float(jsd.mean().item())
    # Per-sample loss correlation on member set
    tl = F.cross_entropy(target_member_log, target_y_member, reduction='none')
    sl = F.cross_entropy(suspect_member_log, target_y_member, reduction='none')
    loss_corr = pearson(tl, sl)
    # Top-1 agreement on member set (does suspect predict what target predicts?)
    top1 = float((target_member_log.argmax(1) == suspect_member_log.argmax(1)).float().mean().item())
    # Wrong-prediction agreement on test
    t_top1 = target_log.argmax(1)
    s_top1 = suspect_log.argmax(1)
    return {"jsd_test": jsd_mean, "loss_corr_member": loss_corr, "top1_member": top1,
            "wrong_agree_test": float((t_top1 == s_top1).float().mean().item())}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    p.add_argument("--shadows", type=Path, nargs="+", required=True, help="paths to shadow .safetensors")
    p.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    p.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    p.add_argument("--features-out", type=Path, default=OUT_DIR / "features.csv")
    p.add_argument("--submission-out", type=Path, default=OUT_DIR / "submission.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-members", type=int, default=4000)
    p.add_argument("--n-nonmembers", type=int, default=4000)
    args = p.parse_args()
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("[exp14] device=%s n_shadows=%d", device, len(args.shadows))

    target_sd = load_state_dict(args.target)
    target = make_model(); target.load_state_dict(target_sd, strict=True); target.eval().to(device)

    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    (member_x, member_y), (nonmember_x, nonmember_y) = build_member_probes(
        args.cifar_root, args.train_main_idx,
        n_members=args.n_members, n_nonmembers=args.n_nonmembers,
    )

    t_test = forward_logits(target, test_x, args.batch_size, device)
    t_member = forward_logits(target, member_x, args.batch_size, device)
    t_nm = forward_logits(target, nonmember_x, args.batch_size, device)

    # ----- Compute shadow→target reference statistics -----
    # For each shadow, run the same suite of similarity measurements vs target.
    # This gives us a distribution of "what does same-data-different-seed look like".
    shadow_sims = []  # list of dicts
    for sp_path in args.shadows:
        logging.info("loading shadow %s", sp_path)
        sd = load_state_dict(sp_path)
        sm = make_model(); sm.load_state_dict(sd, strict=True); sm.eval().to(device)
        s_test = forward_logits(sm, test_x, args.batch_size, device)
        s_member = forward_logits(sm, member_x, args.batch_size, device)
        sims = pairwise_sim(t_test, t_member, member_y, s_test, s_member)
        logging.info("shadow vs target sims: %s", {k: f"{v:.4f}" for k, v in sims.items()})
        shadow_sims.append(sims)
        del sm
    keys = list(shadow_sims[0].keys())
    shadow_means = {k: float(np.mean([s[k] for s in shadow_sims])) for k in keys}
    shadow_stds = {k: float(np.std([s[k] for s in shadow_sims]) + 1e-6) for k in keys}
    logging.info("shadow distribution means=%s", shadow_means)
    logging.info("shadow distribution stds =%s", shadow_stds)

    # ----- Score suspects -----
    paths = sorted(args.suspects.glob("suspect_*.safetensors"))
    rows = []
    for path in tqdm(paths, desc="exp14"):
        row = {"id": int(path.stem.split("_")[-1]), "status": "ok"}
        try:
            sd = load_state_dict(path)
            m = make_model(); m.load_state_dict(sd, strict=True); m.eval().to(device)
            s_test = forward_logits(m, test_x, args.batch_size, device)
            s_member = forward_logits(m, member_x, args.batch_size, device)
            s_nm = forward_logits(m, nonmember_x, args.batch_size, device)
            row.update(behavioral_features(t_test, s_test))
            row.update(label_aware_features(t_test, s_test, test_y, prefix="test"))
            row.update(label_aware_features(t_member, s_member, member_y, prefix="member"))
            row.update(member_gap_features(t_member, t_nm, s_member, s_nm, member_y, nonmember_y))
            sims = pairwise_sim(t_test, t_member, member_y, s_test, s_member)
            # LiRA-style z-scores
            for k, v in sims.items():
                # For "loss_corr_member", "top1_member", "wrong_agree_test": stolen models have HIGHER sim than shadows → high z is stolen
                # For "jsd_test": stolen models have LOWER jsd than shadows (closer match) → low z stolen → we use NEGATIVE z direction.
                z = (v - shadow_means[k]) / shadow_stds[k]
                if k == "jsd_test":
                    z = -z  # invert so higher = more stolen
                row[f"lira_z_{k}"] = float(z)
            row.update(weight_features(target_sd, sd))
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            logging.exception("suspect %s failed", path)
            row["status"] = "error"
            row["error"] = str(e)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.features_out, index=False)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    sub = pd.DataFrame({"id": ok["id"].astype(int).to_numpy(), "score": ensemble(ok)}).sort_values("id").reset_index(drop=True)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.submission_out, index=False)
    print("[exp14] top-15:")
    print(sub.sort_values("score", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
