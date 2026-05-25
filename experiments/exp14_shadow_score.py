#!/usr/bin/env python3
"""Experiment 14 — LiRA-style scoring using *probe statistics* from N shadows.

Each shadow (trained by exp13) emits a small .npz with per-sample top-1
prediction, top-1 confidence, and CE loss on three probe sets. We use
this distribution to compute, for each suspect, z-score-like features:

  - shadow_loss_corr_member_z:  Pearson(target_per_sample_loss, suspect_per_sample_loss) on member set,
                                z-scored against the same Pearson for the
                                target-vs-shadow pairs. Higher = more stolen-like.
  - shadow_top1_member_z:       Fraction of member samples on which suspect predicts target's exact top-1,
                                z-scored against the shadow-vs-target distribution.
  - shadow_target_specific_pred_match_z:  On samples where target's top-1 is rare among shadows (target-specific),
                                how often does suspect match target? Z-scored.
  - shadow_loss_pattern_l2_z:   L2 distance between target's and suspect's per-sample loss vectors on member set,
                                z-scored against shadow distribution. Lower = more stolen-like (we use negative sign).
"""
from __future__ import annotations

import argparse, glob, logging, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.behavioral import behavioral_features, forward_logits, label_aware_features, member_gap_features  # noqa: E402
from detect.model import load_state_dict, make_model  # noqa: E402
from detect.probe import build_member_probes, build_probe_set  # noqa: E402
from detect.weights import weight_features  # noqa: E402


OUT_DIR = Path("outputs/exp14")
SIGNS = {
    "behavioral_jsd": -1, "behavioral_wrong_agree_test": +1,
    "behavioral_loss_corr_member": +1, "behavioral_member_gap_diff_abs": -1,
    "weight_exact_tensor_frac": +1, "weight_l2_relative": -1,
    "shadow_loss_corr_member_z": +1,
    "shadow_top1_member_z": +1,
    "shadow_target_specific_pred_match_z": +1,
    "shadow_loss_pattern_l2_z": -1,
}


def rn(v):
    n = len(v); o = v.argsort(); r = np.empty(n); r[o] = np.arange(n)
    return r / max(n - 1, 1)


def ensemble(df):
    used = [c for c in SIGNS if c in df.columns]
    R = np.stack([rn(df[c].to_numpy() * SIGNS[c]) for c in used], axis=1)
    return R.mean(axis=1)


def pearson(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ac = a - a.mean(); bc = b - b.mean()
    return float((ac * bc).sum() / (np.linalg.norm(ac) * np.linalg.norm(bc) + 1e-12))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    p.add_argument("--shadow-stats-glob", default="outputs/shadow_stats/shadow_*.npz",
                   help="Glob pattern for the .npz files produced by exp13.")
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
    logging.info("[exp14] device=%s", device)

    # ----- Load shadows' probe stats -----
    shadow_files = sorted(glob.glob(args.shadow_stats_glob))
    if not shadow_files:
        raise SystemExit(f"No shadow stats found at {args.shadow_stats_glob}")
    shadows = [np.load(f) for f in shadow_files]
    logging.info("loaded %d shadow stats", len(shadows))

    # ----- Target probe forwards -----
    target_sd = load_state_dict(args.target)
    target = make_model(); target.load_state_dict(target_sd, strict=True); target.eval().to(device)
    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    (mem_x, mem_y), (nm_x, nm_y) = build_member_probes(
        args.cifar_root, args.train_main_idx,
        n_members=args.n_members, n_nonmembers=args.n_nonmembers,
    )

    t_test = forward_logits(target, test_x, args.batch_size, device)
    t_mem = forward_logits(target, mem_x, args.batch_size, device)
    t_nm = forward_logits(target, nm_x, args.batch_size, device)
    t_mem_loss = F.cross_entropy(t_mem, mem_y, reduction='none').cpu().numpy()
    t_mem_top1 = t_mem.argmax(1).cpu().numpy()

    # Shadows' loss correlations / top-1 agreements with target (reference distribution)
    shadow_loss_corrs = []
    shadow_top1s = []
    shadow_l2s = []
    shadow_top1_mem_preds_list = []
    for sh in shadows:
        shadow_loss_corrs.append(pearson(t_mem_loss, sh["mem_loss"]))
        shadow_top1s.append(float(np.mean(sh["mem_top1"] == t_mem_top1)))
        shadow_l2s.append(float(np.linalg.norm(t_mem_loss - sh["mem_loss"])))
        shadow_top1_mem_preds_list.append(sh["mem_top1"])
    sh_loss_corr_mean, sh_loss_corr_std = float(np.mean(shadow_loss_corrs)), float(np.std(shadow_loss_corrs) + 1e-6)
    sh_top1_mean, sh_top1_std = float(np.mean(shadow_top1s)), float(np.std(shadow_top1s) + 1e-6)
    sh_l2_mean, sh_l2_std = float(np.mean(shadow_l2s)), float(np.std(shadow_l2s) + 1e-6)
    logging.info("shadow ref distributions:")
    logging.info("  loss_corr (vs target): mean=%.4f std=%.4f", sh_loss_corr_mean, sh_loss_corr_std)
    logging.info("  top1_member  (vs target): mean=%.4f std=%.4f", sh_top1_mean, sh_top1_std)
    logging.info("  l2_loss_pattern (vs target): mean=%.4f std=%.4f", sh_l2_mean, sh_l2_std)

    # Per-sample target-specific predictions
    # For each test sample, what fraction of shadows predict the SAME class as target?
    # If <= 0.5 (target's prediction is rare among shadows), that sample is "target-specific".
    # Stolen models tend to share target's rare predictions.
    shadow_test_top1 = np.stack([sh["test_top1"] for sh in shadows], axis=0)  # (N_shadows, N_test)
    t_test_top1 = t_test.argmax(1).cpu().numpy()
    shadow_match_target = (shadow_test_top1 == t_test_top1[None, :]).mean(axis=0)  # (N_test,)
    rare_mask = shadow_match_target <= 0.5  # boolean, target-specific predictions
    logging.info("target-specific predictions on test: %d / %d (%.1f%%)",
                 int(rare_mask.sum()), len(rare_mask), 100.0 * rare_mask.mean())

    # ----- Score suspects -----
    paths = sorted(args.suspects.glob("suspect_*.safetensors"))
    rows = []
    for path in tqdm(paths, desc="exp14"):
        row = {"id": int(path.stem.split("_")[-1]), "status": "ok"}
        try:
            sd = load_state_dict(path)
            m = make_model(); m.load_state_dict(sd, strict=True); m.eval().to(device)
            s_test = forward_logits(m, test_x, args.batch_size, device)
            s_mem = forward_logits(m, mem_x, args.batch_size, device)
            s_nm = forward_logits(m, nm_x, args.batch_size, device)
            row.update(behavioral_features(t_test, s_test))
            row.update(label_aware_features(t_test, s_test, test_y, prefix="test"))
            row.update(label_aware_features(t_mem, s_mem, mem_y, prefix="member"))
            row.update(member_gap_features(t_mem, t_nm, s_mem, s_nm, mem_y, nm_y))
            row.update(weight_features(target_sd, sd))

            s_mem_loss = F.cross_entropy(s_mem, mem_y, reduction='none').cpu().numpy()
            s_mem_top1 = s_mem.argmax(1).cpu().numpy()
            s_test_top1 = s_test.argmax(1).cpu().numpy()

            # LiRA z-scores
            sus_loss_corr = pearson(t_mem_loss, s_mem_loss)
            row["shadow_loss_corr_member_z"] = (sus_loss_corr - sh_loss_corr_mean) / sh_loss_corr_std
            sus_top1 = float(np.mean(s_mem_top1 == t_mem_top1))
            row["shadow_top1_member_z"] = (sus_top1 - sh_top1_mean) / sh_top1_std
            sus_l2 = float(np.linalg.norm(t_mem_loss - s_mem_loss))
            row["shadow_loss_pattern_l2_z"] = (sus_l2 - sh_l2_mean) / sh_l2_std

            # Target-specific prediction match rate
            # Among samples where target's prediction is rare in shadow distribution,
            # what fraction does suspect ALSO match target's prediction?
            if rare_mask.any():
                sus_specific_match = float(np.mean(s_test_top1[rare_mask] == t_test_top1[rare_mask]))
                # Shadow reference for the same statistic
                shadow_specific_matches = [
                    float(np.mean(sh["test_top1"][rare_mask] == t_test_top1[rare_mask])) for sh in shadows
                ]
                sm_mean, sm_std = float(np.mean(shadow_specific_matches)), float(np.std(shadow_specific_matches) + 1e-6)
                row["shadow_target_specific_pred_match_z"] = (sus_specific_match - sm_mean) / sm_std
                row["shadow_target_specific_pred_match_raw"] = sus_specific_match
            else:
                row["shadow_target_specific_pred_match_z"] = 0.0

            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            logging.exception("suspect %s failed", path)
            row["status"] = "error"; row["error"] = str(e)
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
