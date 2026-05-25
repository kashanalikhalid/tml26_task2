#!/usr/bin/env python3
"""Experiment 15 — Weight permutation matching.

For each Conv2d / Linear layer in target, find the best output-channel
permutation of the suspect's weights that aligns it to target (Hungarian
assignment maximizing per-filter cosine similarity). Stolen models that
underwent neuron permutation (a common function-preserving transform)
have weights that look unrelated to target's at the byte level but
become near-identical after the right permutation.

Features added:
  perm_cos_layer_<name>: mean cosine of matched output channels per layer
  perm_cos_overall: weighted average across all layers (weight = num filters)
  perm_l2_overall: weighted L2 residual after matching
"""
from __future__ import annotations

import argparse, logging, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.behavioral import behavioral_features, forward_logits, label_aware_features, member_gap_features  # noqa: E402
from detect.model import load_state_dict, make_model  # noqa: E402
from detect.probe import build_member_probes, build_probe_set  # noqa: E402
from detect.weights import weight_features  # noqa: E402


OUT_DIR = Path("outputs/exp15")
SIGNS = {
    "behavioral_jsd": -1, "behavioral_wrong_agree_test": +1,
    "behavioral_loss_corr_member": +1, "behavioral_member_gap_diff_abs": -1,
    "weight_exact_tensor_frac": +1, "weight_l2_relative": -1,
    "perm_cos_overall": +1,
    "perm_cos_conv1": +1,
    "perm_cos_layer4_last": +1,
}


def rn(v):
    n = len(v); o = v.argsort(); r = np.empty(n); r[o] = np.arange(n)
    return r / max(n - 1, 1)


def ensemble(df):
    used = [c for c in SIGNS if c in df.columns]
    R = np.stack([rn(df[c].to_numpy() * SIGNS[c]) for c in used], axis=1)
    return R.mean(axis=1)


def perm_match_score(t_w: torch.Tensor, s_w: torch.Tensor) -> tuple[float, float]:
    """Hungarian-match output channels of s_w to t_w. Returns (mean_cos, mean_l2_rel)."""
    # Both (out_ch, ...) shaped
    t = t_w.detach().to(torch.float32).reshape(t_w.shape[0], -1)
    s = s_w.detach().to(torch.float32).reshape(s_w.shape[0], -1)
    if t.shape[0] != s.shape[0]:
        return 0.0, 1.0
    t_n = t / (t.norm(dim=1, keepdim=True) + 1e-12)
    s_n = s / (s.norm(dim=1, keepdim=True) + 1e-12)
    # Cosine matrix (out_ch x out_ch)
    cos_mat = (t_n @ s_n.T).cpu().numpy()
    # Optimal assignment maximizes total cosine
    row, col = linear_sum_assignment(-cos_mat)
    mean_cos = float(cos_mat[row, col].mean())
    # L2 after permutation
    matched = s[col]  # reorder suspect filters to match target
    diff = (t - matched).norm()
    base = t.norm() + 1e-12
    l2_rel = float(diff / base)
    return mean_cos, l2_rel


def permutation_features(target_sd: dict, suspect_sd: dict) -> dict[str, float]:
    """Walk all Conv2d/Linear weight tensors, do per-layer permutation matching."""
    feats: dict[str, float] = {}
    total_cos = 0.0
    total_l2 = 0.0
    total_w = 0
    conv1_cos = None
    layer4_last_cos = None
    last_layer_name = None
    layer_keys = sorted(target_sd.keys())
    for k in layer_keys:
        if not (k.endswith(".weight") and k in suspect_sd):
            continue
        t_w = target_sd[k]
        s_w = suspect_sd[k]
        if t_w.dim() < 2 or t_w.shape != s_w.shape:
            continue
        try:
            cos, l2 = perm_match_score(t_w, s_w)
        except Exception:
            continue
        n_filters = int(t_w.shape[0])
        total_cos += cos * n_filters
        total_l2 += l2 * n_filters
        total_w += n_filters
        if k == "conv1.weight":
            conv1_cos = cos
        if "layer4" in k:
            layer4_last_cos = cos
            last_layer_name = k
    if total_w > 0:
        feats["perm_cos_overall"] = total_cos / total_w
        feats["perm_l2_overall"] = total_l2 / total_w
    else:
        feats["perm_cos_overall"] = 0.0
        feats["perm_l2_overall"] = 1.0
    if conv1_cos is not None:
        feats["perm_cos_conv1"] = conv1_cos
    if layer4_last_cos is not None:
        feats["perm_cos_layer4_last"] = layer4_last_cos
    return feats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    p.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    p.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    p.add_argument("--features-out", type=Path, default=OUT_DIR / "features.csv")
    p.add_argument("--submission-out", type=Path, default=OUT_DIR / "submission.csv")
    p.add_argument("--device", default="cpu")  # CPU is fine; permutation matching is small numpy
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-members", type=int, default=4000)
    p.add_argument("--n-nonmembers", type=int, default=4000)
    p.add_argument("--skip-behavioral", action="store_true",
                   help="Skip behavioral forward passes (just emit weight-perm features).")
    args = p.parse_args()
    device = torch.device(args.device)
    logging.info("[exp15] device=%s skip_behavioral=%s", device, args.skip_behavioral)

    target_sd = load_state_dict(args.target)
    target = make_model(); target.load_state_dict(target_sd, strict=True); target.eval().to(device)

    if not args.skip_behavioral:
        test_x, test_y = build_probe_set(args.cifar_root, train=False)
        (mem_x, mem_y), (nm_x, nm_y) = build_member_probes(
            args.cifar_root, args.train_main_idx,
            n_members=args.n_members, n_nonmembers=args.n_nonmembers,
        )
        t_test = forward_logits(target, test_x, args.batch_size, device)
        t_mem = forward_logits(target, mem_x, args.batch_size, device)
        t_nm = forward_logits(target, nm_x, args.batch_size, device)

    paths = sorted(args.suspects.glob("suspect_*.safetensors"))
    rows = []
    for path in tqdm(paths, desc="exp15"):
        row = {"id": int(path.stem.split("_")[-1]), "status": "ok"}
        try:
            sd = load_state_dict(path)
            row.update(permutation_features(target_sd, sd))
            row.update(weight_features(target_sd, sd))
            if not args.skip_behavioral:
                m = make_model(); m.load_state_dict(sd, strict=True); m.eval().to(device)
                s_test = forward_logits(m, test_x, args.batch_size, device)
                s_mem = forward_logits(m, mem_x, args.batch_size, device)
                s_nm = forward_logits(m, nm_x, args.batch_size, device)
                row.update(behavioral_features(t_test, s_test))
                row.update(label_aware_features(t_test, s_test, test_y, prefix="test"))
                row.update(label_aware_features(t_mem, s_mem, mem_y, prefix="member"))
                row.update(member_gap_features(t_mem, t_nm, s_mem, s_nm, mem_y, nm_y))
                del m
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
    print("[exp15] top-15:")
    print(sub.sort_values("score", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
