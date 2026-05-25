#!/usr/bin/env python3
"""Experiment 07 — CKA + PGD-strong combined.

Hypothesis: the two strongest new signals (penultimate-layer CKA from
exp06, stronger PGD adversarial transfer from exp05) target different
classes of stolen models. CKA discriminates distilled-with-OOD-transfer
models (representations preserved), PGD discriminates fine-tuned models
(boundary preserved). Combining them in one ensemble should cover both.

Uses a *narrow* rank-mean fusion — only the strongest discriminators
from each feature group, so the rank-mean doesn't get diluted by weak
correlated features.

Outputs: outputs/exp07/{features.csv, submission.csv}.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.adversarial import (  # noqa: E402
    _predict as predict_argmax,
    adversarial_transfer_features,
    craft_pgd_adversaries,
)
from detect.behavioral import (  # noqa: E402
    behavioral_features,
    forward_logits,
    label_aware_features,
    member_gap_features,
)
from detect.model import load_state_dict, make_model  # noqa: E402
from detect.probe import build_member_probes, build_probe_set  # noqa: E402
from detect.representations import (  # noqa: E402
    collect_penultimate,
    representation_features,
)
from detect.weights import weight_features  # noqa: E402


EXPERIMENT = "exp07_cka_pgd"
OUT_DIR = Path("outputs/exp07")

ADV_EPSILON = 16.0 / 255.0
ADV_ALPHA = 4.0 / 255.0
ADV_STEPS = 20
ADV_N = 2000

# Narrow feature set: only the top-1 most-discriminating feature per group.
NARROW_FEATURE_SIGNS: dict[str, int] = {
    # Output similarity (test set)
    "behavioral_jsd": -1,
    "behavioral_wrong_agree_test": +1,
    # Membership signal
    "behavioral_loss_corr_member": +1,
    "behavioral_member_gap_diff_abs": -1,
    # Weight-space
    "weight_exact_tensor_frac": +1,
    "weight_l2_relative": -1,
    # Representation similarity (CKA)
    "repr_cka_linear": +1,
    # Adversarial transfer
    "adv_transfer_class_match": +1,
}


def rank_norm(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = values.argsort()
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def ensemble_score(features: pd.DataFrame) -> np.ndarray:
    used = [c for c in NARROW_FEATURE_SIGNS if c in features.columns]
    ranks = np.zeros(len(features), dtype=np.float64)
    for col in used:
        v = features[col].to_numpy(dtype=np.float64) * float(NARROW_FEATURE_SIGNS[col])
        v = np.where(np.isnan(v), -np.inf, v)
        ranks += rank_norm(v)
    return ranks / len(used)


def pick_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    parser.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    parser.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    parser.add_argument("--features-out", type=Path, default=OUT_DIR / "features.csv")
    parser.add_argument("--submission-out", type=Path, default=OUT_DIR / "submission.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-members", type=int, default=4000)
    parser.add_argument("--n-nonmembers", type=int, default=4000)
    parser.add_argument("--cka-probe-size", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    device = pick_device(args.device)
    logging.info("[%s] device=%s adv_eps=%.4f adv_steps=%d cka_n=%d",
                 EXPERIMENT, device, ADV_EPSILON, ADV_STEPS, args.cka_probe_size)

    target_sd = load_state_dict(args.target)
    target = make_model(); target.load_state_dict(target_sd, strict=True); target.eval().to(device)

    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    (member_x, member_y), (nonmember_x, nonmember_y) = build_member_probes(
        args.cifar_root, args.train_main_idx,
        n_members=args.n_members, n_nonmembers=args.n_nonmembers,
    )

    t0 = time.time()
    target_test_logits = forward_logits(target, test_x, args.batch_size, device)
    target_member_logits = forward_logits(target, member_x, args.batch_size, device)
    target_nm_logits = forward_logits(target, nonmember_x, args.batch_size, device)
    logging.info("target logits in %.1fs", time.time() - t0)

    cka_n = min(args.cka_probe_size, test_x.size(0))
    cka_x = test_x[:cka_n]
    t0 = time.time()
    target_penult = collect_penultimate(target, cka_x, args.batch_size, device)
    logging.info("target penultimate (n=%d, d=%d) in %.1fs", target_penult.size(0), target_penult.size(1), time.time() - t0)

    adv_n = min(ADV_N, test_x.size(0))
    adv_x_clean = test_x[:adv_n]; adv_y = test_y[:adv_n]
    t0 = time.time()
    adv_x = craft_pgd_adversaries(target, adv_x_clean, adv_y,
                                  epsilon=ADV_EPSILON, alpha=ADV_ALPHA, steps=ADV_STEPS,
                                  batch_size=args.batch_size, device=device)
    target_adv_pred = predict_argmax(target, adv_x, args.batch_size, device)
    target_clean_pred = predict_argmax(target, adv_x_clean, args.batch_size, device)
    success = float((target_clean_pred != target_adv_pred).float().mean().item())
    logging.info("PGD (eps=%.4f, steps=%d) in %.1fs, attack-success=%.3f",
                 ADV_EPSILON, ADV_STEPS, time.time() - t0, success)

    suspect_paths = sorted(args.suspects.glob("suspect_*.safetensors"))
    if args.limit is not None:
        suspect_paths = suspect_paths[: args.limit]
    logging.info("scoring %d suspects", len(suspect_paths))

    rows: list[dict[str, object]] = []
    for path in tqdm(suspect_paths, desc=EXPERIMENT):
        row: dict[str, object] = {"id": int(path.stem.split("_")[-1]), "suspect_path": str(path)}
        try:
            sd = load_state_dict(path)
            m = make_model(); m.load_state_dict(sd, strict=True); m.eval().to(device)

            sus_test = forward_logits(m, test_x, args.batch_size, device)
            row.update(behavioral_features(target_test_logits, sus_test))
            row.update(label_aware_features(target_test_logits, sus_test, test_y, prefix="test"))

            sus_member = forward_logits(m, member_x, args.batch_size, device)
            sus_nm = forward_logits(m, nonmember_x, args.batch_size, device)
            row.update(label_aware_features(target_member_logits, sus_member, member_y, prefix="member"))
            row.update(label_aware_features(target_nm_logits, sus_nm, nonmember_y, prefix="nonmember"))
            row.update(member_gap_features(target_member_logits, target_nm_logits, sus_member, sus_nm, member_y, nonmember_y))

            sus_penult = collect_penultimate(m, cka_x, args.batch_size, device)
            row.update(representation_features(target_penult, sus_penult))

            sus_adv = predict_argmax(m, adv_x, args.batch_size, device)
            sus_clean_for_adv = predict_argmax(m, adv_x_clean, args.batch_size, device)
            row.update(adversarial_transfer_features(target_adv_pred, target_clean_pred, sus_adv, sus_clean_for_adv, adv_y))

            row.update(weight_features(target_sd, sd))
            row["status"] = "ok"
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            logging.exception("suspect %s failed", path)
            row["status"] = "error"; row["error"] = str(exc)
        rows.append(row)

    features = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_out, index=False)

    ok = features[features["status"] == "ok"] if "status" in features.columns else features
    submission = pd.DataFrame({"id": ok["id"].astype(int).to_numpy(), "score": ensemble_score(ok)}).sort_values("id").reset_index(drop=True)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_out, index=False)

    print("\n[%s] top-15:" % EXPERIMENT)
    print(submission.sort_values("score", ascending=False).head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
