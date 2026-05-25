#!/usr/bin/env python3
"""Experiment 06 — Penultimate-layer CKA representation similarity.

Hypothesis: distilled and heavily fine-tuned stolen models share the
target's hidden representation (avgpool output) even when their output
logits have drifted. Independents trained on the same task can have
similar representations at a coarse level, but the linear CKA over the
512-d avgpool features should be much higher for stolen models.

We use Linear CKA which is invariant to orthogonal transformations --
critical because stolen-but-permuted models look identical at the
representation level but differ in raw activations.

Outputs: outputs/exp06/{features.csv, submission.csv}.
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


EXPERIMENT = "exp06_cka"
OUT_DIR = Path("outputs/exp06")

FEATURE_SIGNS: dict[str, int] = {
    "behavioral_jsd": -1,
    "behavioral_top1_agree": +1,
    "behavioral_logit_pearson": +1,
    "behavioral_wrong_agree_test": +1,
    "behavioral_loss_corr_member": +1,
    "behavioral_loss_corr_nonmember": +1,
    "behavioral_member_gap_diff_abs": -1,
    "behavioral_member_loss_corr": +1,
    "behavioral_nonmember_loss_corr": +1,
    "weight_exact_tensor_frac": +1,
    "weight_l2_relative": -1,
    # CKA features
    "repr_cka_linear": +1,
    "repr_penult_cos_mean": +1,
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
    used = [c for c in FEATURE_SIGNS if c in features.columns]
    ranks = np.zeros(len(features), dtype=np.float64)
    for col in used:
        v = features[col].to_numpy(dtype=np.float64) * float(FEATURE_SIGNS[col])
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
    parser.add_argument("--cka-probe-size", type=int, default=2000,
                        help="Subset of test images for CKA (smaller = faster, 2000 is plenty for d=512).")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    device = pick_device(args.device)
    logging.info("[%s] device=%s cka_n=%d", EXPERIMENT, device, args.cka_probe_size)

    target_sd = load_state_dict(args.target)
    target = make_model(); target.load_state_dict(target_sd, strict=True); target.eval().to(device)

    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    (member_x, member_y), (nonmember_x, nonmember_y) = build_member_probes(
        args.cifar_root, args.train_main_idx,
        n_members=args.n_members, n_nonmembers=args.n_nonmembers,
    )
    logging.info("probes: test=%d member=%d nm=%d", test_x.size(0), member_x.size(0), nonmember_x.size(0))

    t0 = time.time()
    target_test_logits = forward_logits(target, test_x, args.batch_size, device)
    target_member_logits = forward_logits(target, member_x, args.batch_size, device)
    target_nm_logits = forward_logits(target, nonmember_x, args.batch_size, device)
    logging.info("target logits forwards in %.1fs", time.time() - t0)

    # Compute target penultimate features on a subset of test.
    cka_n = min(args.cka_probe_size, test_x.size(0))
    cka_x = test_x[:cka_n]
    t0 = time.time()
    target_penult = collect_penultimate(target, cka_x, args.batch_size, device)
    logging.info("target penultimate (n=%d, d=%d) in %.1fs", target_penult.size(0), target_penult.size(1), time.time() - t0)

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
