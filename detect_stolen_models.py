#!/usr/bin/env python3
"""TML 2026 Task 2 stolen-model detector.

Loads the target CIFAR ResNet-18 + each suspect, forwards three probe sets
(CIFAR-100 test, target's training members, target's training non-members),
then computes behavioral + weight-space features per suspect and writes:

* `outputs/features.csv` -- one row per suspect with every raw feature.
* `outputs/submission.csv` -- two-column `id,score` ready for the
  TML 2026 leaderboard (rank-mean ensemble of the features).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from detect.behavioral import (
    behavioral_features,
    forward_logits,
    label_aware_features,
    member_gap_features,
)
from detect.ensemble import ensemble_score
from detect.model import load_state_dict, make_model
from detect.probe import build_member_probes, build_probe_set
from detect.weights import weight_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    parser.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    parser.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    parser.add_argument("--features-out", type=Path, default=Path("outputs/features.csv"))
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (auto if unset).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-size", type=int, default=None, help="Limit test probe images for smoke tests.")
    parser.add_argument("--n-members", type=int, default=4000, help="Sample size for the member probe set.")
    parser.add_argument("--n-nonmembers", type=int, default=4000, help="Sample size for the non-member probe set.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N suspects.")
    parser.add_argument("--no-weight-features", action="store_true")
    parser.add_argument("--no-behavioral-features", action="store_true")
    parser.add_argument("--no-member-features", action="store_true",
                        help="Skip features that need train_main_idx.json (member/non-member probes).")
    return parser.parse_args()


def pick_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_suspects(suspects_root: Path) -> list[Path]:
    paths = sorted(suspects_root.glob("suspect_*.safetensors"))
    if not paths:
        paths = sorted(suspects_root.rglob("*.safetensors"))
    return paths


def suspect_id(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse suspect id from filename: {path.name}") from exc


def build_model_on(state_dict: dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    model = make_model()
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = pick_device(args.device)
    logging.info("Using device: %s", device)

    logging.info("Loading target from %s", args.target)
    target_sd = load_state_dict(args.target)
    target = build_model_on(target_sd, device)

    # ----- probe sets -----
    use_behavioral = not args.no_behavioral_features
    use_member = not args.no_member_features and args.train_main_idx.exists()

    test_x = test_y = None
    member_x = member_y = nonmember_x = nonmember_y = None
    target_test_logits = target_member_logits = target_nm_logits = None

    if use_behavioral:
        logging.info("Building test probe set (cifar_root=%s, limit=%s)", args.cifar_root, args.probe_size)
        test_x, test_y = build_probe_set(args.cifar_root, limit=args.probe_size, train=False)
        logging.info("Test probe: %d images", test_x.size(0))
        t0 = time.time()
        target_test_logits = forward_logits(target, test_x, args.batch_size, device)
        logging.info("Target test-forward in %.1fs", time.time() - t0)

    if use_member:
        logging.info("Building member/non-member probes from %s (n_m=%d n_nm=%d)",
                     args.train_main_idx, args.n_members, args.n_nonmembers)
        (member_x, member_y), (nonmember_x, nonmember_y) = build_member_probes(
            args.cifar_root,
            args.train_main_idx,
            n_members=args.n_members,
            n_nonmembers=args.n_nonmembers,
        )
        logging.info("Member probe: %d images   Non-member probe: %d images",
                     member_x.size(0), nonmember_x.size(0))
        t0 = time.time()
        target_member_logits = forward_logits(target, member_x, args.batch_size, device)
        target_nm_logits = forward_logits(target, nonmember_x, args.batch_size, device)
        logging.info("Target member+nm forward in %.1fs", time.time() - t0)

    suspect_paths = discover_suspects(args.suspects)
    if not suspect_paths:
        logging.error("No suspect .safetensors files found under %s", args.suspects)
        return 2
    if args.limit is not None:
        suspect_paths = suspect_paths[: args.limit]
    logging.info("Scoring %d suspects", len(suspect_paths))

    rows: list[dict[str, object]] = []
    for path in tqdm(suspect_paths, desc="suspects"):
        row: dict[str, object] = {"id": suspect_id(path), "suspect_path": str(path)}
        try:
            sd = load_state_dict(path)

            if use_behavioral or use_member:
                model = build_model_on(sd, device)
                if use_behavioral:
                    sus_test = forward_logits(model, test_x, args.batch_size, device)
                    row.update(behavioral_features(target_test_logits, sus_test))
                    row.update(label_aware_features(target_test_logits, sus_test, test_y, prefix="test"))
                if use_member:
                    sus_member = forward_logits(model, member_x, args.batch_size, device)
                    sus_nm = forward_logits(model, nonmember_x, args.batch_size, device)
                    row.update(label_aware_features(target_member_logits, sus_member, member_y, prefix="member"))
                    row.update(label_aware_features(target_nm_logits, sus_nm, nonmember_y, prefix="nonmember"))
                    row.update(member_gap_features(
                        target_member_logits, target_nm_logits,
                        sus_member, sus_nm,
                        member_y, nonmember_y,
                    ))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if not args.no_weight_features:
                row.update(weight_features(target_sd, sd))

            row["status"] = "ok"
        except Exception as exc:  # pragma: no cover - defensive bookkeeping
            logging.exception("Failed scoring suspect %s", path)
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)

    features = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_out, index=False)
    logging.info("Wrote %s (%d rows, %d cols)", args.features_out, len(features), features.shape[1])

    scoring_features = features[features["status"] == "ok"] if "status" in features.columns else features
    if scoring_features.empty:
        logging.error("No suspects scored successfully; cannot write submission.")
        return 3
    submission = pd.DataFrame({
        "id": scoring_features["id"].astype(int).to_numpy(),
        "score": ensemble_score(scoring_features),
    }).sort_values("id").reset_index(drop=True)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_out, index=False)
    logging.info("Wrote %s (%d rows)", args.submission_out, len(submission))

    print("\nTop-15 suspects by ensemble score:")
    print(submission.sort_values("score", ascending=False).head(15).to_string(index=False))
    print("\nBottom-5:")
    print(submission.sort_values("score", ascending=False).tail(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
