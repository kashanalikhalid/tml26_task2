#!/usr/bin/env python3
"""TML 2026 Task 2 stolen-model detector.

Loads the target CIFAR ResNet-18 + each suspect, forwards a fixed probe set
(CIFAR-100 test), then computes behavioral + weight-space features per suspect
and writes:

* `outputs/features.csv` -- one row per suspect with every raw feature.
* `outputs/submission.csv` -- two-column `id,score` file ready for the
  TML 2026 leaderboard (rank-mean ensemble of the features in features.csv).
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

from detect.behavioral import behavioral_features, forward_logits
from detect.ensemble import ensemble_score
from detect.model import load_model, load_state_dict, make_model
from detect.probe import build_probe_set
from detect.weights import weight_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    parser.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    parser.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--features-out", type=Path, default=Path("outputs/features.csv"))
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (auto if unset).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-size", type=int, default=None, help="Limit probe images for smoke tests.")
    parser.add_argument("--probe-train", action="store_true", help="Use CIFAR-100 train instead of test as the probe split.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N suspects.")
    parser.add_argument("--no-weight-features", action="store_true")
    parser.add_argument("--no-behavioral-features", action="store_true")
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
        # Fall back to any safetensors in the directory (recursive).
        paths = sorted(suspects_root.rglob("*.safetensors"))
    return paths


def suspect_id(path: Path) -> int:
    stem = path.stem  # "suspect_037"
    try:
        return int(stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse suspect id from filename: {path.name}") from exc


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = pick_device(args.device)
    logging.info("Using device: %s", device)

    logging.info("Loading target from %s", args.target)
    target_sd = load_state_dict(args.target)
    target = make_model()
    target.load_state_dict(target_sd, strict=True)
    target.eval().to(device)

    target_logits: torch.Tensor | None = None
    if not args.no_behavioral_features:
        logging.info("Building probe set (cifar_root=%s, limit=%s, train=%s)",
                     args.cifar_root, args.probe_size, args.probe_train)
        probe_x, _probe_y = build_probe_set(args.cifar_root, limit=args.probe_size, train=args.probe_train)
        logging.info("Probe set: %s images", probe_x.size(0))
        t0 = time.time()
        target_logits = forward_logits(target, probe_x, args.batch_size, device)
        logging.info("Target forward in %.1fs", time.time() - t0)
    else:
        probe_x = None

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

            if not args.no_behavioral_features:
                assert probe_x is not None and target_logits is not None
                model = load_model(path, device=device, state_dict=sd)
                sus_logits = forward_logits(model, probe_x, args.batch_size, device)
                row.update(behavioral_features(target_logits, sus_logits))
                del model, sus_logits
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
    logging.info("Wrote %s (%d rows)", args.features_out, len(features))

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

    print("\nTop-10 suspects by ensemble score:")
    print(submission.sort_values("score", ascending=False).head(10).to_string(index=False))
    print("\nBottom-5 (clearly not stolen, sanity check):")
    print(submission.sort_values("score", ascending=False).tail(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
