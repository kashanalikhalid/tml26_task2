#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is missing
    def tqdm(iterable, **_: object):
        return iterable

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover - exercised only when safetensors is missing
    load_safetensors = None


SUPPORTED_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}
PREFERRED_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
    "checkpoint.pt",
    "weights.pt",
)
STATE_DICT_WRAPPERS = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "network",
    "module",
)
STRIP_PREFIXES = ("module.", "_orig_mod.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline stolen-model detector for TML 2026 Assignment 2."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/target"),
        help="Path to the target checkpoint file or target model directory.",
    )
    parser.add_argument(
        "--suspects",
        type=Path,
        default=Path("data/suspects"),
        help="Directory containing one file or one folder per suspect model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/suspect_scores.csv"),
        help="Where to write the ranked CSV.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=4096,
        help="Maximum number of tensor elements sampled per layer for similarity.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    return parser.parse_args()


def normalize_key(key: str) -> str:
    while True:
        for prefix in STRIP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        else:
            return key


def maybe_unwrap_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for wrapper_key in STATE_DICT_WRAPPERS:
            wrapped = obj.get(wrapper_key)
            if isinstance(wrapped, dict):
                candidate = filter_tensor_items(wrapped)
                if candidate:
                    return candidate

        candidate = filter_tensor_items(obj)
        if candidate:
            return candidate

    if hasattr(obj, "state_dict"):
        candidate = filter_tensor_items(obj.state_dict())
        if candidate:
            return candidate

    raise TypeError("Could not extract a tensor state_dict from the checkpoint.")


def filter_tensor_items(items: dict[str, object]) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    for raw_key, value in items.items():
        if isinstance(value, torch.Tensor):
            filtered[normalize_key(str(raw_key))] = value.detach().cpu()
    return filtered


def resolve_weight_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported checkpoint format: {path}")
        return path

    for filename in PREFERRED_FILENAMES:
        candidate = path / filename
        if candidate.exists():
            return candidate

    recursive_matches = sorted(
        item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not recursive_matches:
        raise FileNotFoundError(f"No checkpoint file found under: {path}")
    return recursive_matches[0]


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    resolved = resolve_weight_path(path)
    suffix = resolved.suffix.lower()

    if suffix == ".safetensors":
        if load_safetensors is None:
            raise ImportError(
                "Encountered a .safetensors file, but safetensors is not installed."
            )
        checkpoint = load_safetensors(str(resolved))
    else:
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)

    return maybe_unwrap_state_dict(checkpoint)


def list_suspects(root: Path) -> list[tuple[str, Path]]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Suspect directory not found: {root}")

    if root.is_file():
        return [(root.stem, root)]

    suspects: list[tuple[str, Path]] = []
    for item in sorted(root.iterdir()):
        if item.name.startswith("."):
            continue
        try:
            weight_path = resolve_weight_path(item)
        except (FileNotFoundError, ValueError):
            continue
        suspect_id = item.stem if item.is_file() else item.name
        suspects.append((suspect_id, weight_path))

    if suspects:
        return suspects

    fallback = sorted(
        item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return [(item.stem, item) for item in fallback]


def total_parameter_count(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(int(tensor.numel()) for tensor in state_dict.values())


def sample_tensor(tensor: torch.Tensor, max_elements: int) -> torch.Tensor:
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    flat = flat.to(torch.float32)
    if flat.numel() <= max_elements:
        return flat

    stride = max(1, flat.numel() // max_elements)
    return flat[::stride][:max_elements]


def weighted_average(values: Iterable[tuple[float, int]]) -> float:
    total_weight = 0
    total = 0.0
    for value, weight in values:
        total += value * weight
        total_weight += weight
    return total / total_weight if total_weight else 0.0


def compare_models(
    target_state: dict[str, torch.Tensor],
    suspect_state: dict[str, torch.Tensor],
    sample_size: int,
) -> dict[str, float]:
    target_keys = set(target_state)
    suspect_keys = set(suspect_state)
    shared_keys = sorted(target_keys & suspect_keys)
    same_shape_keys = [
        key
        for key in shared_keys
        if tuple(target_state[key].shape) == tuple(suspect_state[key].shape)
    ]

    target_param_total = total_parameter_count(target_state)
    shared_param_total = sum(target_state[key].numel() for key in same_shape_keys)
    exact_param_total = 0
    sample_agreement_hits = 0
    cosine_pairs: list[tuple[float, int]] = []
    relative_delta_pairs: list[tuple[float, int]] = []

    for key in same_shape_keys:
        target_tensor = target_state[key]
        suspect_tensor = suspect_state[key]
        weight = int(target_tensor.numel())

        if torch.equal(target_tensor, suspect_tensor):
            exact_param_total += weight

        target_sample = sample_tensor(target_tensor, sample_size)
        suspect_sample = sample_tensor(suspect_tensor, sample_size)
        if target_sample.numel() == 0 or suspect_sample.numel() == 0:
            continue

        if torch.allclose(target_sample, suspect_sample, rtol=0.0, atol=1e-6):
            sample_agreement_hits += 1

        cosine = float(
            F.cosine_similarity(target_sample.unsqueeze(0), suspect_sample.unsqueeze(0)).item()
        )
        cosine_pairs.append((cosine, weight))

        denom = max(float(target_sample.norm().item()), 1e-12)
        rel_delta = float((target_sample - suspect_sample).norm().item() / denom)
        relative_delta_pairs.append((rel_delta, weight))

    key_overlap_fraction = len(shared_keys) / max(len(target_keys), 1)
    shape_overlap_fraction = len(same_shape_keys) / max(len(target_keys), 1)
    shared_parameter_fraction = shared_param_total / max(target_param_total, 1)
    exact_parameter_fraction = exact_param_total / max(shared_param_total, 1)
    sampled_tensor_match_fraction = sample_agreement_hits / max(len(same_shape_keys), 1)
    cosine_similarity = weighted_average(cosine_pairs)
    relative_delta = weighted_average(relative_delta_pairs)

    score = (
        0.45 * exact_parameter_fraction
        + 0.25 * sampled_tensor_match_fraction
        + 0.20 * ((cosine_similarity + 1.0) / 2.0)
        + 0.10 * shared_parameter_fraction
    )
    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "key_overlap_fraction": key_overlap_fraction,
        "shape_overlap_fraction": shape_overlap_fraction,
        "shared_parameter_fraction": shared_parameter_fraction,
        "exact_parameter_fraction": exact_parameter_fraction,
        "sampled_tensor_match_fraction": sampled_tensor_match_fraction,
        "cosine_similarity": cosine_similarity,
        "relative_delta": relative_delta,
        "shared_key_count": float(len(shared_keys)),
        "same_shape_key_count": float(len(same_shape_keys)),
    }


def main() -> None:
    args = parse_args()

    print(f"Loading target checkpoint from: {args.target}")
    target_state = load_checkpoint(args.target)
    suspects = list_suspects(args.suspects)
    if args.limit is not None:
        suspects = suspects[: args.limit]

    if not suspects:
        raise SystemExit(f"No suspect checkpoints found under: {args.suspects}")

    print(f"Found {len(suspects)} suspect checkpoints.")

    results: list[dict[str, object]] = []
    for suspect_id, suspect_path in tqdm(suspects, desc="Scoring suspects"):
        row: dict[str, object] = {
            "suspect_id": suspect_id,
            "weight_file": str(suspect_path),
        }
        try:
            suspect_state = load_checkpoint(suspect_path)
            row.update(compare_models(target_state, suspect_state, args.sample_size))
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:  # pragma: no cover - defensive bookkeeping
            row.update(
                {
                    "score": 0.0,
                    "key_overlap_fraction": 0.0,
                    "shape_overlap_fraction": 0.0,
                    "shared_parameter_fraction": 0.0,
                    "exact_parameter_fraction": 0.0,
                    "sampled_tensor_match_fraction": 0.0,
                    "cosine_similarity": 0.0,
                    "relative_delta": 0.0,
                    "shared_key_count": 0.0,
                    "same_shape_key_count": 0.0,
                    "status": "error",
                    "error": str(exc),
                }
            )
        results.append(row)

    sorted_results = sorted(
        results,
        key=lambda row: (
            float(row["score"]),
            float(row["exact_parameter_fraction"]),
            float(row["cosine_similarity"]),
        ),
        reverse=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(sorted_results[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_results)

    print(f"Saved ranked scores to: {args.output}")
    print()
    print("Top suspects:")
    for row in sorted_results[:10]:
        print(
            f"{row['suspect_id']}: "
            f"score={float(row['score']):.4f}, "
            f"exact={float(row['exact_parameter_fraction']):.4f}, "
            f"cosine={float(row['cosine_similarity']):.4f}"
        )


if __name__ == "__main__":
    main()
