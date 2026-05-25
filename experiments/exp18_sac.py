#!/usr/bin/env python3
"""Experiment 18 — SAC (Sample Correlation) fingerprinting.

Paper: "Are You Stealing My Model? Sample Correlation for Fingerprinting
Deep Neural Networks" (Guan et al., NeurIPS 2022, arXiv:2210.15427).

The trick: every output-similarity feature we've tried so far compares
target and suspect *per-input independently* (target says X on input i,
does suspect also say X?). SAC compares the *pairwise relationship*
between inputs in the output space.

For N probe inputs, compute the NxN correlation matrix C[i,j] = cos(o_i, o_j)
of model outputs. Two functionally-related models have similar pairwise
relationships; two independents trained on the same data have DIFFERENT
pairwise structure even when their per-input outputs match.

Score per suspect:  ||C_target - C_suspect||_1 / N^2
(lower distance = more stolen-like; we use negative-distance as score sign).

We compute three variants from the SAC paper:
  - sac_test:    plain CIFAR-100 test images (any N from probe)
  - sac_wrong:   only test images target classifies WRONGLY
  - sac_jpeg:    JPEG-compressed test images (quality=10) -- SAC-JC variant
"""
from __future__ import annotations

import argparse, io, logging, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.behavioral import behavioral_features, forward_logits, label_aware_features, member_gap_features  # noqa: E402
from detect.model import load_state_dict, make_model  # noqa: E402
from detect.probe import build_member_probes, build_probe_set, CIFAR100_MEAN, CIFAR100_STD  # noqa: E402
from detect.weights import weight_features  # noqa: E402

OUT_DIR = Path("outputs/exp18")

SIGNS = {
    "behavioral_jsd": -1, "behavioral_wrong_agree_test": +1,
    "behavioral_loss_corr_member": +1, "behavioral_member_gap_diff_abs": -1,
    "weight_exact_tensor_frac": +1, "weight_l2_relative": -1,
    "sac_dist_test": -1,        # lower distance = stolen
    "sac_dist_wrong": -1,
    "sac_dist_jpeg": -1,
}


def rn(v):
    n = len(v); o = v.argsort(); r = np.empty(n); r[o] = np.arange(n)
    return r / max(n - 1, 1)


def ensemble(df):
    used = [c for c in SIGNS if c in df.columns]
    R = np.stack([rn(df[c].to_numpy() * SIGNS[c]) for c in used], axis=1)
    return R.mean(axis=1)


def correlation_matrix(outputs: torch.Tensor) -> torch.Tensor:
    """Cosine similarity NxN matrix of model outputs (after softmax)."""
    probs = F.softmax(outputs, dim=1)  # (N, num_classes)
    norm = probs / (probs.norm(dim=1, keepdim=True) + 1e-12)
    return norm @ norm.T  # (N, N)


def sac_distance(C_target: torch.Tensor, C_suspect: torch.Tensor) -> float:
    """Mean L1 distance between two correlation matrices."""
    return float((C_target - C_suspect).abs().mean().item())


def jpeg_compress(images: torch.Tensor, quality: int = 10) -> torch.Tensor:
    """JPEG-compress each image (after de-normalizing), re-normalize, return tensor."""
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    out = []
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    for img in images:
        # de-normalize to [0, 1]
        x = (img.cpu() * std + mean).clamp(0, 1)
        pil = to_pil(x)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil_recon = Image.open(buf).convert("RGB")
        x_recon = norm(to_tensor(pil_recon))
        out.append(x_recon)
    return torch.stack(out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    p.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    p.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    p.add_argument("--features-out", type=Path, default=OUT_DIR / "features.csv")
    p.add_argument("--submission-out", type=Path, default=OUT_DIR / "submission.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-members", type=int, default=4000)
    p.add_argument("--n-nonmembers", type=int, default=4000)
    p.add_argument("--n-probe-sac", type=int, default=100,
                   help="Number of inputs for SAC correlation matrix (paper uses 50-100).")
    args = p.parse_args()
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("[exp18 SAC] device=%s n_probe=%d", device, args.n_probe_sac)

    target_sd = load_state_dict(args.target)
    target = make_model(); target.load_state_dict(target_sd, strict=True); target.eval().to(device)

    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    (mem_x, mem_y), (nm_x, nm_y) = build_member_probes(
        args.cifar_root, args.train_main_idx,
        n_members=args.n_members, n_nonmembers=args.n_nonmembers,
    )

    # Target's predictions on test, used to pick SAC probe samples
    t_test = forward_logits(target, test_x, args.batch_size, device)
    t_mem = forward_logits(target, mem_x, args.batch_size, device)
    t_nm = forward_logits(target, nm_x, args.batch_size, device)
    t_test_top1 = t_test.argmax(1)
    target_wrong_mask = (t_test_top1 != test_y)
    logging.info("target test acc: %.4f  (%d wrong of %d)",
                 (~target_wrong_mask).float().mean().item(), int(target_wrong_mask.sum()), len(test_y))

    # ===== Build 3 SAC probe sets =====
    n_sac = args.n_probe_sac
    # Variant 1: plain test (first n_sac)
    probe_test = test_x[:n_sac]
    # Variant 2: target-wrong samples (first n_sac of the wrong ones)
    wrong_idx = torch.nonzero(target_wrong_mask).flatten()[:n_sac]
    probe_wrong = test_x[wrong_idx]
    # Variant 3: JPEG-compressed first n_sac
    logging.info("computing JPEG-compressed probes...")
    probe_jpeg = jpeg_compress(test_x[:n_sac], quality=10)
    logging.info("SAC probe sizes: test=%d wrong=%d jpeg=%d",
                 probe_test.size(0), probe_wrong.size(0), probe_jpeg.size(0))

    # Target's correlation matrices
    def cmat(model, x):
        logits = forward_logits(model, x, args.batch_size, device)
        return correlation_matrix(logits)
    t_C_test = cmat(target, probe_test)
    t_C_wrong = cmat(target, probe_wrong) if probe_wrong.size(0) > 0 else None
    t_C_jpeg = cmat(target, probe_jpeg)
    logging.info("target C shapes: %s %s %s", tuple(t_C_test.shape),
                 tuple(t_C_wrong.shape) if t_C_wrong is not None else "(none)", tuple(t_C_jpeg.shape))

    paths = sorted(args.suspects.glob("suspect_*.safetensors"))
    rows = []
    for path in tqdm(paths, desc="exp18"):
        row = {"id": int(path.stem.split("_")[-1]), "status": "ok"}
        try:
            sd = load_state_dict(path)
            m = make_model(); m.load_state_dict(sd, strict=True); m.eval().to(device)
            # Existing reference features
            s_test = forward_logits(m, test_x, args.batch_size, device)
            s_mem = forward_logits(m, mem_x, args.batch_size, device)
            s_nm = forward_logits(m, nm_x, args.batch_size, device)
            row.update(behavioral_features(t_test, s_test))
            row.update(label_aware_features(t_test, s_test, test_y, prefix="test"))
            row.update(label_aware_features(t_mem, s_mem, mem_y, prefix="member"))
            row.update(member_gap_features(t_mem, t_nm, s_mem, s_nm, mem_y, nm_y))
            row.update(weight_features(target_sd, sd))
            # SAC distances
            s_C_test = cmat(m, probe_test)
            row["sac_dist_test"] = sac_distance(t_C_test, s_C_test)
            if t_C_wrong is not None:
                s_C_wrong = cmat(m, probe_wrong)
                row["sac_dist_wrong"] = sac_distance(t_C_wrong, s_C_wrong)
            s_C_jpeg = cmat(m, probe_jpeg)
            row["sac_dist_jpeg"] = sac_distance(t_C_jpeg, s_C_jpeg)
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
    print("[exp18] top-15:")
    print(sub.sort_values("score", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
