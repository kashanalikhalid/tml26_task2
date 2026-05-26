#!/usr/bin/env python3
"""Conferrable Adversarial Examples (Lukas et al., ICLR 2021).

Generate perturbations x' that simultaneously fool the target model and
leave independent shadow models correct. Stolen / derived models inherit
the target's adversarial vulnerabilities; truly independent models do not.

Score per suspect = fraction of conferrable AEs where the suspect predicts
the same wrong class as the target.
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

OUT_DIR = Path("outputs/conferrable_ae")


def joint_pgd_attack(target, shadow_models, x, y, eps, alpha, steps, alpha_shadow):
    x_adv = x.clone().detach() + (torch.rand_like(x) * 2 - 1) * eps
    x_adv = x_adv.clamp(-3, 3).detach().requires_grad_(True)
    for _ in range(steps):
        t_loss = -F.cross_entropy(target(x_adv), y)
        s_loss = 0.0
        for sm in shadow_models:
            s_loss = s_loss + F.cross_entropy(sm(x_adv), y)
        s_loss = s_loss / max(len(shadow_models), 1)
        total = t_loss + alpha_shadow * s_loss
        grad = torch.autograd.grad(total, x_adv)[0]
        x_adv = x_adv.detach() - alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(-3, 3)
        x_adv = x_adv.detach().requires_grad_(True)
    return x_adv.detach()


@torch.no_grad()
def predict(model, x, batch_size, device):
    out = []
    for i in range(0, x.size(0), batch_size):
        chunk = x[i:i + batch_size].to(device, non_blocking=True)
        out.append(model(chunk).argmax(1).cpu())
    return torch.cat(out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=Path("data/target_model/weights.safetensors"))
    p.add_argument("--shadows-glob", default="outputs/shadow_weights/shadow_*.safetensors")
    p.add_argument("--max-shadows", type=int, default=8)
    p.add_argument("--suspects", type=Path, default=Path("data/suspect_models"))
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    p.add_argument("--features-out", type=Path, default=OUT_DIR / "features.csv")
    p.add_argument("--submission-out", type=Path, default=OUT_DIR / "submission.csv")
    p.add_argument("--ae-cache", type=Path, default=OUT_DIR / "aes.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--pgd-alpha", type=float, default=2 / 255)
    p.add_argument("--pgd-steps", type=int, default=40)
    p.add_argument("--alpha-shadow", type=float, default=1.0)
    p.add_argument("--conferrability-thr", type=float, default=0.7)
    p.add_argument("--no-regen-aes", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target = make_model()
    target.load_state_dict(load_state_dict(args.target), strict=True)
    target.eval().to(device)
    for p_ in target.parameters():
        p_.requires_grad_(False)

    shadow_paths = sorted(glob.glob(args.shadows_glob))[:args.max_shadows]
    if not shadow_paths:
        raise SystemExit(f"no shadow weights at {args.shadows_glob}")
    shadow_models = []
    for sp in shadow_paths:
        sm = make_model()
        sm.load_state_dict(load_state_dict(sp), strict=True)
        sm.eval().to(device)
        for p_ in sm.parameters():
            p_.requires_grad_(False)
        shadow_models.append(sm)
    logging.info("loaded %d shadow models", len(shadow_models))

    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    seed_x = test_x[:args.n_seeds]
    seed_y = test_y[:args.n_seeds]

    if args.ae_cache.exists() and args.no_regen_aes:
        cache = torch.load(args.ae_cache)
        x_adv, t_wrong_class, confer_mask = cache["x_adv"], cache["t_wrong_class"], cache["confer_mask"]
    else:
        x_adv_list, t_list, sr_list = [], [], []
        for i in tqdm(range(0, args.n_seeds, args.batch_size), desc="generate"):
            xb = seed_x[i:i + args.batch_size].to(device)
            yb = seed_y[i:i + args.batch_size].to(device)
            xa = joint_pgd_attack(target, shadow_models, xb, yb,
                                  eps=args.eps, alpha=args.pgd_alpha,
                                  steps=args.pgd_steps, alpha_shadow=args.alpha_shadow)
            with torch.no_grad():
                t_pred = target(xa).argmax(1)
                shadow_right = torch.stack(
                    [(sm(xa).argmax(1) == yb).float() for sm in shadow_models], dim=0
                ).mean(0)
            x_adv_list.append(xa.cpu())
            t_list.append(t_pred.cpu())
            sr_list.append(shadow_right.cpu())
        x_adv = torch.cat(x_adv_list, dim=0)
        t_wrong_class = torch.cat(t_list, dim=0)
        shadow_right_frac = torch.cat(sr_list, dim=0)
        target_wrong = (t_wrong_class != seed_y)
        confer_mask = target_wrong & (shadow_right_frac >= args.conferrability_thr)
        logging.info("target wrong: %d/%d   conferrable: %d/%d",
                     int(target_wrong.sum()), len(target_wrong),
                     int(confer_mask.sum()), len(confer_mask))
        torch.save({"x_adv": x_adv, "t_wrong_class": t_wrong_class,
                    "confer_mask": confer_mask, "seed_y": seed_y}, args.ae_cache)

    n_confer = int(confer_mask.sum().item())
    if n_confer == 0:
        raise SystemExit("0 conferrable AEs; lower --conferrability-thr or --alpha-shadow")

    confer_x = x_adv[confer_mask]
    confer_t_wrong = t_wrong_class[confer_mask]
    logging.info("evaluating suspects on %d conferrable AEs", n_confer)

    paths = sorted(args.suspects.glob("suspect_*.safetensors"))
    rows = []
    for path in tqdm(paths, desc="eval"):
        row = {"id": int(path.stem.split("_")[-1]), "status": "ok"}
        try:
            m = make_model()
            m.load_state_dict(load_state_dict(path), strict=True)
            m.eval().to(device)
            s_pred = predict(m, confer_x, args.batch_size, device)
            row["confer_match_rate"] = float((s_pred == confer_t_wrong).float().mean().item())
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
        "score": ok["confer_match_rate"].fillna(0).to_numpy(),
    }).sort_values("id").reset_index(drop=True)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.submission_out, index=False)
    print(sub.sort_values("score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
