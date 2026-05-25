#!/usr/bin/env python3
"""Experiment 16 — Multi-layer activation correlation.

Hook layer1, layer2, layer3, layer4, avgpool on target + each suspect.
Compute per-suspect, per-layer activation similarity (Linear CKA + flat
per-sample cosine). Earlier layers capture mid-level features that
distilled-from-target stolen models often retain even when their output
logits and penultimate features have drifted.
"""
from __future__ import annotations

import argparse, logging, sys, time
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
from detect.representations import linear_cka  # noqa: E402
from detect.weights import weight_features  # noqa: E402


OUT_DIR = Path("outputs/exp16")
LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
SIGNS = {
    "behavioral_jsd": -1, "behavioral_wrong_agree_test": +1,
    "behavioral_loss_corr_member": +1, "behavioral_member_gap_diff_abs": -1,
    "weight_exact_tensor_frac": +1, "weight_l2_relative": -1,
    "act_cka_layer1": +1, "act_cka_layer2": +1, "act_cka_layer3": +1,
    "act_cka_layer4": +1, "act_cka_avgpool": +1,
    "act_cos_layer1": +1, "act_cos_layer4": +1,
}


def rn(v):
    n = len(v); o = v.argsort(); r = np.empty(n); r[o] = np.arange(n)
    return r / max(n - 1, 1)


def ensemble(df):
    used = [c for c in SIGNS if c in df.columns]
    R = np.stack([rn(df[c].to_numpy() * SIGNS[c]) for c in used], axis=1)
    return R.mean(axis=1)


@torch.no_grad()
def get_layer_outputs(model, x, layer_names, batch_size, device):
    captured = {name: [] for name in layer_names}
    handles = []
    name_to_module = dict(model.named_modules())
    for name in layer_names:
        module = name_to_module[name]
        def make_hook(n):
            def hook(_m, _inp, out):
                captured[n].append(out.detach().flatten(1).float().cpu())
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        for i in range(0, x.size(0), batch_size):
            chunk = x[i:i+batch_size].to(device, non_blocking=True)
            model(chunk)
    finally:
        for h in handles: h.remove()
    return {n: torch.cat(captured[n], dim=0) for n in layer_names}


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
    p.add_argument("--act-probe-size", type=int, default=1000)
    args = p.parse_args()
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("[exp16] device=%s layers=%s n_probe=%d", device, LAYERS, args.act_probe_size)

    target_sd = load_state_dict(args.target)
    target = make_model(); target.load_state_dict(target_sd, strict=True); target.eval().to(device)

    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    (mem_x, mem_y), (nm_x, nm_y) = build_member_probes(
        args.cifar_root, args.train_main_idx,
        n_members=args.n_members, n_nonmembers=args.n_nonmembers,
    )
    act_x = test_x[:args.act_probe_size]

    t_test = forward_logits(target, test_x, args.batch_size, device)
    t_mem = forward_logits(target, mem_x, args.batch_size, device)
    t_nm = forward_logits(target, nm_x, args.batch_size, device)
    t_layers = get_layer_outputs(target, act_x, LAYERS, args.batch_size, device)
    for n, t in t_layers.items():
        logging.info("target %s shape: %s", n, tuple(t.shape))

    paths = sorted(args.suspects.glob("suspect_*.safetensors"))
    rows = []
    for path in tqdm(paths, desc="exp16"):
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
            s_layers = get_layer_outputs(m, act_x, LAYERS, args.batch_size, device)
            for name in LAYERS:
                row[f"act_cka_{name}"] = linear_cka(t_layers[name], s_layers[name])
                # per-sample cosine of flattened activations
                tn = t_layers[name] / (t_layers[name].norm(dim=1, keepdim=True) + 1e-12)
                sn = s_layers[name] / (s_layers[name].norm(dim=1, keepdim=True) + 1e-12)
                row[f"act_cos_{name}"] = float((tn * sn).sum(dim=1).mean().item())
            row.update(weight_features(target_sd, sd))
            del m, s_layers
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
    print("[exp16] top-15:")
    print(sub.sort_values("score", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
