#!/usr/bin/env python3
"""Experiment 13 — Train a shadow model + save its probe statistics.

After training a shadow with target's exact training recipe (biased crop,
train_main_idx subset, target's hyperparams) but a different random seed,
the shadow forwards the three probe sets (test / member / non-member)
and we save *only* the per-sample summary statistics needed for the
LiRA-style attribution score in exp14:

  - per-sample argmax prediction (uint16)
  - per-sample top-1 confidence (float16)
  - per-sample CE loss against true label (float16)

This is ~100 KB per shadow vs 45 MB for the full checkpoint, so we can
launch 32+ shadows in parallel without exhausting the contended NFS home.
"""
from __future__ import annotations

import argparse, json, logging, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.behavioral import forward_logits  # noqa: E402
from detect.model import make_model  # noqa: E402
from detect.probe import build_member_probes, build_probe_set  # noqa: E402


class BiasedRandomCrop:
    def __init__(self, size=32, pad=4, bias_x=0.5, bias_y=-0.25, jitter=0.25):
        self.size = size
        self.pad = pad
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.jitter = jitter

    def __call__(self, img):
        padded = TF.pad(img, [self.pad], padding_mode="reflect")
        W, H = padded.size
        pr_w = W - self.size
        pr_h = H - self.size
        jx = random.uniform(-self.jitter, self.jitter)
        jy = random.uniform(-self.jitter, self.jitter)
        cx = (self.bias_x + jx + 1) / 2.0
        cy = (self.bias_y + jy + 1) / 2.0
        ox = max(0, min(pr_w, int(round(cx * pr_w))))
        oy = max(0, min(pr_h, int(round(cy * pr_h))))
        return TF.crop(padded, oy, ox, self.size, self.size)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--stats-out", type=Path, required=True,
                   help="Path to save per-probe statistics (.npz, ~100 KB).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    p.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-members", type=int, default=4000)
    p.add_argument("--n-nonmembers", type=int, default=4000)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("[exp13 seed=%d] device=%s epochs=%d", args.seed, device, args.epochs)

    train_tf = transforms.Compose([
        BiasedRandomCrop(size=32, pad=4, bias_x=0.5, bias_y=-0.25, jitter=0.25),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    full = datasets.CIFAR100(root=str(args.cifar_root), train=True, download=True, transform=train_tf)
    with open(args.train_main_idx) as fh:
        idx = sorted(set(int(i) for i in json.load(fh)))
    train_ds = Subset(full, idx)
    logging.info("train subset: %d images", len(train_ds))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True)

    model = make_model().to(device)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
            total += float(loss.item()) * x.size(0); n += x.size(0)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            logging.info("epoch %2d/%d  loss=%.4f  time=%.1fs", ep+1, args.epochs, total/max(n,1), time.time()-t0)

    # ----- After training: forward probes and save stats -----
    logging.info("training done; computing probe stats")
    model.eval()
    test_x, test_y = build_probe_set(args.cifar_root, train=False)
    (mem_x, mem_y), (nm_x, nm_y) = build_member_probes(
        args.cifar_root, args.train_main_idx,
        n_members=args.n_members, n_nonmembers=args.n_nonmembers,
    )

    def probe(x, y, name):
        with torch.no_grad():
            logits = forward_logits(model, x, args.batch_size, device)
            ce = F.cross_entropy(logits, y, reduction='none')
            top1 = logits.argmax(1)
            conf = F.softmax(logits, dim=1).max(dim=1).values
        logging.info("%s: shape=%s top-1 acc=%.4f mean_loss=%.4f", name, tuple(logits.shape),
                     float((top1 == y).float().mean().item()), float(ce.mean().item()))
        return top1.cpu().numpy().astype(np.int32), conf.cpu().numpy().astype(np.float32), ce.cpu().numpy().astype(np.float32)

    test_top1, test_conf, test_loss = probe(test_x, test_y, "test")
    mem_top1, mem_conf, mem_loss = probe(mem_x, mem_y, "member")
    nm_top1, nm_conf, nm_loss = probe(nm_x, nm_y, "nonmember")

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.stats_out,
        seed=args.seed,
        test_top1=test_top1, test_conf=test_conf, test_loss=test_loss,
        mem_top1=mem_top1, mem_conf=mem_conf, mem_loss=mem_loss,
        nm_top1=nm_top1, nm_conf=nm_conf, nm_loss=nm_loss,
    )
    size_kb = args.stats_out.stat().st_size / 1024
    logging.info("saved %s (%.1f KB)", args.stats_out, size_kb)


if __name__ == "__main__":
    main()
