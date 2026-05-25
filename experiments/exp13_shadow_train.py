#!/usr/bin/env python3
"""Experiment 13 — Train a shadow model with target's exact training recipe.

For LiRA-style attribution: train M shadow ResNet-18s on the same
train_main_idx subset of CIFAR-100 with the same biased-crop augmentation
target used, but with different random seeds. The shadows are
"definitely-not-stolen" reference points trained on the same data.

A suspect that is *much* more similar to target than to the shadow
average is statistical evidence of stealing (vs. just being trained on
similar data).
"""
from __future__ import annotations

import argparse, json, logging, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect.model import make_model  # noqa: E402


class BiasedRandomCrop:
    """Random crop biased toward (bias_x, bias_y) with jitter, on PIL."""
    def __init__(self, size=32, pad=4, bias_x=0.5, bias_y=-0.25, jitter=0.25):
        self.size = size
        self.pad = pad
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.jitter = jitter

    def __call__(self, img):
        padded = TF.pad(img, [self.pad], padding_mode="reflect")
        W, H = padded.size  # PIL is (W, H)
        pad_room_w = W - self.size
        pad_room_h = H - self.size
        jx = random.uniform(-self.jitter, self.jitter)
        jy = random.uniform(-self.jitter, self.jitter)
        cx = (self.bias_x + jx + 1) / 2.0
        cy = (self.bias_y + jy + 1) / 2.0
        off_x = max(0, min(pad_room_w, int(round(cx * pad_room_w))))
        off_y = max(0, min(pad_room_h, int(round(cy * pad_room_h))))
        return TF.crop(padded, off_y, off_x, self.size, self.size)


def build_train_subset(cifar_root: Path, train_main_idx_path: Path):
    train_tf = transforms.Compose([
        BiasedRandomCrop(size=32, pad=4, bias_x=0.5, bias_y=-0.25, jitter=0.25),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    full = datasets.CIFAR100(root=str(cifar_root), train=True, download=True, transform=train_tf)
    with open(train_main_idx_path) as fh:
        idx = sorted(set(int(i) for i in json.load(fh)))
    return torch.utils.data.Subset(full, idx)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar100"))
    p.add_argument("--train-main-idx", type=Path, default=Path("data/target_model/train_main_idx.json"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("[exp13 seed=%d] device=%s epochs=%d", args.seed, device, args.epochs)

    train_ds = build_train_subset(args.cifar_root, args.train_main_idx)
    logging.info("train subset: %d images", len(train_ds))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        drop_last=True)

    model = make_model().to(device)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        sched.step()
        logging.info("epoch %2d/%d   loss=%.4f   time=%.1fs   lr=%.4f",
                     ep + 1, args.epochs, total / max(n, 1), time.time() - t0, opt.param_groups[0]["lr"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Save in safetensors format for consistency with target / suspects
    try:
        from safetensors.torch import save_file
        save_file(model.state_dict(), str(args.out))
    except ImportError:
        torch.save(model.state_dict(), args.out)
    logging.info("saved %s", args.out)


if __name__ == "__main__":
    main()
