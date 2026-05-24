#!/usr/bin/env python3
"""Download SprintML/tml26_task2 from HuggingFace into $TASK_DIR/data/.

Pure stdlib (urllib + ThreadPoolExecutor) so it works in any Python env
including the pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel docker image
which doesn't ship curl. Resumable via HTTP Range; idempotent for files
already at their expected size.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

TASK_DIR = os.environ.get("TASK_DIR", os.path.expanduser("~/tml26_task2"))
DATA_DIR = os.path.join(TASK_DIR, "data")
PARALLEL = int(os.environ.get("PARALLEL", "4"))
BASE = "https://huggingface.co/SprintML/tml26_task2/resolve/main"

SUSPECT_SIZE = 44_929_864  # bytes, verified from HF API
TARGET_WEIGHTS_SIZE = 44_929_864
RETRY = int(os.environ.get("DOWNLOAD_RETRY", "5"))


def fetch(url: str, dst: str, expected_size: int | None = None) -> str:
    """Download `url` to `dst`. Resume via HTTP Range when partial.
    Returns a one-line status string.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if expected_size is not None and os.path.exists(dst) and os.path.getsize(dst) == expected_size:
        return f"skip  {dst}  ({expected_size} bytes already present)"

    for attempt in range(1, RETRY + 1):
        start = os.path.getsize(dst) if os.path.exists(dst) else 0
        headers: dict[str, str] = {}
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                # status 200 = full content (server ignored Range); status 206 = partial.
                mode = "ab" if start > 0 and resp.status == 206 else "wb"
                if mode == "wb":
                    # If server returned 200 even though we asked for a range, restart from 0.
                    start = 0
                with open(dst, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)  # 1 MB
                        if not chunk:
                            break
                        f.write(chunk)
            size = os.path.getsize(dst)
            if expected_size is not None and size != expected_size:
                # Wrong size — discard and retry from scratch
                os.remove(dst)
                raise OSError(f"size mismatch: got {size}, expected {expected_size}")
            return f"ok    {dst}  ({size} bytes)"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if attempt == RETRY:
                return f"fail  {dst}  ({type(exc).__name__}: {exc})"
            wait = 2 ** attempt
            print(f"retry {attempt}/{RETRY} after {exc}; sleep {wait}s for {dst}", flush=True)
            time.sleep(wait)
    return f"fail  {dst}  (all retries exhausted)"


def build_jobs() -> list[tuple[str, str, int | None]]:
    jobs: list[tuple[str, str, int | None]] = [
        (f"{BASE}/target_model/weights.safetensors",
         os.path.join(DATA_DIR, "target_model/weights.safetensors"),
         TARGET_WEIGHTS_SIZE),
        (f"{BASE}/target_model/train_main_idx.json",
         os.path.join(DATA_DIR, "target_model/train_main_idx.json"),
         None),
        (f"{BASE}/task_template.py",
         os.path.join(TASK_DIR, "task_template.py"),
         None),
        (f"{BASE}/submission.py",
         os.path.join(TASK_DIR, "submission.py"),
         None),
    ]
    for i in range(360):
        jobs.append((
            f"{BASE}/suspect_models/suspect_{i:03d}.safetensors",
            os.path.join(DATA_DIR, f"suspect_models/suspect_{i:03d}.safetensors"),
            SUSPECT_SIZE,
        ))
    return jobs


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"::: TASK_DIR={TASK_DIR}  DATA_DIR={DATA_DIR}  PARALLEL={PARALLEL}")
    jobs = build_jobs()
    print(f"::: {len(jobs)} files to fetch")

    failures = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(fetch, url, dst, sz): dst for url, dst, sz in jobs}
        for i, fut in enumerate(as_completed(futures), start=1):
            line = fut.result()
            if line.startswith("fail"):
                failures += 1
            print(f"[{i:3d}/{len(jobs)}] {line}", flush=True)

    dt = time.time() - t0
    print(f"::: done in {dt:.1f}s with {failures} failures")

    # Final inventory
    n_full = 0
    n_partial = 0
    for _, dst, sz in jobs:
        if not os.path.exists(dst):
            continue
        if sz is not None and os.path.getsize(dst) == sz:
            n_full += 1
        else:
            n_partial += 1
    print(f"::: suspect+target files at expected size: {n_full}")
    print(f"::: suspect+target files at wrong/unknown size: {n_partial}")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
