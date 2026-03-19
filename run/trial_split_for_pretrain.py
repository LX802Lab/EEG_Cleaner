#!/usr/bin/env python3
"""Build pretraining samples by aligning clean EEG windows with DE windows.

Input:
- data/clean/SEED-*.mat, key: data (62, N)
- data/de/SEED-*.mat, key: de (T, 62, 5)

Output:
- data/emotion-detect/pretrainData/SEED-{dataset}-{person}-{sample_idx}.npz
  keys:
    - cleaned_raw: shape (2048, 62)
  - de_feature: shape (40, 62, 5)

Mapping rule:
- clean window length is 2048 points
- DE frame centers are mapped to clean sample time range
- keep 40 DE points per clean window (center-cropped), equivalent to
  approximately 42 mapped points with first/last removed.
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio


CLEAN_DIR = os.path.join("data", "clean")
DE_DIR = os.path.join("data", "de")
OUT_DIR = os.path.join("data", "emotion-detect", "pretrainData")

SAMPLING_RATE = 200.0
CLEAN_WIN = 2048
CLEAN_OVERLAP = 0.5
CLEAN_STRIDE = int(round(CLEAN_WIN * (1.0 - CLEAN_OVERLAP)))
DE_WINDOW_SEC = 0.5
DE_OVERLAP = 0.5
DE_STEP_SEC = DE_WINDOW_SEC * (1.0 - DE_OVERLAP)  # 0.25
TARGET_DE_LEN = 40


def parse_seed_name(path: str) -> Tuple[str, str, str]:
    """Parse SEED-{dataset}-{person}-{exp...} from basename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"SEED-([A-Z]+)-(\d+)-(.+)$", stem)
    if not m:
        raise ValueError(f"Invalid file name: {path}")
    return m.group(1), m.group(2), m.group(3)


def load_clean(path: str) -> np.ndarray:
    mat = sio.loadmat(path)
    if "data" not in mat:
        raise ValueError(f"Missing data key in clean mat: {path}")
    arr = np.asarray(mat["data"], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Clean data must be 2D: {path}")
    if arr.shape[0] != 62 and arr.shape[1] == 62:
        arr = arr.T
    if arr.shape[0] != 62:
        raise ValueError(f"Clean data channels must be 62, got {arr.shape} in {path}")
    return arr


def load_de(path: str) -> np.ndarray:
    mat = sio.loadmat(path)
    if "de" not in mat:
        raise ValueError(f"Missing de key in DE mat: {path}")
    arr = np.asarray(mat["de"], dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 62 or arr.shape[2] != 5:
        raise ValueError(f"Invalid DE shape {arr.shape} in {path}")
    return arr


def center_crop_indices(indices: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop index vector to fixed target length."""
    n = int(indices.size)
    if n < target_len:
        raise ValueError(f"Index length {n} < target length {target_len}")
    left = (n - target_len) // 2
    return indices[left : left + target_len]


def map_clean_window_to_de_indices(
    clean_start: int,
    clean_end: int,
    de_len: int,
    sfreq: float,
    de_step_sec: float,
    de_window_sec: float,
    target_len: int,
) -> np.ndarray:
    """Map clean sample window to DE frame indices by DE window centers."""
    t0 = float(clean_start) / float(sfreq)
    # Inclusive end time based on sample index.
    t1 = float(max(clean_end - 1, clean_start)) / float(sfreq)

    centers = np.arange(de_len, dtype=np.float64) * float(de_step_sec) + float(de_window_sec) / 2.0
    picked = np.where((centers >= t0) & (centers <= t1))[0]
    if picked.size < target_len:
        return np.asarray([], dtype=np.int64)

    # For 2048 points this is usually around 42; center-crop keeps stable 40.
    return center_crop_indices(picked.astype(np.int64), target_len)


def split_clean_starts(total_len: int, win_len: int, stride: int) -> List[int]:
    if total_len < win_len:
        return []
    return list(range(0, total_len - win_len + 1, stride))


def clear_old_npz(out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    old = glob.glob(os.path.join(out_dir, "*.npz"))
    for path in old:
        os.remove(path)
    return len(old)


def build_pretrain_dataset(
    clean_dir: str,
    de_dir: str,
    out_dir: str,
    clean_win: int = CLEAN_WIN,
    clean_stride: int = CLEAN_STRIDE,
    sfreq: float = SAMPLING_RATE,
    de_step_sec: float = DE_STEP_SEC,
    de_window_sec: float = DE_WINDOW_SEC,
    target_de_len: int = TARGET_DE_LEN,
) -> Dict[str, int]:
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "SEED-*.mat")))
    de_files = sorted(glob.glob(os.path.join(de_dir, "SEED-*.mat")))

    de_map = {os.path.splitext(os.path.basename(p))[0]: p for p in de_files}

    removed = clear_old_npz(out_dir)
    matched = 0
    saved = 0
    skipped_short = 0
    skipped_de_map = 0

    # per (dataset, person) sample counter
    sample_counter: Dict[Tuple[str, str], int] = {}

    for clean_path in clean_files:
        base = os.path.splitext(os.path.basename(clean_path))[0]
        de_path = de_map.get(base)
        if de_path is None:
            skipped_de_map += 1
            continue

        dataset, person, _ = parse_seed_name(clean_path)
        key = (dataset, person)
        if key not in sample_counter:
            sample_counter[key] = 0

        clean = load_clean(clean_path)
        de = load_de(de_path)
        starts = split_clean_starts(clean.shape[1], clean_win, clean_stride)
        if not starts:
            skipped_short += 1
            continue

        matched += 1

        for s in starts:
            e = s + clean_win
            de_idx = map_clean_window_to_de_indices(
                clean_start=s,
                clean_end=e,
                de_len=de.shape[0],
                sfreq=sfreq,
                de_step_sec=de_step_sec,
                de_window_sec=de_window_sec,
                target_len=target_de_len,
            )
            if de_idx.size != target_de_len:
                continue

            clean_seg = clean[:, s:e]
            de_seg = de[de_idx, :, :]

            sample_counter[key] += 1
            out_name = f"SEED-{dataset}-{person}-{sample_counter[key]}.npz"
            out_path = os.path.join(out_dir, out_name)
            np.savez_compressed(
                out_path,
                cleaned_raw=np.asarray(clean_seg.T, dtype=np.float32),
                de_feature=np.asarray(de_seg, dtype=np.float32),
            )
            saved += 1

    return {
        "clean_files": len(clean_files),
        "de_files": len(de_files),
        "matched_files": matched,
        "saved_samples": saved,
        "removed_old_npz": removed,
        "skipped_no_matching_de": skipped_de_map,
        "skipped_too_short": skipped_short,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split clean+de into pretrain aligned samples")
    parser.add_argument("--clean-dir", default=CLEAN_DIR)
    parser.add_argument("--de-dir", default=DE_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--clean-win", type=int, default=CLEAN_WIN)
    parser.add_argument("--clean-stride", type=int, default=CLEAN_STRIDE)
    parser.add_argument("--sfreq", type=float, default=SAMPLING_RATE)
    parser.add_argument("--de-step-sec", type=float, default=DE_STEP_SEC)
    parser.add_argument("--de-window-sec", type=float, default=DE_WINDOW_SEC)
    parser.add_argument("--target-de-len", type=int, default=TARGET_DE_LEN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_pretrain_dataset(
        clean_dir=args.clean_dir,
        de_dir=args.de_dir,
        out_dir=args.out_dir,
        clean_win=int(args.clean_win),
        clean_stride=int(args.clean_stride),
        sfreq=float(args.sfreq),
        de_step_sec=float(args.de_step_sec),
        de_window_sec=float(args.de_window_sec),
        target_de_len=int(args.target_de_len),
    )

    summary_path = os.path.join(args.out_dir, "pretrain_split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
