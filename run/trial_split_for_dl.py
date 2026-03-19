#!/usr/bin/env python3
"""Split SEED7 trial files into smaller overlapping samples.

Input:
- data/emotion-detect/seed7_trial/SEED-VII-{subject}-{session}-{trial}.npz

Output:
- data/emotion-detect/8s
- data/emotion-detect/16s
- data/emotion-detect/32s

For each trial file, this script:
1) computes window lengths from split seconds and DE time step
2) splits with 50% overlap
3) aligns windows to the center of the trial by first computing n windows:
   total_needed = win_len + (n - 1) * (win_len / 2)
4) computes each window score from subject continuous labels (trial mean over window)
5) keeps windows with score > 20
6) saves each kept window to one npz file containing only data and label
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.io as sio


TRIAL_DIR = os.path.join("data", "emotion-detect", "seed7_trial")
SEED7_ROOT = "/mnt/data/Datasets/EEG/SJTU-SEED/seed7"
CONT_LABEL_DIR = os.path.join(SEED7_ROOT, "continuous_labels")
OUT_ROOT = os.path.join("data", "emotion-detect")

# User requirement uses 0.25s step for converting seconds to DE frames.
DEFAULT_DE_STEP_SEC = 0.25
SPLIT_SECONDS = (8, 16, 32)
LABEL_THRESHOLD = 20.0


def parse_trial_file_name(path: str) -> Tuple[int, int, int]:
    """Parse subject/session/trial from SEED-VII-{subject}-{session}-{trial}.npz."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"SEED-VII-(\d+)-(\d+)-(\d+)$", stem)
    if not m:
        raise ValueError(f"Invalid trial file name: {path}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def trial_global_index(session: int, trial: int) -> int:
    """Map (session 1..4, trial 1..20) to global trial index 1..80."""
    return (session - 1) * 20 + trial


def load_trial_data(path: str) -> np.ndarray:
    """Load trial DE data from npz, expected shape (T, 62, 5)."""
    npz = np.load(path)
    if "data" not in npz:
        raise ValueError(f"Missing 'data' in trial npz: {path}")
    data = np.asarray(npz["data"], dtype=np.float32)
    if data.ndim != 3 or data.shape[1] != 62 or data.shape[2] != 5:
        raise ValueError(f"Unexpected data shape {data.shape} in {path}")
    return data


def load_subject_continuous_labels(subject: int, cont_label_dir: str) -> Dict[int, np.ndarray]:
    """Load subject continuous labels as mapping trial_idx(1..80) -> 1D array."""
    path = os.path.join(cont_label_dir, f"{subject}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing continuous labels file: {path}")

    mat = sio.loadmat(path)
    out: Dict[int, np.ndarray] = {}
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if not key.isdigit():
            continue
        out[int(key)] = np.asarray(value, dtype=np.float32).reshape(-1)
    return out


def centered_window_starts(total_len: int, win_len: int, stride: int) -> List[int]:
    """Compute centered starts for fixed-length windows with fixed stride.

    n is the largest count satisfying:
    win_len + (n - 1) * stride <= total_len

    Then we center the whole covered span inside [0, total_len).
    """
    if total_len < win_len:
        return []

    n = 1 + (total_len - win_len) // stride
    covered = win_len + (n - 1) * stride
    left = (total_len - covered) // 2
    return [int(left + i * stride) for i in range(int(n))]


def segment_label_from_continuous_index(
    cont_label: np.ndarray,
    trial_len: int,
    seg_start: int,
    seg_end: int,
) -> float:
    """Estimate segment score by interpolation from continuous labels to trial length.

    The continuous label sequence is first interpolated to match trial frame count,
    then the segment label is computed as the mean value inside [seg_start, seg_end).
    """
    y = np.asarray(cont_label, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return float("nan")
    if trial_len <= 0:
        return float("nan")
    if y.size == 1:
        return float(y[0])

    max_idx = float(max(trial_len - 1, 1))
    x = np.linspace(0.0, max_idx, y.size, endpoint=True, dtype=np.float64)
    dense_x = np.arange(trial_len, dtype=np.float64)
    dense_y = np.interp(dense_x, x, y.astype(np.float64))

    s = int(np.clip(seg_start, 0, trial_len))
    e = int(np.clip(seg_end, 0, trial_len))
    if e <= s:
        return float("nan")
    return float(np.mean(dense_y[s:e]))


def save_sample(out_path: str, data: np.ndarray, label: float) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        data=np.asarray(data, dtype=np.float32),
        label=np.asarray(label, dtype=np.float32),
    )


def build_dataset(
    trial_dir: str,
    cont_label_dir: str,
    out_root: str,
    split_seconds: Sequence[int],
    de_step_sec: float,
    label_threshold: float,
) -> Dict[str, dict]:
    files = sorted(glob.glob(os.path.join(trial_dir, "SEED-VII-*.npz")))
    if not files:
        raise FileNotFoundError(f"No trial files found in {trial_dir}")

    # per-subject continuous labels cache
    cont_cache: Dict[int, Dict[int, np.ndarray]] = {}

    # stats
    summary: Dict[str, dict] = {}
    kept_counter: Dict[int, int] = {sec: 0 for sec in split_seconds}
    dropped_by_label: Dict[int, int] = {sec: 0 for sec in split_seconds}
    dropped_by_length: Dict[int, int] = {sec: 0 for sec in split_seconds}

    for sec in split_seconds:
        summary[f"{sec}s"] = {
            "window_len": int(round(sec / de_step_sec)),
            "stride": int(round(sec / de_step_sec)) // 2,
            "saved": 0,
            "dropped_by_label": 0,
            "dropped_by_length": 0,
        }

    # ensure output dirs exist
    for sec in split_seconds:
        os.makedirs(os.path.join(out_root, f"{sec}s"), exist_ok=True)

    for path in files:
        subject, session, trial = parse_trial_file_name(path)
        gtrial = trial_global_index(session, trial)

        trial_data = load_trial_data(path)
        tlen = int(trial_data.shape[0])

        if subject not in cont_cache:
            cont_cache[subject] = load_subject_continuous_labels(subject, cont_label_dir)
        cont_map = cont_cache[subject]
        cont = cont_map.get(gtrial)
        if cont is None or cont.size == 0:
            for sec in split_seconds:
                dropped_by_label[sec] += 1
            continue

        for sec in split_seconds:
            win_len = int(round(sec / de_step_sec))
            stride = max(1, win_len // 2)

            starts = centered_window_starts(total_len=tlen, win_len=win_len, stride=stride)
            if not starts:
                dropped_by_length[sec] += 1
                continue

            kept_paths: List[str] = []
            for sidx in starts:
                eidx = sidx + win_len
                seg = trial_data[sidx:eidx, :, :]
                score = segment_label_from_continuous_index(cont, tlen, sidx, eidx)
                if not np.isfinite(score) or score <= label_threshold:
                    dropped_by_label[sec] += 1
                    continue

                out_dir = os.path.join(out_root, f"{sec}s")
                # Avoid name collision when one trial produces multiple samples.
                sample_idx = len(kept_paths) + 1
                out_path = os.path.join(out_dir, f"SEED-VII-{subject}-{gtrial}-{sample_idx}.npz")
                save_sample(out_path, seg, score)
                kept_paths.append(out_path)
                kept_counter[sec] += 1

    for sec in split_seconds:
        summary[f"{sec}s"]["saved"] = int(kept_counter[sec])
        summary[f"{sec}s"]["dropped_by_label"] = int(dropped_by_label[sec])
        summary[f"{sec}s"]["dropped_by_length"] = int(dropped_by_length[sec])

    summary["meta"] = {
        "trial_files": len(files),
        "de_step_sec": float(de_step_sec),
        "label_threshold": float(label_threshold),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split SEED7 trial files into small samples")
    parser.add_argument("--trial-dir", default=TRIAL_DIR, help="Input trial npz directory")
    parser.add_argument("--seed7-root", default=SEED7_ROOT, help="SEED7 dataset root path")
    parser.add_argument("--out-root", default=OUT_ROOT, help="Output root for sample npz files")
    parser.add_argument(
        "--split-seconds",
        default=",".join(str(x) for x in SPLIT_SECONDS),
        help="Comma-separated split lengths in seconds, e.g. 8,16,32",
    )
    parser.add_argument(
        "--de-step-sec",
        type=float,
        default=DEFAULT_DE_STEP_SEC,
        help="Frame step used for sec->frame conversion (default 0.25)",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=LABEL_THRESHOLD,
        help="Keep sample only when mean continuous label > threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_seconds = tuple(int(x.strip()) for x in str(args.split_seconds).split(",") if x.strip())
    cont_label_dir = os.path.join(args.seed7_root, "continuous_labels")

    summary = build_dataset(
        trial_dir=args.trial_dir,
        cont_label_dir=cont_label_dir,
        out_root=args.out_root,
        split_seconds=split_seconds,
        de_step_sec=float(args.de_step_sec),
        label_threshold=float(args.label_threshold),
    )

    os.makedirs(args.out_root, exist_ok=True)
    summary_path = os.path.join(args.out_root, "seed7_trial_split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
