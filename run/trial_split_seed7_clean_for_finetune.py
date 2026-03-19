#!/usr/bin/env python3
"""Split SEED-VII clean EEG into 2048-point samples with emotion labels.

References:
- run/make_labels_seed7.py for trigger/label parsing
- run/trial_split_for_dl.py for centered split logic

Input:
- data/clean/SEED-VII-{subject}-{session}.mat (key: data, shape 62 x N)
- /mnt/data/Datasets/EEG/SJTU-SEED/seed7/save/*_trigger_info.csv
- /mnt/data/Datasets/EEG/SJTU-SEED/seed7/emotion_label_and_stimuli_order.xlsx

Output:
- data/emotion-detect/fine-turning/set1-ED/SEED-VII-{person}-{sample_idx}.npz
  keys:
    - data: shape (2048, 62)
  - label: int32 emotion class
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio


CLEAN_DIR = os.path.join("data", "clean")
SEED7_ROOT = "/mnt/data/Datasets/EEG/SJTU-SEED/seed7"
SAVE_DIR = os.path.join(SEED7_ROOT, "save")
EMOTION_XLSX = os.path.join(SEED7_ROOT, "emotion_label_and_stimuli_order.xlsx")
OUT_DIR = os.path.join("data", "emotion-detect", "fine-turning", "set1-ED")

SAMPLING_RATE = 200.0
SAMPLE_LEN = 2048
SAMPLE_OVERLAP = 0.5
SAMPLE_STRIDE = int(round(SAMPLE_LEN * (1.0 - SAMPLE_OVERLAP)))

EMOTION_TO_LABEL = {
    "Disgust": 0,
    "Fear": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4,
    "Anger": 5,
    "Surprise": 6,
}


def read_csv_try(path: str) -> pd.DataFrame:
    for sep in (",", "\t", ";"):
        try:
            df = pd.read_csv(path, header=None, sep=sep, engine="python")
            if df.shape[1] >= 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, header=None, engine="python")


def parse_seed7_name(path: str) -> Tuple[int, int]:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"SEED-VII-(\d+)-(\d+)$", stem)
    if not m:
        raise ValueError(f"Invalid SEED-VII clean file name: {path}")
    return int(m.group(1)), int(m.group(2))


def find_trigger_file(subject: int, session: int, save_dir: str) -> Optional[str]:
    pattern = os.path.join(save_dir, f"{subject}_*_{session}_trigger_info.csv")
    files = sorted(glob.glob(pattern))
    return files[0] if files else None


def parse_trigger_pairs(trigger_csv: str) -> List[Tuple[float, float]]:
    df = read_csv_try(trigger_csv)
    if df.shape[1] < 2:
        return []

    markers = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    ts = pd.to_datetime(df.iloc[:, 1], errors="coerce")
    valid = ts.notna()
    if valid.sum() == 0:
        return []

    t0 = ts[valid].iloc[0]
    sec = (ts - t0).dt.total_seconds()
    starts = sec[markers == 1].dropna().tolist()
    ends = sec[markers == 2].dropna().tolist()

    n = min(len(starts), len(ends))
    out = []
    for i in range(n):
        s = float(starts[i])
        e = float(ends[i])
        if e > s:
            out.append((s, e))
    return out


def parse_emotion_order_xlsx(xlsx_path: str) -> Dict[int, List[int]]:
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Missing label xlsx: {xlsx_path}")

    df = pd.read_excel(xlsx_path, header=None)
    session_map: Dict[int, List[int]] = {}

    for row_idx in [1, 3, 5, 7]:
        if row_idx >= len(df):
            continue
        row = df.iloc[row_idx].tolist()
        head = str(row[0]).strip() if row else ""
        m = re.search(r"Session\s*(\d+)", head, flags=re.IGNORECASE)
        if not m:
            continue

        session = int(m.group(1))
        labels: List[int] = []
        for value in row[1:21]:
            emo = str(value).strip()
            mapped = EMOTION_TO_LABEL.get(emo)
            if mapped is None:
                low = emo.lower()
                mapped = -1
                for key, label in EMOTION_TO_LABEL.items():
                    if key.lower() == low:
                        mapped = int(label)
                        break
            labels.append(int(mapped))

        if len(labels) != 20:
            raise ValueError(f"Session {session} labels count != 20")
        session_map[session] = labels

    if len(session_map) != 4:
        raise ValueError(f"Expected 4 sessions in xlsx, got {len(session_map)}")
    return session_map


def load_clean_data(path: str) -> Tuple[np.ndarray, float]:
    mat = sio.loadmat(path)
    if "data" not in mat:
        raise ValueError(f"Missing data key in clean mat: {path}")

    data = np.asarray(mat["data"], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Clean data must be 2D: {path}")
    if data.shape[0] != 62 and data.shape[1] == 62:
        data = data.T
    if data.shape[0] != 62:
        raise ValueError(f"Clean data channels must be 62, got {data.shape} in {path}")

    sfreq = SAMPLING_RATE
    if "sfreq" in mat:
        value = float(np.asarray(mat["sfreq"]).squeeze())
        if value > 0:
            sfreq = value
    return data, sfreq


def centered_window_starts(total_len: int, win_len: int, stride: int) -> List[int]:
    if total_len < win_len:
        return []
    n = 1 + (total_len - win_len) // stride
    covered = win_len + (n - 1) * stride
    left = (total_len - covered) // 2
    return [int(left + i * stride) for i in range(int(n))]


def clear_old_npz(out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    old = glob.glob(os.path.join(out_dir, "*.npz"))
    for path in old:
        os.remove(path)
    return len(old)


def split_seed7_clean_for_finetune(
    clean_dir: str,
    save_dir: str,
    xlsx_path: str,
    out_dir: str,
    sample_len: int = SAMPLE_LEN,
    stride: int = SAMPLE_STRIDE,
) -> Dict[str, int]:
    files = sorted(glob.glob(os.path.join(clean_dir, "SEED-VII-*.mat")))
    if not files:
        raise FileNotFoundError(f"No SEED-VII clean files in {clean_dir}")

    session_labels = parse_emotion_order_xlsx(xlsx_path)
    removed = clear_old_npz(out_dir)

    sample_counter: Dict[int, int] = {}

    saved = 0
    skipped_missing_trigger = 0
    skipped_invalid_label = 0
    skipped_short_trial = 0

    for file_path in files:
        subject, session = parse_seed7_name(file_path)
        trigger_csv = find_trigger_file(subject, session, save_dir)
        if not trigger_csv:
            skipped_missing_trigger += 1
            continue

        trial_pairs = parse_trigger_pairs(trigger_csv)
        labels = session_labels.get(session, [])
        n_trial = min(len(trial_pairs), len(labels))
        if n_trial <= 0:
            continue

        data, sfreq = load_clean_data(file_path)
        total_n = data.shape[1]

        if subject not in sample_counter:
            sample_counter[subject] = 0

        for i in range(n_trial):
            label = int(labels[i])
            if label < 0:
                skipped_invalid_label += 1
                continue

            start_sec, end_sec = trial_pairs[i]
            sidx = max(0, int(round(start_sec * sfreq)))
            eidx = min(total_n, int(round(end_sec * sfreq)))
            if eidx <= sidx:
                continue

            trial_data = data[:, sidx:eidx]
            starts = centered_window_starts(trial_data.shape[1], sample_len, stride)
            if not starts:
                skipped_short_trial += 1
                continue

            for st in starts:
                ed = st + sample_len
                seg = trial_data[:, st:ed]
                sample_counter[subject] += 1
                out_name = f"SEED-VII-{subject}-{sample_counter[subject]}.npz"
                out_path = os.path.join(out_dir, out_name)
                np.savez_compressed(
                    out_path,
                    data=np.asarray(seg.T, dtype=np.float32),
                    label=np.asarray(label, dtype=np.int32),
                )
                saved += 1

    return {
        "seed7_clean_files": len(files),
        "saved_samples": saved,
        "removed_old_npz": removed,
        "skipped_missing_trigger": skipped_missing_trigger,
        "skipped_invalid_label": skipped_invalid_label,
        "skipped_short_trial": skipped_short_trial,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split SEED7 clean EEG for fine-tuning")
    parser.add_argument("--clean-dir", default=CLEAN_DIR)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--xlsx", default=EMOTION_XLSX)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--sample-len", type=int, default=SAMPLE_LEN)
    parser.add_argument("--stride", type=int, default=SAMPLE_STRIDE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = split_seed7_clean_for_finetune(
        clean_dir=args.clean_dir,
        save_dir=args.save_dir,
        xlsx_path=args.xlsx,
        out_dir=args.out_dir,
        sample_len=int(args.sample_len),
        stride=int(args.stride),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "set1_ed_split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
