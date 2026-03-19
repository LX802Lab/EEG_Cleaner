#!/usr/bin/env python3
"""Split SEED7 DE features into trial-level NPZ files.

Inputs:
- DE files: data/de/SEED-VII-<subject>-<session>.mat
- save info: /mnt/data/Datasets/EEG/SJTU-SEED/seed7/save/*_save_info.csv
- trigger info: /mnt/data/Datasets/EEG/SJTU-SEED/seed7/save/*_trigger_info.csv
- emotion labels: /mnt/data/Datasets/EEG/SJTU-SEED/seed7/emotion_label_and_stimuli_order.xlsx

Output:
- data/emotion-detect/seed7_trial/SEED-VII-<subject>-<session>-<trial>.npz
  with keys: {data, label}

Notes:
- data shape for each NPZ: (len, 62, 5)
- label is the emotion class index from xlsx (session/trial order)
- trials with activation score <= 0.2 are only printed and not saved
"""

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio


# Paths
DE_DIR = "data/de"
SAVE_INFO_DIR = "/mnt/data/Datasets/EEG/SJTU-SEED/seed7/save"
EMOTION_XLSX = "/mnt/data/Datasets/EEG/SJTU-SEED/seed7/emotion_label_and_stimuli_order.xlsx"
OUTPUT_DIR = "data/emotion-detect/seed7_trial"

# DE timing params (must match de.py extraction config)
DE_WINDOW_SEC = 0.5
DE_OVERLAP = 0.5
DE_STEP_SEC = DE_WINDOW_SEC * (1.0 - DE_OVERLAP)  # 0.25s
ACTIVATION_SKIP_THRESHOLD = 0.2


# Emotion mapping used by this project.
EMOTION_TO_LABEL = {
    "Disgust": 0,
    "Fear": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4,
    "Anger": 5,
    "Surprise": 6,
}


def time_to_index(t: float, step: float = DE_STEP_SEC) -> int:
    """Convert time in seconds to DE frame index (0-based)."""
    return int(round(t / step))


def read_csv_try(path: str) -> pd.DataFrame:
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, header=None, sep=sep, engine="python")
            if df.shape[1] >= 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, header=None, engine="python")


def find_trigger_file(subject: str, session: str) -> Optional[str]:
    trig_pattern = os.path.join(SAVE_INFO_DIR, f"{subject}_*_{session}_trigger_info.csv")
    trig_files = sorted(glob.glob(trig_pattern))
    return trig_files[0] if trig_files else None


def find_save_file(subject: str, session: str) -> Optional[str]:
    save_pattern = os.path.join(SAVE_INFO_DIR, f"{subject}_*_{session}_save_info.csv")
    save_files = sorted(glob.glob(save_pattern))
    return save_files[0] if save_files else None


def parse_scores(save_path: str) -> List[float]:
    """Read activation scores from save_info third column."""
    df = read_csv_try(save_path)
    if df.shape[1] < 3:
        return []
    vals = pd.to_numeric(df.iloc[:, 2], errors="coerce").tolist()
    return [float(v) if pd.notna(v) else np.nan for v in vals]


def parse_timestamps(ts_path: str) -> List[Tuple[float, float]]:
    """Parse trial start/end times in seconds (relative to first timestamp)."""
    df = read_csv_try(ts_path)
    if df.shape[1] < 2:
        return []

    col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    col1 = pd.to_datetime(df.iloc[:, 1], errors="coerce")

    if col0.notna().sum() == 0 or col1.notna().sum() == 0:
        return []

    t0 = col1.dropna().iloc[0]
    rel_sec = (col1 - t0).dt.total_seconds()

    starts = rel_sec[col0 == 1].dropna().tolist()
    ends = rel_sec[col0 == 2].dropna().tolist()

    n = min(len(starts), len(ends))
    pairs = []
    for i in range(n):
        s = float(starts[i])
        e = float(ends[i])
        if e > s:
            pairs.append((s, e))
    return pairs


def parse_emotion_order_xlsx(xlsx_path: str) -> Dict[int, List[int]]:
    """Load session->20 labels from xlsx laid out horizontally.

    Expected 8 rows:
    - row 1: Session 1 + 20 emotion names
    - row 3: Session 2 + 20 emotion names
    - row 5: Session 3 + 20 emotion names
    - row 7: Session 4 + 20 emotion names
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Missing label xlsx: {xlsx_path}")

    df = pd.read_excel(xlsx_path, header=None)
    session_map: Dict[int, List[int]] = {}

    for row_idx in [1, 3, 5, 7]:
        if row_idx >= len(df):
            continue
        row = df.iloc[row_idx].tolist()
        # col0 should be "Session k"; labels are in col1..col20
        head = str(row[0]).strip() if row else ""
        m = re.search(r"Session\s*(\d+)", head, flags=re.IGNORECASE)
        if not m:
            continue

        session = int(m.group(1))
        labels: List[int] = []
        for v in row[1:21]:
            emo = str(v).strip()
            if not emo:
                labels.append(-1)
                continue

            label = EMOTION_TO_LABEL.get(emo)
            if label is None:
                # fallback: case-insensitive lookup
                found = None
                low = emo.lower()
                for k, val in EMOTION_TO_LABEL.items():
                    if k.lower() == low:
                        found = val
                        break
                label = found if found is not None else -1
            labels.append(int(label))

        if len(labels) != 20:
            raise ValueError(f"Session {session} labels count != 20")
        session_map[session] = labels

    if len(session_map) != 4:
        raise ValueError(f"XLSX parsing failed, expected 4 sessions, got {len(session_map)}")
    return session_map


def load_de(path: str) -> np.ndarray:
    mat = sio.loadmat(path)
    if "de" not in mat:
        raise ValueError(f"Missing 'de' in {path}")
    de = np.asarray(mat["de"], dtype=np.float32)
    if de.ndim != 3 or de.shape[1:] != (62, 5):
        raise ValueError(f"Unexpected DE shape {de.shape} in {path}")
    return de


def parse_subject_session(de_file: str) -> Tuple[str, str, str]:
    stem = os.path.splitext(os.path.basename(de_file))[0]
    m = re.match(r"SEED-VII-(\d+)-(\d+)$", stem)
    if not m:
        raise ValueError(f"Unrecognized filename: {de_file}")
    return m.group(1), m.group(2), stem


def clear_output_npz(output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    old_files = glob.glob(os.path.join(output_dir, "*.npz"))
    for p in old_files:
        os.remove(p)
    return len(old_files)


def process_one_file(de_file: str, session_labels: Dict[int, List[int]]) -> Tuple[int, int, List[int]]:
    subject, session, stem = parse_subject_session(de_file)
    ts_path = find_trigger_file(subject, session)
    save_path = find_save_file(subject, session)
    if not ts_path or not save_path:
        print(f"[SKIP] Missing save/trigger csv for {stem}")
        return 0, 0, []

    ts_pairs = parse_timestamps(ts_path)
    scores = parse_scores(save_path)
    de = load_de(de_file)
    labels = session_labels.get(int(session), [])

    total_len = de.shape[0]
    n_trial = min(len(labels), len(ts_pairs), len(scores))
    if n_trial == 0:
        print(f"[SKIP] Empty label/timestamp/score for {stem}")
        return 0, 0, []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = 0
    zero_skipped = 0
    trial_lengths = []

    for i in range(n_trial):
        trial_num = i + 1
        label = int(labels[i])
        score = float(scores[i])
        start_sec, end_sec = ts_pairs[i]

        out_name = f"{stem}-{trial_num}.npz"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if score <= ACTIVATION_SKIP_THRESHOLD:
            print(f"[NO] {out_name} | score={score}  <= 0.2 | skipped saving, label={label}")
            zero_skipped += 1
            continue

        sidx = max(0, time_to_index(start_sec))
        eidx = min(total_len, time_to_index(end_sec))
        if eidx <= sidx:
            continue

        trial_data = de[sidx:eidx, :, :]
        trial_lengths.append(int(trial_data.shape[0]))

        np.savez_compressed(
            out_path,
            data=trial_data.astype(np.float32),
            label=np.asarray(label, dtype=np.int32),
        )
        saved += 1

        print(
            f"[OK] {out_name} | trial_data={trial_data.shape} | label={label} | score={score} "
        )
    return saved, zero_skipped, trial_lengths


def main() -> None:
    de_files = sorted(glob.glob(os.path.join(DE_DIR, "SEED-VII-*.mat")))
    if not de_files:
        print(f"No SEED-VII mat files found in {DE_DIR}")
        return

    session_labels = parse_emotion_order_xlsx(EMOTION_XLSX)
    removed = clear_output_npz(OUTPUT_DIR)

    total_saved = 0
    total_zero_skipped = 0
    all_lengths: List[int] = []

    print(f"Found {len(de_files)} SEED-VII files in {DE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"DE step seconds: {DE_STEP_SEC}")
    print(f"Loaded labels from: {EMOTION_XLSX}")
    print(f"Session labels shape: 4 x 20")
    print(f"Removed old npz files: {removed}")

    for de_file in de_files:
        try:
            saved, zero_skipped, lengths = process_one_file(de_file, session_labels)
            total_saved += saved
            total_zero_skipped += zero_skipped
            all_lengths.extend(lengths)
        except Exception as e:
            print(f"[ERR] {de_file}: {e}")

    if all_lengths:
        print("\n=== Summary ===")
        print(f"Total source files: {len(de_files)}")
        print(f"Total saved trial npz: {total_saved}")
        print(f"Total score<=0.2 skipped: {total_zero_skipped}")
        print(f"Trial len min/max/mean: {min(all_lengths)}/{max(all_lengths)}/{np.mean(all_lengths):.2f}")
        print("Each npz keys: {data, label}")
        print("data shape: (len, 62, 5), label: int32 emotion class (from xlsx)")
    else:
        print("No trial npz generated.")


if __name__ == "__main__":
    main()
