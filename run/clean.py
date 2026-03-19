import glob
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import mne
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

from config import Config


def _log(message):
    try:
        from tqdm import tqdm as tqdm_class

        tqdm_class.write(message)
    except Exception:
        print(message)


def _env_flag(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _default_workers():
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu // 2 if cpu > 1 else 1))


def _extract_sfreq_from_mat(mat, default_sfreq=None):
    for key in ("sfreq", "fs", "sampling_rate", "srate"):
        if key in mat:
            value = float(np.asarray(mat[key]).squeeze())
            if value > 0:
                return value
    return float(default_sfreq or Config.SAMPLING_RATE)


def _to_channels_first_2d(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("输入数据必须是二维矩阵")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


def _filter_eeg_array(data, sfreq=Config.SAMPLING_RATE):
    """Apply bandpass 0.1-75 Hz and notch 50 Hz to shape (n_channels, n_samples)."""
    sos_bp = butter(4, [0.1, 75.0], btype="bandpass", fs=sfreq, output="sos")
    data = sosfiltfilt(sos_bp, data, axis=1)
    sos_notch = butter(4, [49.0, 51.0], btype="bandstop", fs=sfreq, output="sos")
    data = sosfiltfilt(sos_notch, data, axis=1)
    return data


def _extract_standard_62(data_matrix, channel_order):
    if data_matrix.shape[0] != len(channel_order):
        raise ValueError("data_matrix 通道数与 channel_order 长度不一致")

    index_map = {ch.upper(): idx for idx, ch in enumerate(channel_order)}
    reordered_indices = []
    missing_channels = []
    for ch in Config.STANDARD_62_ELECTRODES:
        idx = index_map.get(ch.upper())
        if idx is None:
            missing_channels.append(ch)
        else:
            reordered_indices.append(idx)

    if len(reordered_indices) < 62:
        raise ValueError(
            f"无法提取完整的 62 个脑电极。找到 {len(reordered_indices)} 个, 缺失: {missing_channels}"
        )

    return data_matrix[reordered_indices, :]


def save_data_mat(data_matrix, sfreq, output_dir, output_name):
    os.makedirs(output_dir, exist_ok=True)
    if not output_name.lower().endswith(".mat"):
        output_name = f"{output_name}.mat"
    save_path = os.path.join(output_dir, output_name)
    sio.savemat(save_path, {"data": data_matrix, "sfreq": float(sfreq)})
    return save_path


def _find_primary_2d_key(mat):
    preferred = ("data", "eeg", "X", "signals", "cnt", "EEG")
    for key in preferred:
        if key in mat:
            value = np.asarray(mat[key])
            if value.ndim == 2:
                return key

    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 2 and min(arr.shape) >= 10:
            return key

    return None


def _load_cnt_raw(file_path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\s*Could not parse meas date from the header\. Setting to None\.",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_cnt(
            file_path,
            preload=True,
            data_format="int32",
            verbose="ERROR",
        )
    return raw


def process_cnt_file(file_path, output_dir=Config.CLEANED_DATA_PATH):
    raw = _load_cnt_raw(file_path)

    sfreq_target = float(Config.SAMPLING_RATE)
    raw.resample(sfreq=sfreq_target)
    raw.filter(l_freq=0.1, h_freq=75.0, verbose="ERROR")
    raw.notch_filter(freqs=[50.0], verbose="ERROR")

    # CNT 数据单位固定按 V 处理，统一转换为 uV。
    data_matrix = raw.get_data() * 1e6

    data_matrix = _extract_standard_62(data_matrix, raw.ch_names.copy())
    output_name = name_from_path(file_path)
    save_path = save_data_mat(data_matrix, sfreq_target, output_dir, output_name)
    return save_path


def _suffix_from_key(key):
    m = re.search(r"eeg(\d+)$", str(key).lower())
    if m:
        return m.group(1)
    return str(key)


def _normalized_parts(file_path):
    path = os.path.normpath(file_path)
    return [p.lower() for p in path.split(os.sep)]


def _is_seed4_path(file_path):
    parts = _normalized_parts(file_path)
    return any(p in {"seed4", "seed_iv", "seed-iv"} for p in parts)


def _is_seed5_path(file_path):
    parts = _normalized_parts(file_path)
    return any(p in {"seed5", "seed-v", "seed_5", "seed-5"} for p in parts)


def _is_seed7_path(file_path):
    parts = _normalized_parts(file_path)
    return any(p in {"seed7", "seed-vii", "seed_7", "seed-7"} for p in parts)


def _parse_stem_tokens(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    return [t for t in stem.split("_") if t]


def parse_name_info(file_path):
    """
    命名规则:
    1) seed4: SEED-IV-{person}-{exp}
    2) seed : SEED-I-{person}-{exp}
    3) seed5: SEED-V-{person}-{exp} (第二段)
    4) seed7: SEED-VII-{person}-{exp} (第三段)
    """
    tokens = _parse_stem_tokens(file_path)

    if _is_seed4_path(file_path):
        if len(tokens) < 1:
            raise ValueError(f"无法解析 seed4 文件名: {file_path}")
        num_person = tokens[0]
        num_exp = os.path.basename(os.path.dirname(file_path))
        return "IV", num_person, num_exp

    if _is_seed7_path(file_path):
        if len(tokens) < 3:
            raise ValueError(f"无法解析 seed7 文件名: {file_path}")
        num_person = tokens[0]
        num_exp = tokens[2]
        return "VII", num_person, num_exp

    if _is_seed5_path(file_path):
        if len(tokens) < 2:
            raise ValueError(f"无法解析 seed5 文件名: {file_path}")
        num_person = tokens[0]
        num_exp = tokens[1]
        return "V", num_person, num_exp

    if "seed" in file_path.lower():
        if len(tokens) < 2:
            raise ValueError(f"无法解析 seed 文件名: {file_path}")
        num_person = tokens[0]
        num_exp = tokens[1]
        return "I", num_person, num_exp

    raise ValueError(f"无法从路径识别数据集类型: {file_path}")


def base_name_from_path(file_path):
    version, num_person, num_exp = parse_name_info(file_path)
    return f"SEED-{version}-{num_person}-{num_exp}"


def process_mat_file(file_path, output_dir=Config.CLEANED_DATA_PATH):
    mat = sio.loadmat(file_path)
    sfreq = _extract_sfreq_from_mat(mat, default_sfreq=Config.SAMPLING_RATE)
    base_name = base_name_from_path(file_path)

    if _is_seed4_path(file_path):
        save_paths = []
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.ndim != 2:
                continue

            m = re.search(r"eeg(\d+)$", key, flags=re.IGNORECASE)
            if m is None:
                continue

            eeg_num = m.group(1)
            data = _to_channels_first_2d(arr)
            data = _filter_eeg_array(data, sfreq=sfreq)
            out_name = f"{base_name}-{eeg_num}.mat"
            save_paths.append(save_data_mat(data, sfreq, output_dir, out_name))

        if not save_paths:
            raise ValueError(f"seed4 文件未找到形如 *_eeg{{num}} 的二维数据键: {file_path}")
        return save_paths

    primary_key = _find_primary_2d_key(mat)
    if primary_key == "data":
        data = _to_channels_first_2d(mat["data"])
        data = _filter_eeg_array(data, sfreq=sfreq)
        save_path = save_data_mat(data, sfreq, output_dir, f"{base_name}.mat")
        return save_path

    save_paths = []
    if primary_key is not None:
        data = _to_channels_first_2d(mat[primary_key])
        data = _filter_eeg_array(data, sfreq=sfreq)
        save_paths.append(save_data_mat(data, sfreq, output_dir, f"{base_name}.mat"))
        return save_paths[0]

    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim != 2:
            continue

        data = _to_channels_first_2d(arr)
        data = _filter_eeg_array(data, sfreq=sfreq)

        suffix = _suffix_from_key(key)
        out_name = f"{base_name}-{suffix}.mat"
        save_paths.append(save_data_mat(data, sfreq, output_dir, out_name))

    if not save_paths:
        raise ValueError(f"MAT 文件中未找到可处理的二维数据: {file_path}")

    return save_paths


def name_from_path(file_path):
    try:
        return f"{base_name_from_path(file_path)}.mat"
    except Exception:
        _log(f"无法从路径中识别 SEED 版本，使用原始文件名: {file_path}")
        return os.path.basename(file_path)


def process_one_file(file_path, output_dir=Config.CLEANED_DATA_PATH, enable_log=True):
    if not os.path.exists(file_path):
        error_msg = f"文件不存在: {file_path}"
        if enable_log:
            _log(f"❌ {error_msg}")
        return {"error": error_msg}

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".cnt":
            save_path = process_cnt_file(file_path, output_dir=output_dir)
        elif ext == ".mat":
            save_path = process_mat_file(file_path, output_dir=output_dir)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        if enable_log:
            if isinstance(save_path, list):
                _log(f"✓ 完成: 共保存 {len(save_path)} 个文件")
            else:
                _log(f"✓ 完成: {save_path}")
        return {"save_path": save_path}
    except Exception as e:
        if enable_log:
            _log(f"❌ 处理失败: {file_path}, 错误: {str(e)}")
        return {"error": str(e)}


def _process_one_file_task(args):
    file_path, output_dir = args
    result = process_one_file(file_path, output_dir=output_dir, enable_log=False)
    result["file_path"] = file_path
    return result


def process_files(file_paths, output_dir=Config.CLEANED_DATA_PATH, max_workers=None):
    if not file_paths:
        return []

    max_workers = _default_workers() if max_workers is None else max(1, int(max_workers))
    results = []
    log_success = _env_flag("CLEAN_LOG_SUCCESS", default=False)
    from tqdm import tqdm

    if max_workers == 1:
        for file_path in tqdm(file_paths, desc="清洗处理", unit="file"):
            result = process_one_file(file_path, output_dir=output_dir, enable_log=False)
            result["file_path"] = file_path
            if "error" in result:
                _log(f"❌ 处理失败: {file_path}, 错误: {result['error']}")
            elif log_success:
                save_path = result["save_path"]
                if isinstance(save_path, list):
                    _log(f"✓ 完成: 共保存 {len(save_path)} 个文件")
                else:
                    _log(f"✓ 完成: {save_path}")
            results.append(result)
        return results

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one_file_task, (file_path, output_dir)): file_path
            for file_path in file_paths
        }
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="清洗处理", unit="file"):
                file_path = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"file_path": file_path, "error": str(e)}

                if "error" in result:
                    _log(f"❌ 处理失败: {result['file_path']}, 错误: {result['error']}")
                elif log_success:
                    save_path = result["save_path"]
                    if isinstance(save_path, list):
                        _log(f"✓ 完成: 共保存 {len(save_path)} 个文件")
                    else:
                        _log(f"✓ 完成: {save_path}")
                results.append(result)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    return results


def collect_all_raw_files(input_root):
    cnt_files = glob.glob(os.path.join(input_root, "**", "*.cnt"), recursive=True)
    mat_files = glob.glob(os.path.join(input_root, "seed4", "**", "*.mat"), recursive=True)
    all_files = cnt_files + mat_files

    def _valid(path):
        norm = os.path.normpath(path).lower()
        if f"{os.sep}.venv{os.sep}" in norm:
            return False
        if f"{os.sep}site-packages{os.sep}" in norm:
            return False
        return True

    return sorted([p for p in all_files if _valid(p)])


def default_input_root():
    preferred = os.getenv("RAW_DATA_ROOT", "../../Datasets/EEG/SJTU-SEED")
    if os.path.exists(preferred):
        return preferred

    fallback = "../../Datasets/EEG/SJTU-SEED"
    if os.path.exists(fallback):
        return fallback

    return preferred


if __name__ == "__main__":
    input_root = default_input_root()
    all_files = collect_all_raw_files(input_root)
    workers = int(os.getenv("CLEAN_MAX_WORKERS", _default_workers()))
    _log(f"输入目录: {input_root}")
    _log(f"找到 {len(all_files)} 个待处理文件，使用 {workers} 个进程并行处理")
    try:
        process_files(all_files, output_dir=Config.CLEANED_DATA_PATH, max_workers=workers)
    except KeyboardInterrupt:
        _log("⚠️ 清洗任务被用户中断")
