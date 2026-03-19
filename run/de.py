import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def _default_workers():
    cpu = os.cpu_count() or 2
    return max(1, min(4, cpu // 2 if cpu > 1 else 1))


def load_data_and_meta(file_path, default_sfreq=None):
    mat = sio.loadmat(file_path)
    if "data" not in mat:
        raise ValueError(f"文件缺少 data 字段: {file_path}")

    data = np.asarray(mat["data"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data 必须是二维矩阵: {file_path}")

    if data.shape[0] > data.shape[1]:
        data = data.T

    if data.shape[0] != 62:
        raise ValueError(f"DE 提取要求 62 通道输入，当前为 {data.shape[0]}: {file_path}")

    sfreq = None
    if "sfreq" in mat:
        sfreq = float(np.asarray(mat["sfreq"]).squeeze())
    if sfreq is None or sfreq <= 0:
        sfreq = float(default_sfreq or Config.SAMPLING_RATE)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    person_id = parse_person_id(base_name)
    return data, sfreq, base_name, person_id


def parse_person_id(base_name):
    parts = re.split(r"[-_]", base_name)
    for token in parts:
        if token.isdigit():
            return token
    raise ValueError(f"无法从文件名解析被试编号: {base_name}")


def extract_de_features(data, sfreq, window_sec=1.0, overlap=0.5):
    n_channels, n_samples = data.shape
    if n_channels != 62:
        raise ValueError(f"DE 提取要求 62 通道，当前为 {n_channels}")

    sfreq = float(sfreq)
    win_samples = int(round(window_sec * sfreq))
    if win_samples <= 0:
        raise ValueError("window_sec 对应样本点数必须大于 0")
    if not (0 <= overlap < 1):
        raise ValueError("overlap 必须满足 0 <= overlap < 1")

    step_samples = int(round(win_samples * (1.0 - overlap)))
    if step_samples <= 0:
        raise ValueError("overlap 过大导致步长为 0")

    if n_samples < win_samples:
        raise ValueError("数据长度不足以形成至少 1 个窗口")

    length = 1 + (n_samples - win_samples) // step_samples

    bands = [
        ("delta", 0.5, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma", 30.0, 50.0),
    ]

    de_features = np.zeros((length, n_channels, len(bands)), dtype=np.float32)

    for band_idx, (_, low, high) in enumerate(bands):
        sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
        band_data = sosfiltfilt(sos, data, axis=1)

        segments = np.zeros((n_channels, length, win_samples), dtype=np.float64)
        for i in range(length):
            start = i * step_samples
            end = start + win_samples
            segments[:, i, :] = band_data[:, start:end]

        var = np.var(segments, axis=2)
        var = np.maximum(var, 1e-12)
        de = 0.5 * np.log(2.0 * np.pi * np.e * var)
        de_features[:, :, band_idx] = de.T.astype(np.float32)

    return de_features, [b[0] for b in bands]


def save_de_result(de_features, band_names, output_dir, base_name, sfreq):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{base_name}.mat")
    sio.savemat(
        save_path,
        {
            "de": de_features,
            "band_names": np.asarray(band_names, dtype=object),
            "sfreq": float(sfreq),
        },
    )
    return save_path


def build_person_channel_stats(file_paths):
    stats = defaultdict(lambda: {"sum": np.zeros(62), "sumsq": np.zeros(62), "count": 0})

    for file_path in file_paths:
        data, _, _, person_id = load_data_and_meta(file_path, default_sfreq=Config.SAMPLING_RATE)
        stats[person_id]["sum"] += np.sum(data, axis=1)
        stats[person_id]["sumsq"] += np.sum(np.square(data), axis=1)
        stats[person_id]["count"] += int(data.shape[1])

    person_norm = {}
    for person_id, s in stats.items():
        if s["count"] <= 0:
            raise ValueError(f"被试 {person_id} 的样本数为 0")
        mean = s["sum"] / s["count"]
        var = s["sumsq"] / s["count"] - np.square(mean)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)
        person_norm[person_id] = (mean, std)

    return person_norm


def process_one_file(
    file_path,
    person_norm,
    de_output_dir=Config.DE_DATA_PATH,
    window_sec=1.0,
    overlap=0.5,
    enable_log=True,
):
    data, sfreq, base_name, person_id = load_data_and_meta(file_path, default_sfreq=Config.SAMPLING_RATE)
    if person_id not in person_norm:
        raise ValueError(f"被试 {person_id} 缺少标准化统计量")

    mean, std = person_norm[person_id]
    data_norm = (data - mean[:, None]) / std[:, None]

    de_features, band_names = extract_de_features(data_norm, sfreq, window_sec=window_sec, overlap=overlap)
    de_save_path = save_de_result(de_features, band_names, output_dir=de_output_dir, base_name=base_name, sfreq=sfreq)

    if enable_log:
        _log(f"完成: {file_path}")
        _log(f"DE 保存: {de_save_path}, 形状: {de_features.shape}")
    return {"de": de_save_path}


def _process_one_de_task(args):
    file_path, person_norm, de_output_dir, window_sec, overlap = args
    result = process_one_file(
        file_path=file_path,
        person_norm=person_norm,
        de_output_dir=de_output_dir,
        window_sec=window_sec,
        overlap=overlap,
        enable_log=False,
    )
    result["file_path"] = file_path
    return result


def process_path(
    input_path,
    de_output_dir=Config.DE_DATA_PATH,
    window_sec=0.5,
    overlap=0.5,
    max_workers=None,
):
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        if not os.path.isdir(input_path):
            raise ValueError(f"输入路径不存在: {input_path}")
        files = sorted(glob.glob(os.path.join(input_path, "*.mat")))

    if not files:
        raise ValueError(f"未找到 mat 文件: {input_path}")

    _log("开始统计每个被试的通道 mean/std")
    person_norm = build_person_channel_stats(files)
    _log(f"共 {len(person_norm)} 位被试")

    from tqdm import tqdm

    max_workers = _default_workers() if max_workers is None else max(1, int(max_workers))
    save_paths = []

    if max_workers == 1:
        for file_path in tqdm(files, desc="DE 处理", unit="file"):
            result = process_one_file(
                file_path,
                person_norm,
                de_output_dir,
                window_sec,
                overlap,
                enable_log=False,
            )
            result["file_path"] = file_path
            _log(f"完成: {file_path}")
            _log(f"DE 保存: {result['de']}")
            save_paths.append(result)
        return save_paths

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_one_de_task,
                (file_path, person_norm, de_output_dir, window_sec, overlap),
            ): file_path
            for file_path in files
        }
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="DE 处理", unit="file"):
                file_path = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"file_path": file_path, "error": str(e)}

                if "error" in result:
                    _log(f"❌ DE 处理失败: {result['file_path']}, 错误: {result['error']}")
                else:
                    _log(f"完成: {result['file_path']}")
                    _log(f"DE 保存: {result['de']}")
                save_paths.append(result)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    return save_paths


if __name__ == "__main__":
    workers = int(os.getenv("DE_MAX_WORKERS", _default_workers()))
    _log(f"使用 {workers} 个进程并行提取 DE 特征")
    try:
        process_path(Config.ICA_DATA_PATH, de_output_dir=Config.DE_DATA_PATH, max_workers=8)
    except KeyboardInterrupt:
        _log("⚠️ DE 任务被用户中断")
    except Exception as e:
        _log(f"❌ DE 批处理失败: {str(e)}")
