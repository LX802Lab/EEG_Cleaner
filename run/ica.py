import ast
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import mne
import numpy as np
import scipy.io as sio
from mne.preprocessing import ICA

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


def _parse_electrode_list(text):
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass

    return [x.strip() for x in text.split(",") if x.strip()]


def load_electrodes_from_env():
    raw = os.getenv("ELECTRODES")
    if not raw:
        raise ValueError("未找到环境变量 ELECTRODES")

    electrodes = _parse_electrode_list(raw)
    if len(electrodes) != 67:
        raise ValueError(f"ELECTRODES 长度必须为 67，当前为 {len(electrodes)}")
    return electrodes


def load_data_and_basename(file_path, default_sfreq=None):
    mat = sio.loadmat(file_path)
    if "data" not in mat:
        raise ValueError(f"文件缺少 data 字段: {file_path}")

    data = np.asarray(mat["data"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data 必须是二维矩阵: {file_path}")

    if data.shape[0] > data.shape[1]:
        data = data.T

    sfreq = None
    if "sfreq" in mat:
        sfreq = float(np.asarray(mat["sfreq"]).squeeze())
    if sfreq is None or sfreq <= 0:
        sfreq = float(default_sfreq or Config.SAMPLING_RATE)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return data, sfreq, base_name


def _find_channel_name(ch_names, candidates, required=True):
    upper_map = {name.upper(): name for name in ch_names}
    for candidate in candidates:
        key = str(candidate).upper()
        if key in upper_map:
            return upper_map[key]
    if required:
        raise ValueError(f"未找到通道: {candidates}")
    return None


def run_ica_if_needed(data, sfreq, n_components=20, method="infomax"):
    n_channels = data.shape[0]
    if n_channels == 62:
        return data

    if n_channels not in {66, 67}:
        raise ValueError(f"仅支持 62/66/67 通道输入，当前: {n_channels}")

    electrodes = load_electrodes_from_env()
    ch_names = electrodes[:n_channels]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    m1_name = _find_channel_name(raw.ch_names, ["M1"])
    m2_name = _find_channel_name(raw.ch_names, ["M2"])
    raw_ref, _ = mne.set_eeg_reference(raw, ref_channels=[m1_name, m2_name], copy=True, verbose="ERROR")

    heo_name = _find_channel_name(raw_ref.ch_names, ["HEO", "HEOG", "EOGH"])
    veo_name = _find_channel_name(raw_ref.ch_names, ["VEO", "VEOG", "EOGV"])
    ch_type_map = {heo_name: "eog", veo_name: "eog"}

    ecg_name = None
    if n_channels == 67:
        ecg_name = _find_channel_name(raw_ref.ch_names, ["ECG"], required=False)
        if ecg_name is not None:
            ch_type_map[ecg_name] = "ecg"

    raw_ref.set_channel_types(ch_type_map, verbose="ERROR")

    raw_for_ica = raw_ref.copy().filter(l_freq=0.5, h_freq=55.0, verbose="ERROR")
    ica = ICA(n_components=n_components, method=method, random_state=97, max_iter="auto")
    ica.fit(raw_for_ica, picks="eeg")

    excluded = set(ica.exclude)
    try:
        eog_idx, _ = ica.find_bads_eog(raw_for_ica, ch_name=[heo_name, veo_name])
        excluded.update(eog_idx)
    except Exception:
        pass

    if ecg_name is not None:
        try:
            ecg_idx, _ = ica.find_bads_ecg(raw_for_ica, ch_name=ecg_name)
            excluded.update(ecg_idx)
        except Exception:
            pass

    ica.exclude = sorted(excluded)
    raw_clean = ica.apply(raw_ref.copy())

    drop_names = [m1_name, m2_name, heo_name, veo_name]
    if n_channels == 67 and ecg_name is not None:
        drop_names.append(ecg_name)
    raw_clean.drop_channels(drop_names)

    out = raw_clean.get_data()
    if out.shape[0] != 62:
        raise ValueError(f"预处理后通道数不是 62，而是 {out.shape[0]}")

    return out


def save_ica_result(data_matrix, sfreq, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{base_name}.mat")
    sio.savemat(save_path, {"data": data_matrix, "sfreq": float(sfreq)})
    return save_path


def process_one_file(
    file_path,
    ica_output_dir=Config.ICA_DATA_PATH,
    n_components=20,
    method="infomax",
):
    data, sfreq, base_name = load_data_and_basename(file_path, default_sfreq=Config.SAMPLING_RATE)
    data62 = run_ica_if_needed(data, sfreq, n_components=n_components, method=method)
    save_path = save_ica_result(data62, sfreq, output_dir=ica_output_dir, base_name=base_name)
    return {"ica": save_path}


def _process_one_file_task(args):
    file_path, ica_output_dir, n_components, method = args
    result = process_one_file(file_path=file_path, ica_output_dir=ica_output_dir, n_components=n_components, method=method)
    result["file_path"] = file_path
    return result


def process_path(
    input_path,
    ica_output_dir=Config.ICA_DATA_PATH,
    n_components=20,
    method="infomax",
    max_workers=None,
):
    if os.path.isfile(input_path):
        return [process_one_file(input_path, ica_output_dir, n_components, method)]

    if not os.path.isdir(input_path):
        raise ValueError(f"输入路径不存在: {input_path}")

    # 启动时仅做一次环境变量检查，确保可加载完整 67 电极列表。
    load_electrodes_from_env()

    files = sorted(glob.glob(os.path.join(input_path, "*.mat")))
    if not files:
        raise ValueError(f"目录中未找到 mat 文件: {input_path}")

    from tqdm import tqdm

    max_workers = _default_workers() if max_workers is None else max(1, int(max_workers))
    save_paths = []

    if max_workers == 1:
        for file_path in tqdm(files, desc="ICA 处理", unit="file"):
            try:
                result = process_one_file(file_path, ica_output_dir, n_components, method)
            except Exception as e:
                result = {"file_path": file_path, "error": str(e)}
            result["file_path"] = file_path
            if "error" in result:
                _log(f"❌ ICA 处理失败: {result['file_path']}, 错误: {result['error']}")
            save_paths.append(result)
        return save_paths

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_one_file_task,
                (file_path, ica_output_dir, n_components, method),
            ): file_path
            for file_path in files
        }
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="ICA 处理", unit="file"):
                file_path = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"file_path": file_path, "error": str(e)}

                if "error" in result:
                    _log(f"❌ ICA 处理失败: {result['file_path']}, 错误: {result['error']}")
                save_paths.append(result)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    return save_paths


if __name__ == "__main__":
    workers = int(os.getenv("ICA_MAX_WORKERS", _default_workers()))
    try:
        process_path(Config.CLEANED_DATA_PATH, ica_output_dir=Config.ICA_DATA_PATH, max_workers=workers)
    except KeyboardInterrupt:
        _log("⚠️ ICA 任务被用户中断")
    except Exception as e:
        _log(f"❌ ICA 批处理失败: {str(e)}")
