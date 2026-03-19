import os

import numpy as np
import scipy.io as sio
import matplotlib

# 兼容旧版 Matplotlib 和无图形环境
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def load_mat_data(file_path):
    mat = sio.loadmat(file_path)
    if "data" not in mat:
        raise ValueError(f"文件缺少 data 字段: {file_path}")

    data = np.asarray(mat["data"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data 必须是二维矩阵: {file_path}")
    if data.shape[0] > data.shape[1]:
        data = data.T

    if data.shape[0] != 62:
        raise ValueError(f"当前脚本要求 62 通道，实际为 {data.shape[0]} 通道")

    sfreq = 200.0
    if "sfreq" in mat:
        value = float(np.asarray(mat["sfreq"]).squeeze())
        if value > 0:
            sfreq = value

    return data, sfreq


def plot_62ch_signal(data, sfreq, output_png, seconds=30.0):
    n_channels, n_samples = data.shape
    max_samples = int(seconds * sfreq)
    max_samples = min(max_samples, n_samples)

    seg = data[:, :max_samples]
    times = np.arange(max_samples) / sfreq

    # 采用稳健间距，避免通道互相重叠。
    per_ch_amp = np.percentile(np.abs(seg), 95, axis=1)
    spacing = max(float(np.median(per_ch_amp) * 4.0), 1.0)

    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
    for ch_idx in range(n_channels):
        y = seg[ch_idx] + ch_idx * spacing
        ax.plot(times, y, linewidth=0.5, color="#1f77b4")

    yticks = [i * spacing for i in range(n_channels)]
    ylabels = [str(i + 1) for i in range(n_channels)]

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel Index")
    ax.set_title(f"EEG 62-channel signal (first {times[-1]:.1f}s)")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    out_dir = os.path.dirname(output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    path = "data/ica/SEED-VII-1-1.mat"
    out_png = "data/ica/read_62ch_signal.png"

    data, sfreq = load_mat_data(path)
    print("data shape:", data.shape)
    print("sfreq:", sfreq)

    plot_62ch_signal(data, sfreq, out_png, seconds=30.0)
    print("已保存 62 通道信号图:", out_png)