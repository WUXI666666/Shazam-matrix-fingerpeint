from genericpath import isfile
from typing import Tuple
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle
from fingerprint import createfingerprint, createhashes
import glob
from record import recordaudio
# 指纹提取参数
dist_freq = 11
dist_time = 7
tol_freq = 1
tol_time = 1

# 指纹提取功能
def compute_constellation_map(Y, dist_freq=7, dist_time=7, thresh=0.01):
    result = ndimage.maximum_filter(Y, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
    Cmap = np.logical_and(Y == result, result > thresh)
    return Cmap

def plot_constellation_map(Cmap, Y=None, xlim=None, ylim=None, title='',
                           xlabel='Time (sample)', ylabel='Frequency (bins)',
                           s=5, color='r', marker='o', figsize=(7, 3), dpi=72):
    if Cmap.ndim > 1:
        (K, N) = Cmap.shape
    else:
        K = Cmap.shape[0]
        N = 1
    if Y is None:
        Y = np.zeros((K, N))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    im = ax.imshow(Y, origin='lower', aspect='auto', cmap='gray_r', interpolation='nearest')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    Fs = 1
    if xlim is None:
        xlim = [-0.5/Fs, (N-0.5)/Fs]
    if ylim is None:
        ylim = [-0.5/Fs, (K-0.5)/Fs]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    n, k = np.argwhere(Cmap == 1).T
    ax.scatter(k, n, color=color, s=s, marker=marker)
    plt.tight_layout()
    return fig, ax, im

def compute_spectrogram(audio_path, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    x, Fs = librosa.load(audio_path, sr=Fs)
    x_duration = len(x) / Fs
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
    if bin_max is None:
        bin_max = X.shape[0]
    if frame_max is None:
        frame_max = X.shape[0]
    Y = np.abs(X[:bin_max, :frame_max])
    return Y

def match_binary_matrices_tol(C_ref, C_est, tol_freq=0, tol_time=0):
    assert C_ref.shape == C_est.shape, "Dimensions need to agree"
    N = np.sum(C_ref)
    M = np.sum(C_est)
    C_est_max = ndimage.maximum_filter(C_est, size=(2*tol_freq+1, 2*tol_time+1), mode='constant')
    C_AND = np.logical_and(C_est_max, C_ref)
    TP = np.sum(C_AND)
    FN = N - TP
    FP = M - TP
    return TP, FN, FP, C_AND

def compute_matching_function(C_D, C_Q, tol_freq=1, tol_time=1):
    L = C_D.shape[1]
    N = C_Q.shape[1]
    M = L - N
    assert M >= 0, "Query must be shorter than document"
    Delta = np.zeros(L)
    for m in range(M + 1):
        C_D_crop = C_D[:, m:m+N]
        TP, FN, FP, C_AND = match_binary_matrices_tol(C_D_crop, C_Q, tol_freq=tol_freq, tol_time=tol_time)
        Delta[m] = TP
    shift_max = np.argmax(Delta)
    return Delta, shift_max



# 加载数据的函数
def load_database():
    with open('server_tables.pkl', 'rb') as f:#with用于上下文管理，确保文件在使用完毕后自动关闭。
        server_tables = pickle.load(f)#反序列化
    with open('offset_times.pkl', 'rb') as f:
        offset_times = pickle.load(f)
    with open('song_names.pkl', 'rb') as f:
        song_names = pickle.load(f)
    return server_tables, offset_times, song_names

# 比较两个指纹表
def comparetables(hash_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    server_tables, offset_times, _ = load_database()
    matched_pairs_list = np.zeros((len(server_tables), 1))
    hist_list = np.zeros((len(server_tables), 1))

    for h in range(len(server_tables)):
        matched_pairs = 0
        add_to_hist = []

        for song_hash in hash_matrix:
            freq_anchor, freq_other, delta_time, time_anchor, _ = song_hash
            matches = server_tables[h][(server_tables[h][:, 0] == freq_anchor) & 
                                       (server_tables[h][:, 1] == freq_other) &
                                       (server_tables[h][:, 2] == delta_time)]
            matched_pairs += len(matches)
            add_to_hist.extend(matches[:, 3])

        matched_pairs_list[h] = matched_pairs
        if add_to_hist:
            maxh = np.max(add_to_hist)
            hist, edges = np.histogram(add_to_hist, bins=range(0, (maxh + (maxh + 1) % 2050), 2050))
            hist_list[h] = np.max(hist)

    return matched_pairs_list, hist_list

def fetchID(hash_matrix: np.ndarray) :
    matched_pairs_list, hist_list = comparetables(hash_matrix)
    num_pairs = hash_matrix.shape[0]

    if num_pairs == 0 or len(matched_pairs_list) == 0:
        return [("Not found!", 0, 0)]
    
    ratios = matched_pairs_list / num_pairs
    if len(ratios) == 0 or np.all(ratios == 0):
        return [("Not found!", 0, 0)]
    
    # 获取前三个匹配度最高的歌曲信息
    top_matches = sorted(
        [(i, ratios[i][0], hist_list[i][0]) for i in range(len(ratios))],
        key=lambda x: (-x[1], -x[2])
    )[:3]

    _, _, song_names = load_database()
    top_match_info = [(song_names[i], ratio, hist) for i, ratio, hist in top_matches]

    return top_match_info

def recognize_song_from_path(query_path: str) -> None:
    x, fs = librosa.load(query_path, sr=8192, mono=True)
    F_print = createfingerprint(x)
    song_id = 0  # 对于查询，我们使用一个临时的song_id，例如0
    hash_matrix = createhashes(F_print, song_id=song_id)
    top_matches = fetchID(hash_matrix)

    print('Top 3 Matches:')
    print('Song Name\t\tRatio\t\tHist max')
    for match in top_matches:
        song_name, ratio, hist = match
        print(f"{song_name}\t\t{ratio:.4f}\t\t{hist:.4f}")
    # 输出最匹配的歌曲
    best_match = top_matches[0][0]
    print(f"\nMost matched song: {best_match}")

def recognize_song(x: np.ndarray) -> None:
    F_print = createfingerprint(x)
    song_id = 0  # 对于查询，我们使用一个临时的song_id，例如0
    hash_matrix = createhashes(F_print, song_id=song_id)
    top_matches = fetchID(hash_matrix)

    print('Top 3 Matches:')
    print('Song Name\t\tRatio\t\tHist max')
    for match in top_matches:
        song_name, ratio, hist = match
        print(f"{song_name}\t\t{ratio:.4f}\t\t{hist:.4f}")
    # 输出最匹配的歌曲
    best_match = top_matches[0][0]
    print(f"\nMost matched song: {best_match}")
# 比较两个音频文件
def compare_2songs(path1, path2):
    Y1 = compute_spectrogram(path1)
    Y2 = compute_spectrogram(path2)
    CM1 = compute_constellation_map(Y1, dist_freq, dist_time)
    CM2 = compute_constellation_map(Y2, dist_freq, dist_time)
    Delta, shift_max = compute_matching_function(CM1, CM2, tol_freq=tol_freq, tol_time=tol_time)
    print(Delta[shift_max])
    plot_constellation_map(CM1, np.log(1 + 1 * Y1), color='r', s=30, title=path1)
    plot_constellation_map(CM2, np.log(1 + 1 * Y2), color='r', s=30, title=path2)

# 比较目录中的音频文件
def compare_dir(path, fn_query):
    Y_q = compute_spectrogram(fn_query)
    CMP_q = compute_constellation_map(Y_q, dist_freq, dist_time)
    for fn in os.listdir(path):
        if os.path.isfile(os.path.join(path, fn)):
            if fn.endswith(".wav"):
                fn = os.path.join(path, fn)
                print(fn)
                Y_d = compute_spectrogram(fn)
                CMP_d = compute_constellation_map(Y_d, dist_freq, dist_time)
                Delta, shift_max = compute_matching_function(CMP_d, CMP_q, tol_freq=0, tol_time=0)
                print(Delta[shift_max])
                plot_constellation_map(CMP_d, np.log(1 + 1 * Y_d), color='r', s=30, title=fn)

# 创建数据库
# create_database()

# 比较两个音频文件
# compare_2songs("./songs/NationalAnthemIndia.wav", "./tests/test_3.wav")

# 听歌识曲
song_name = recognize_song_from_path("./tests/test_3.wav")
# song_name=recognize_song (recordaudio())


plt.show()
