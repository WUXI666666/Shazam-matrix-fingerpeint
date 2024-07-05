from genericpath import isfile
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle
from fingerprint import createfingerprint, createhashes
import glob
from typing import List

# 保存数据的函数
def save_database(server_tables: List[np.ndarray], offset_times: List[np.ndarray], song_names: List[str]) -> None:
    with open('server_tables.pkl', 'wb') as f:
        pickle.dump(server_tables, f)
    with open('offset_times.pkl', 'wb') as f:
        pickle.dump(offset_times, f)
    with open('song_names.pkl', 'wb') as f:
        pickle.dump(song_names, f)

# 创建数据库
def create_database(directory: str = './songs', extensions: List[str] = ['*.wav']) -> None:
    server_tables = []
    offset_times = []
    song_names = []
    files = []

    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))

    for i, file in enumerate(files):
        song_name = os.path.basename(file)
        song_names.append(song_name)

        x, fs = librosa.load(file, sr=8192, mono=True)
        F_print = createfingerprint(x)
        hash_matrix = createhashes(F_print, song_id=i)
        
        server_tables.append(hash_matrix)
        offset_times.append(hash_matrix[:, 3])  # 使用时间锚点作为偏移时间
        print(f"Processed {i+1}/{len(files)}: {song_name}")

    save_database(server_tables, offset_times, song_names)

# Example usage
create_database()
