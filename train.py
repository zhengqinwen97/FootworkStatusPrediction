import json
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def parse_file(path):
    features = []
    labels = []

    parse_data = {}

    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data = obj.get("Numbers", [])
                foot_vec = -1
                foot_name = obj.get("Foot_Name", None)
                if 'Right' in foot_name:
                    foot_vec = 0
                elif 'Left' in foot_name:
                    foot_vec = 1

                time = obj.get("Save_Time_Now", None).replace(':', '_').replace('.', '_')

                if time not in parse_data:
                    parse_data[time] = {}
                parse_data[time][foot_vec] = data

            except json.JSONDecodeError:
                continue

    return parse_data

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            print(f"processing {filename}")

            # 可选：跳过非 .json 文件
            if not filename.endswith('.json'):
                continue

            try:
                names = filename.split("__")

                left_area = names[0]
                right_area = names[1]
                motion_state = names[2]
                travel_state = names[3]
                trips_count = names[4]

                left_area = int(left_area.split('_')[-1])
                right_area = int(right_area.split('_')[-1])
                trips_count = int(trips_count.split('.')[0])

                file_path = os.path.join(root, filename)
                parse_data = parse_file(file_path)

                yield {
                    'left_area_label': left_area,
                    'right_area_label': right_area,
                    'motion_state_label': motion_state,
                    'travel_state_label': travel_state,
                    'trips_count_label': trips_count,
                    'data': parse_data,
                }
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue  # 或者 raise，取决于你是否希望中断

def extract_sample_features(sample):
    def _extract_window_features(channel_list, window_size=500, stride=200):
        arr = np.array(channel_list)   # shape (T, 246)
        T, C = arr.shape

        samples = []

        for start in range(0, T - window_size + 1, stride):
            window = arr[start:start + window_size]   # (200, 246)
            feature_dict = {}

            # 针对每个通道提取特征
            for ch in range(C):
                x = window[:, ch]
                dx = np.diff(x)

                # --- 基础统计特征 ---
                feature_dict[f'ch{ch}_mean']  = float(np.mean(x))
                feature_dict[f'ch{ch}_std']   = float(np.std(x))
                feature_dict[f'ch{ch}_median'] = float(np.median(x))
                feature_dict[f'ch{ch}_min']   = float(np.min(x))
                feature_dict[f'ch{ch}_max']   = float(np.max(x))
                # feature_dict[f'ch{ch}_range'] = float(np.max(x) - np.min(x))
                # feature_dict[f'ch{ch}_p25']   = float(np.percentile(x, 25))
                # feature_dict[f'ch{ch}_p75']   = float(np.percentile(x, 75))

                # # --- 能量特征 ---
                # feature_dict[f'ch{ch}_energy'] = float(np.mean(x ** 2))
                # feature_dict[f'ch{ch}_rms']    = float(np.sqrt(np.mean(x ** 2)))

                # # --- 变化特征 ---
                # feature_dict[f'ch{ch}_mad']    = float(np.mean(np.abs(dx)))
                # feature_dict[f'ch{ch}_maxdiff'] = float(np.max(np.abs(dx)))
                # feature_dict[f'ch{ch}_slope_mean'] = float(np.mean(dx))

                # # 过零率（零点变号次数）
                # feature_dict[f'ch{ch}_zcr'] = float(np.sum(np.sign(x[1:]) != np.sign(x[:-1])))

                # # --- 频域特征 ---
                # fft_vals = np.abs(np.fft.rfft(x))
                # freqs = np.fft.rfftfreq(len(x))

                # # 主频与主频幅
                # idx = np.argmax(fft_vals)
                # feature_dict[f'ch{ch}_dom_freq'] = float(freqs[idx])
                # feature_dict[f'ch{ch}_dom_amp']  = float(fft_vals[idx])

                # # 谱熵
                # psd = fft_vals ** 2
                # psd_norm = psd / (np.sum(psd) + 1e-8)
                # feature_dict[f'ch{ch}_spec_entropy'] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-8)))

            samples.append(feature_dict)

        return samples
    
    timestamps = list(sample['data'].keys())
    channel_list = []
    for value in sample['data'].values():
        if 0 in value and 1 in value:
            # channel_list.append(value[0] + value[1])
            channel_list.append(value[1])

    features = _extract_window_features(channel_list)
    for feature in features:
        feature['left_area_label'] = sample['left_area_label']
        feature['right_area_label'] = sample['right_area_label']
        feature['motion_state_label'] = sample['motion_state_label']
        feature['travel_state_label'] = sample['travel_state_label']
        feature['trips_count_label'] = sample['trips_count_label']

    return features

def train():
    data_folder = "train_datas"
    feature_dicts = []
    for sample in process_folder(data_folder):
        features = extract_sample_features(sample)
        for feature in features:
            feature_dicts.append(feature)

    df_features = pd.DataFrame(feature_dicts)
    # ================================ 训练模型 =================================
    label_columns = ['left_area_label']
    all_label_columns = ['left_area_label', 'right_area_label', 'motion_state_label', 'travel_state_label', 'trips_count_label']
    confusion_matrices = []

    for label in label_columns:
        print(f"Processing {label}...")
        
        X = df_features.drop(columns=all_label_columns)  # 特征
        y_raw = df_features[label]  # 原始标签（可能是 0,2,3 等）

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)

        model_filename = f'{label}_model.joblib'
        encoder_filename = f'{label}_label_encoder.joblib'
        joblib.dump(model, model_filename)
        joblib.dump(le, encoder_filename)  # 保存编码器，用于推理时还原标签

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append((label, cm))

        print(f"Label mapping for {label}:")
        for idx, cls in enumerate(le.classes_):
            print(f"  Encoded {idx} -> Original {cls}")

    # ================================ 画热力图 =================================
    for label, cm in confusion_matrices:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix for {label}', fontsize=16, pad=20)
        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        plt.tight_layout()
        
        filename = f'confusion_matrix_{label}.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved confusion matrix to {filename}")
        
        plt.close()  # 重要：关闭当前 figure，防止内存泄漏

if __name__ == "__main__":
    train()