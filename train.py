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
    datas = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            left_area, right_area, motion_state, travel_state, trips_count = filename.split("__")
            left_area = int(left_area.split('_')[-1])
            right_area = int(right_area.split('_')[-1])
            trips_count = int(trips_count.split('.')[0])
            file_path = os.path.join(root, filename)
            parse_data = parse_file(file_path)

            datas.append(
                {
                    'left_area_label': left_area,
                    'right_area_label': right_area,
                    'motion_state_label': motion_state,
                    'travel_state_label': travel_state,
                    'trips_count_label': trips_count,
                    'data': parse_data,
                }
            )
    return datas

def extract_features(sample):
    labels = {
        'left_area': sample['left_area_label'],
        'right_area': sample['right_area_label'],
        'motion_state': sample['motion_state_label'],
        'travel_state': sample['travel_state_label'],
        'trips_count': sample['trips_count_label']
    }

    features = []
    for timestamp, data in sample['data'].items():
        row = {
            'timestamp': timestamp,
            'channel_0': data[0], # 左脚
            'channel_1': data[1], # 右脚
            **labels
        }
        features.append(row)
    return pd.DataFrame(features)

def extract_sample_features(sample):
    timestamps = list(sample['data'].keys())
    channel_0_list = [np.array(v[0]) for v in sample['data'].values()]
    channel_1_list = [np.array(v[1]) for v in sample['data'].values()]

    all_ch0 = np.concatenate(channel_0_list) if channel_0_list else np.array([])
    all_ch1 = np.concatenate(channel_1_list) if channel_1_list else np.array([])

    def safe_stat(arr, func, default=0.0):
        return func(arr) if len(arr) > 0 else default

    features = {}

    features['ch0_mean'] = safe_stat(all_ch0, np.mean)
    features['ch0_std'] = safe_stat(all_ch0, np.std)
    features['ch0_max'] = safe_stat(all_ch0, np.max)
    features['ch0_min'] = safe_stat(all_ch0, np.min)
    features['ch0_nonzero_ratio'] = safe_stat(all_ch0, lambda x: np.count_nonzero(x) / len(x))
    features['ch0_energy'] = safe_stat(all_ch0, lambda x: np.sum(x ** 2))

    features['ch1_mean'] = safe_stat(all_ch1, np.mean)
    features['ch1_std'] = safe_stat(all_ch1, np.std)
    features['ch1_max'] = safe_stat(all_ch1, np.max)
    features['ch1_min'] = safe_stat(all_ch1, np.min)
    features['ch1_nonzero_ratio'] = safe_stat(all_ch1, lambda x: np.count_nonzero(x) / len(x))
    features['ch1_energy'] = safe_stat(all_ch1, lambda x: np.sum(x ** 2))

    if len(all_ch0) == len(all_ch1) and len(all_ch0) > 0:
        features['ch0_ch1_corr'] = np.corrcoef(all_ch0, all_ch1)[0, 1]
        features['ch_diff_mean'] = np.mean(np.abs(all_ch0 - all_ch1))
    else:
        features['ch0_ch1_corr'] = 0.0
        features['ch_diff_mean'] = 0.0

    features['num_timestamps'] = len(timestamps)
    if len(timestamps) > 1:
        def parse_ts(ts):
            parts = ts.split('_')
            return int(parts[0]) * 3600000 + int(parts[1]) * 60000 + int(parts[2]) * 1000 + int(parts[3])
        ts_ms = [parse_ts(ts) for ts in timestamps]
        features['duration_ms'] = max(ts_ms) - min(ts_ms)
    else:
        features['duration_ms'] = 0

    features['left_area_label'] = sample['left_area_label']
    features['right_area_label'] = sample['right_area_label']
    features['motion_state_label'] = sample['motion_state_label']
    features['travel_state_label'] = sample['travel_state_label']
    features['trips_count_label'] = sample['trips_count_label']

    return features

def train():
    data_folder = "train_datas"
    parsed_json_datas = process_folder(data_folder)

    # all_data = []
    # for sample in parsed_json_datas:
    #     df = extract_features(sample)
    #     all_data.append(df)

    # final_df = pd.concat(all_data, ignore_index=True)

    feature_dicts = [extract_sample_features(sample) for sample in parsed_json_datas]
    df_features = pd.DataFrame(feature_dicts)

    # ================================ 训练模型 =================================
    label_columns = ['left_area_label', 'right_area_label', 'motion_state_label', 'travel_state_label', 'trips_count_label']
    confusion_matrices = []

    for label in label_columns:
        print(f"Processing {label}...")
        X = df_features.drop(columns=label_columns)  # 特征
        y = df_features[label]  # 当前处理的标签
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        model_filename = f'{label}_model.joblib'
        joblib.dump(model, model_filename)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append((label, cm))

    # ================================ 画热力图 =================================
    for label, cm in confusion_matrices:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {label}', fontsize=16, pad=20)
        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    train()