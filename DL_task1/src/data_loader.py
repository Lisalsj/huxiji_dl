import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from src.config import Config


class VentilatorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.config = Config()

    def load_data(self, file_path):
        """加载CSV数据"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def preprocess_data(self, df):
        """预处理数据"""
        # VENTMODE 映射（字符串映射为数字）
        df['SET_VENTMODE'] = df['SET_VENTMODE'].astype(str).map(
            self.config.VENTMODE_MAPPING
        ).fillna(df['SET_VENTMODE'])

        df['next_SET_VENTMODE'] = df['next_SET_VENTMODE'].astype(str).map(
            self.config.VENTMODE_MAPPING
        ).fillna(df['next_SET_VENTMODE'])

        # 显式转换 PREDICTION_FIELDS 及对应 next 字段为 float 类型
        for field in self.config.PREDICTION_FIELDS:
            df[field] = pd.to_numeric(df[field], errors='coerce')
            df[f'next_{field}'] = pd.to_numeric(df[f'next_{field}'], errors='coerce')

        df['SET_VENTMODE'] = pd.to_numeric(df['SET_VENTMODE'], errors='coerce')
        df['next_SET_VENTMODE'] = pd.to_numeric(df['next_SET_VENTMODE'], errors='coerce')

        # 提取输入特征
        input_features = [col for col in self.config.INPUT_FEATURES if col in df.columns]
        X = df[input_features].values

        # 生成标签
        y = self._generate_labels(df)

        return X, y

    def _generate_labels(self, df):
        """生成标签"""
        labels = []
        for _, row in df.iterrows():
            label = []
            for field in self.config.PREDICTION_FIELDS:
                current_val = row[field]
                next_val = row[f'next_{field}']
                try:
                    current_val = float(current_val)
                    next_val = float(next_val)
                    if np.isnan(current_val) or np.isnan(next_val):
                        label.append(1)  # treat NaNs as "no change"
                    elif next_val > current_val:
                        label.append(2)  # increase
                    elif next_val < current_val:
                        label.append(0)  # decrease
                    else:
                        label.append(1)  # no change
                except:
                    label.append(1)  # fallback: no change

            # VENTMODE
            try:
                current_mode = int(row['SET_VENTMODE'])
                next_mode = int(row['next_SET_VENTMODE'])
                label.append(1 if current_mode != next_mode else 0)
            except:
                label.append(0)  # fallback: no change

            labels.append(label)
        return np.array(labels)

    def split_data(self, X, y):
        """划分数据集"""
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def normalize_features(self, X_train, X_val, X_test):
        """标准化特征"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """创建数据加载器"""
        train_dataset = VentilatorDataset(X_train, y_train)
        val_dataset = VentilatorDataset(X_val, y_val)
        test_dataset = VentilatorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

        return train_loader, val_loader, test_loader

    def print_label_distribution(self, y):
        """打印每个标签字段的标签分布"""
        y = np.array(y)
        num_fields = y.shape[1]
        total_fields = self.config.PREDICTION_FIELDS + ["SET_VENTMODE"]

        for i in range(num_fields):
            field = total_fields[i] if i < len(total_fields) else f"Field {i}"
            counter = Counter(y[:, i])
            print(f"\n字段: {field}")
            for label_id, count in sorted(counter.items()):
                label_name = self._id_to_label(label_id, is_mode=(field == "SET_VENTMODE"))
                print(f"  {label_name:<10}: {count} samples")

    def _id_to_label(self, id_, is_mode=False):
        if is_mode:
            return "changed" if id_ == 1 else "no change"
        return {0: "decrease", 1: "no change", 2: "increase"}.get(id_, "unknown")
