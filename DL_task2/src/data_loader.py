import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from .utils import map_ventmode_to_int, extract_trend_features, validate_labels
from .config import Config


class VentilatorDataset(Dataset):
    def __init__(self, features, targets_numerical, targets_categorical):
        self.features = torch.FloatTensor(features)
        self.targets_numerical = torch.FloatTensor(targets_numerical)
        self.targets_categorical = torch.LongTensor(targets_categorical)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.targets_numerical[idx],
            self.targets_categorical[idx]
        )


class VentilatorDataLoader:
    def __init__(self, csv_path, is_training=True, use_test_split=False):
        self.csv_path = csv_path
        self.is_training = is_training
        self.use_test_split = use_test_split  # 新参数：是否返回测试集
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()

        # Input feature columns (non next_ columns)
        self.input_columns = [
            'Sex', 'Age', 'VTI', 'VTE', 'RATE', 'FSPN', 'MVSPN', 'PPEAK',
            'PMEAN', 'PPLAT', 'MVLEAK', 'Vent_Paw_Wave', 'Vent_Flow_Wave',
            'Vent_Vol_Wave', 'SET_SIMVRR', 'SET_VENTMODE', 'SET_TRIGGERFLOW',
            'SET_OXYGEN', 'SET_PEEP', 'SET_PSUPP'
        ]

        # Target columns
        self.numerical_targets = ['next_SET_SIMVRR', 'next_SET_TRIGGERFLOW',
                                  'next_SET_OXYGEN', 'next_SET_PEEP', 'next_SET_PSUPP']
        self.categorical_targets = ['next_SET_VENTMODE']

    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        df = pd.read_csv(self.csv_path)

        # 打印原始数据中的 VENTMODE 值统计
        print("Original SET_VENTMODE values:")
        print(df['SET_VENTMODE'].value_counts())
        print("\nOriginal next_SET_VENTMODE values:")
        print(df['next_SET_VENTMODE'].value_counts())

        # Map SET_VENTMODE and next_SET_VENTMODE to integers
        df['SET_VENTMODE'] = df['SET_VENTMODE'].apply(map_ventmode_to_int)
        df['next_SET_VENTMODE'] = df['next_SET_VENTMODE'].apply(map_ventmode_to_int)

        # 打印映射后的值统计
        print("\nMapped SET_VENTMODE values:")
        print(df['SET_VENTMODE'].value_counts().sort_index())
        print("\nMapped next_SET_VENTMODE values:")
        print(df['next_SET_VENTMODE'].value_counts().sort_index())

        # Extract basic features
        features = df[self.input_columns].values

        # Extract trend features
        trend_features = extract_trend_features(df)

        # Combine features
        all_features = np.concatenate([features, trend_features], axis=1)

        # Extract targets
        targets_numerical = df[self.numerical_targets].values
        targets_categorical = df[self.categorical_targets].values.flatten()

        # 验证标签范围
        if Config.DEBUG_LABELS:
            print(f"\nValidating categorical labels...")
            if not validate_labels(targets_categorical, Config.VENTMODE_CLASSES):
                raise ValueError("Categorical labels are out of range!")

        # Check for any invalid values
        invalid_mask = (targets_categorical < 0) | (targets_categorical >= Config.VENTMODE_CLASSES)
        if invalid_mask.any():
            print(f"Found {invalid_mask.sum()} invalid categorical labels")
            print(f"Invalid values: {targets_categorical[invalid_mask]}")
            # Replace invalid values with most common class
            most_common_class = np.bincount(targets_categorical[~invalid_mask]).argmax()
            targets_categorical[invalid_mask] = most_common_class
            print(f"Replaced with most common class: {most_common_class}")

        # Normalize features
        if self.is_training:
            all_features = self.scaler_features.fit_transform(all_features)
            targets_numerical = self.scaler_targets.fit_transform(targets_numerical)
        else:
            all_features = self.scaler_features.transform(all_features)
            targets_numerical = self.scaler_targets.transform(targets_numerical)

        return all_features, targets_numerical, targets_categorical

    def split_data(self, features, targets_numerical, targets_categorical):
        """Split data into train, test, and validation sets"""
        n_samples = len(features)

        # Calculate split indices
        train_end = int(n_samples * Config.TRAIN_RATIO)
        val_end = train_end + int(n_samples * Config.VAL_RATIO)

        # Split data sequentially
        train_features = features[:train_end]
        train_targets_num = targets_numerical[:train_end]
        train_targets_cat = targets_categorical[:train_end]

        val_features = features[train_end:val_end]
        val_targets_num = targets_numerical[train_end:val_end]
        val_targets_cat = targets_categorical[train_end:val_end]

        test_features = features[val_end:]
        test_targets_num = targets_numerical[val_end:]
        test_targets_cat = targets_categorical[val_end:]

        # 验证每个分割的标签范围
        if Config.DEBUG_LABELS:
            print("Validating train labels...")
            validate_labels(train_targets_cat, Config.VENTMODE_CLASSES)
            print("Validating val labels...")
            validate_labels(val_targets_cat, Config.VENTMODE_CLASSES)
            print("Validating test labels...")
            validate_labels(test_targets_cat, Config.VENTMODE_CLASSES)

        print(f"\nData split summary:")
        print(f"  Total samples: {n_samples}")
        print(f"  Train: {len(train_features)} ({len(train_features) / n_samples * 100:.1f}%)")
        print(f"  Validation: {len(val_features)} ({len(val_features) / n_samples * 100:.1f}%)")
        print(f"  Test: {len(test_features)} ({len(test_features) / n_samples * 100:.1f}%)")

        return (
            (train_features, train_targets_num, train_targets_cat),
            (val_features, val_targets_num, val_targets_cat),
            (test_features, test_targets_num, test_targets_cat)
        )

    def create_data_loaders(self):
        """Create PyTorch data loaders"""
        features, targets_numerical, targets_categorical = self.load_and_preprocess_data()

        if self.is_training:
            train_data, val_data, test_data = self.split_data(features, targets_numerical, targets_categorical)

            train_dataset = VentilatorDataset(*train_data)
            val_dataset = VentilatorDataset(*val_data)

            train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

            if self.use_test_split:
                # 只有在明确要求时才返回测试集
                test_dataset = VentilatorDataset(*test_data)
                test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
                return train_loader, val_loader, test_loader
            else:
                # 训练时不返回测试集，确保测试集完全未见过
                return train_loader, val_loader, None

        else:
            # 预测模式：可以选择使用全部数据或只使用测试集
            if self.use_test_split:
                # 只使用测试集部分进行预测（真正的未见过数据）
                _, _, test_data = self.split_data(features, targets_numerical, targets_categorical)
                dataset = VentilatorDataset(*test_data)
                print(f"Using test split for prediction: {len(dataset)} samples")
            else:
                # 使用全部数据进行预测
                dataset = VentilatorDataset(features, targets_numerical, targets_categorical)
                print(f"Using full dataset for prediction: {len(dataset)} samples")

            loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
            return loader