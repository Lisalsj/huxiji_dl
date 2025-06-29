import torch
import pandas as pd
import numpy as np
import json
import torch.serialization
from src.model import create_model
from src.data_loader import DataProcessor, VentilatorDataset
from src.config import Config
from src.utils import format_prediction_result
from torch.utils.data import DataLoader


# 添加对自定义类的反序列化支持
torch.serialization.add_safe_globals([Config])


class Predictor:
    def __init__(self, model_path):
        self.config = Config()
        self.device = self.config.DEVICE
        self.data_processor = DataProcessor()

        # 初始化模型并加载权重
        self.model = create_model().to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def preprocess_single_sample(self, df):
        """预处理单个样本"""
        df['SET_VENTMODE'] = df['SET_VENTMODE'].astype(str).map(
            self.config.VENTMODE_MAPPING
        ).fillna(df['SET_VENTMODE'])

        input_features = [col for col in self.config.INPUT_FEATURES if col in df.columns]
        X = df[input_features].values
        X_normalized = self.simple_normalize(X)
        return X_normalized

    def simple_normalize(self, X):
        """简单归一化方法（仅用于单样本）"""
        return (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-8)

    def predict(self, data_path, output_path=None):
        """对验证集进行预测"""
        df = self.data_processor.load_data(data_path)
        X, y = self.data_processor.preprocess_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.split_data(X, y)
        X_train, X_val, X_test = self.data_processor.normalize_features(X_train, X_val, X_test)

        val_dataset = VentilatorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

        predictions = []
        sample_id = 0

        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(self.device)
                numerical_outputs, mode_output = self.model(x_batch)

                numerical_preds_batch = [torch.argmax(output, dim=1).tolist() for output in numerical_outputs]
                mode_preds_batch = torch.argmax(mode_output, dim=1).tolist()

                for i in range(len(x_batch)):
                    numerical_preds = [preds[i] for preds in numerical_preds_batch]
                    mode_pred = mode_preds_batch[i]

                    prediction_text = format_prediction_result(numerical_preds, mode_pred, self.config)
                    predictions.append({
                        "id": sample_id,
                        "task1_output": prediction_text
                    })
                    sample_id += 1

        if output_path is None:
            output_path = self.config.PREDICTION_OUTPUT_PATH

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to {output_path}")
        return predictions

    def predict_single(self, sample_data):
        """预测单个样本（dict输入）"""
        df = pd.DataFrame([sample_data])
        X = self.preprocess_single_sample(df)
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            numerical_outputs, mode_output = self.model(X_tensor)

            numerical_preds = [torch.argmax(output, dim=1).item() for output in numerical_outputs]
            mode_pred = torch.argmax(mode_output, dim=1).item()

            prediction_text = format_prediction_result(numerical_preds, mode_pred, self.config)
            return prediction_text


def predict_from_csv(model_path, data_path, output_path=None):
    """CSV文件预测入口"""
    predictor = Predictor(model_path)
    return predictor.predict(data_path, output_path)
