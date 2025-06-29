import json
import numpy as np
from src.config import Config

def format_prediction_result(numerical_preds, mode_pred, config):
    """格式化预测结果为指定的字符串格式"""
    result_parts = []
    
    # 处理数值变化字段
    for i, field in enumerate(config.PREDICTION_FIELDS):
        change_type = config.CHANGE_LABELS[numerical_preds[i]]
        result_parts.append(f"{field}: {change_type}")
    
    # 处理模式变化字段
    mode_change = config.MODE_LABELS[mode_pred]
    result_parts.append(f"SET_VENTMODE: {mode_change}")
    
    return "; ".join(result_parts)

def save_predictions_to_json(predictions, output_path):
    """保存预测结果到JSON文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

def load_predictions_from_json(file_path):
    """从JSON文件加载预测结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        return predictions
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None

def calculate_accuracy(predictions, ground_truth):
    """计算预测准确率"""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    correct = 0
    total = len(predictions)
    
    for pred, truth in zip(predictions, ground_truth):
        if pred == truth:
            correct += 1
    
    accuracy = correct / total
    return accuracy

def print_model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model architecture:\n{model}")

def create_sample_data():
    """创建示例数据用于测试"""
    sample = {
        'Sex': 1,
        'Age': 80,
        'VTI': 514.58,
        'VTE': 525.16,
        'RATE': 15.25,
        'FSPN': 4.58,
        'MVSPN': 1.31,
        'PPEAK': 17.0,
        'PMEAN': 8.88,
        'PPLAT': 17.0,
        'MVLEAK': 0.0,
        'Vent_Paw_Wave': 8.50,
        'Vent_Flow_Wave': -0.11,
        'Vent_Vol_Wave': 175.66,
        'SET_SIMVRR': 10,
        'SET_VENTMODE': '5119',
        'SET_TRIGGERFLOW': 2.0,
        'SET_OXYGEN': 30,
        'SET_PEEP': 5,
        'SET_PSUPP': 12
    }
    return sample

def validate_data_format(df):
    """验证数据格式是否正确"""
    config = Config()
    required_columns = config.INPUT_FEATURES + [f'next_{field}' for field in config.PREDICTION_FIELDS] + ['next_SET_VENTMODE']
    
    missing_columns = []
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print("Data format validation passed!")
    return True

def get_class_distribution(labels):
    """获取类别分布统计"""
    distribution = {}
    for i in range(labels.shape[1]):
        unique, counts = np.unique(labels[:, i], return_counts=True)
        distribution[f'task_{i}'] = dict(zip(unique, counts))
    
    return distribution