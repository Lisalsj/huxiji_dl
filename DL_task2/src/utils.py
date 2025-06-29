import json
import torch
import numpy as np
import os
# Mapping for SET_VENTMODE (修改为从0开始)
VENTMODE_MAPPING = {
    "50007^MNDRY_VENT_MODE_SIMVVC^99MNDRY": 0,
    "50009^MNDRY_VENT_MODE_SIMVPC^99MNDRY": 1,
    "50021^MNDRY_VENT_MODE_CPAP_PLUS_PS^99MNDRY": 2,
    "50022^MNDRY_VENT_MODE_SIMVPC_PLUS_PRVC^99MNDRY": 3,
    "50054^MNDRY_VENT_MODE_PACV^99MNDRY": 4,
    "50062^MNDRY_VENT_MODE_VACV^99MNDRY": 5,
    "5116": 6,
    "5117": 7,
    "5118": 8,
    "5119": 9,
    "5120": 10
}

# Reverse mapping for prediction output
REVERSE_VENTMODE_MAPPING = {v: k for k, v in VENTMODE_MAPPING.items()}

def map_ventmode_to_int(ventmode_value):
    """Map ventmode string to integer (0-based indexing for PyTorch)"""
    # 确保返回值在有效范围内
    mapped_value = VENTMODE_MAPPING.get(str(ventmode_value), -1)
    if mapped_value == -1:
        print(f"Warning: Unknown VENTMODE value: {ventmode_value}, using default (9 for '5119')")
        return 9  # 默认使用 '5119' 对应的索引
    return mapped_value

def map_int_to_ventmode(int_value):
    """Map integer back to ventmode string"""
    return REVERSE_VENTMODE_MAPPING.get(int_value, "5119")

def calculate_trend(current_value, next_value, is_ventmode=False):
    """
    Calculate trend between current and next values
    Returns: 0 (increase), 1 (decrease), 2 (no change), 3 (changed for ventmode)
    """
    if is_ventmode:
        return 2 if current_value == next_value else 3
    else:
        if next_value > current_value:
            return 0  # increase
        elif next_value < current_value:
            return 1  # decrease
        else:
            return 2  # no change

def extract_trend_features(df):
    """Extract trend features from dataframe"""
    trend_features = []
    
    # Parameters to calculate trends for
    trend_params = ['SET_SIMVRR', 'SET_TRIGGERFLOW', 'SET_OXYGEN', 'SET_PEEP', 'SET_PSUPP']
    
    for param in trend_params:
        current_col = param
        next_col = f'next_{param}'
        
        if current_col in df.columns and next_col in df.columns:
            trend = df.apply(lambda row: calculate_trend(
                row[current_col], row[next_col], is_ventmode=False
            ), axis=1)
            trend_features.append(trend.values)
    
    # Special handling for SET_VENTMODE
    if 'SET_VENTMODE' in df.columns and 'next_SET_VENTMODE' in df.columns:
        ventmode_trend = df.apply(lambda row: calculate_trend(
            row['SET_VENTMODE'], row['next_SET_VENTMODE'], is_ventmode=True
        ), axis=1)
        trend_features.append(ventmode_trend.values)
    
    return np.array(trend_features).T

def save_predictions_json(predictions, output_path):
    """Save predictions in the required JSON format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  #
    results = []
    
    for i, pred in enumerate(predictions):
        # pred should be a dict with keys: SET_SIMVRR, SET_TRIGGERFLOW, SET_OXYGEN, SET_PEEP, SET_PSUPP, SET_VENTMODE
        task2_output = {
            "SET_SIMVRR": str(int(pred['SET_SIMVRR'])),
            "SET_VENTMODE": map_int_to_ventmode(int(pred['SET_VENTMODE'])),
            "SET_TRIGGERFLOW": str(pred['SET_TRIGGERFLOW']),
            "SET_OXYGEN": str(int(pred['SET_OXYGEN'])),
            "SET_PEEP": str(int(pred['SET_PEEP'])),
            "SET_PSUPP": str(int(pred['SET_PSUPP']))
        }
        
        result = {
            "id": i,
            "task2_output": json.dumps(task2_output)
        }
        results.append(result)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def validate_labels(labels, num_classes):
    """Validate that all labels are in the correct range"""
    min_label = labels.min()
    max_label = labels.max()
    
    print(f"Label range: [{min_label}, {max_label}]")
    print(f"Expected range: [0, {num_classes-1}]")
    
    if min_label < 0 or max_label >= num_classes:
        print(f"ERROR: Labels out of range! Min: {min_label}, Max: {max_label}, Expected: [0, {num_classes-1}]")
        return False
    return True