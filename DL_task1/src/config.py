import torch

class Config:
    # 数据相关配置
    DATA_PATH = "data.csv"
    TRAIN_RATIO = 0.5
    TEST_RATIO = 0.2
    VAL_RATIO = 0.3
    
    # 模型相关配置
    INPUT_DIM = 15  # 非next_开头的特征数量
    HIDDEN_DIM = 128
    OUTPUT_DIM = 6  # 5个数值变化 + 1个模式变化
    
    # 训练相关配置
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型保存路径
    MODEL_SAVE_PATH = "E:/aaaa_ubuntu_code/huxiji_vla/our_data_work_v0.1/DL/model_out/model.pth"
    PREDICTION_OUTPUT_PATH = "E:/aaaa_ubuntu_code/huxiji_vla/our_data_work_v0.1/DL/results/predictions.json"
    
    # 字段映射
    VENTMODE_MAPPING = {
        "50007^MNDRY_VENT_MODE_SIMVVC^99MNDRY": 1,
        "50009^MNDRY_VENT_MODE_SIMVPC^99MNDRY": 2,
        "50021^MNDRY_VENT_MODE_CPAP_PLUS_PS^99MNDRY": 3,
        "50022^MNDRY_VENT_MODE_SIMVPC_PLUS_PRVC^99MNDRY": 4,
        "50054^MNDRY_VENT_MODE_PACV^99MNDRY": 5,
        "50062^MNDRY_VENT_MODE_VACV^99MNDRY": 6,
        "5116": 7,
        "5117": 8,
        "5118": 9,
        "5119": 10,
        "5120": 11
    }
    
    # 输入特征列名（非next_开头的列）
    INPUT_FEATURES = [
        'Sex', 'Age', 'VTI', 'VTE', 'RATE', 'FSPN', 'MVSPN', 'PPEAK', 
        'PMEAN', 'PPLAT', 'MVLEAK', 'Vent_Paw_Wave', 'Vent_Flow_Wave', 
        'Vent_Vol_Wave', 'SET_SIMVRR', 'SET_VENTMODE', 'SET_TRIGGERFLOW',
        'SET_OXYGEN', 'SET_PEEP', 'SET_PSUPP'
    ]
    
    # 需要预测变化的字段
    PREDICTION_FIELDS = [
        'SET_SIMVRR', 'SET_TRIGGERFLOW', 'SET_OXYGEN', 'SET_PEEP', 'SET_PSUPP'
    ]
    
    # 分类标签映射
    CHANGE_LABELS = {
        0: 'decrease',
        1: 'no change', 
        2: 'increase'
    }
    
    MODE_LABELS = {
        0: 'no change',
        1: 'changed'
    }