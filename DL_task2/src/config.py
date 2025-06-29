import torch


class Config:
    # Model configuration
    INPUT_SIZE = 26  # 20 original features + 6 trend features
    HIDDEN_SIZES = [256, 128, 64]
    DROPOUT_RATE = 0.2

    # Output configuration
    NUMERICAL_OUTPUTS = 5  # SET_SIMVRR, SET_TRIGGERFLOW, SET_OXYGEN, SET_PEEP, SET_PSUPP
    CATEGORICAL_OUTPUTS = 1  # SET_VENTMODE
    VENTMODE_CLASSES = 11  # Number of unique ventmode categories (0-10)

    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data split ratios - 修改为更合理的划分
    TRAIN_RATIO = 0.5  # 60% 训练集
    VAL_RATIO = 0.2  # 20% 验证集（用于模型选择）
    TEST_RATIO = 0.3  # 20% 测试集（真正的未见过数据，训练时不使用）

    # File paths
    MODEL_PATH = 'model.pth'
    PREDICTIONS_PATH = 'predictions.json'

    # Random seed for reproducibility
    RANDOM_SEED = 42

    # Debug settings
    DEBUG_LABELS = True  # Set to True to validate label ranges

    # 新增：训练时是否使用测试集进行监控
    USE_TEST_DURING_TRAINING = False  # 设为False确保测试集完全未见过