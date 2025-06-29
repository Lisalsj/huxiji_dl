import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config

class VentilatorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VentilatorMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        
        # 为5个数值变化预测创建分类器（每个3分类）
        self.numerical_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim // 4, 3) for _ in range(5)
        ])
        
        # 为VENTMODE变化预测创建分类器（2分类）
        self.mode_classifier = nn.Linear(hidden_dim // 4, 2)
        
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
        
    def forward(self, x):
        # 前向传播
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # 多任务输出
        numerical_outputs = []
        for classifier in self.numerical_classifiers:
            numerical_outputs.append(classifier(x))
        
        mode_output = self.mode_classifier(x)
        
        return numerical_outputs, mode_output

def create_model():
    """创建模型实例"""
    config = Config()
    
    # 动态计算输入维度
    input_dim = len(config.INPUT_FEATURES)
    
    model = VentilatorMLP(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM
    )
    
    return model

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.numerical_criterion = nn.CrossEntropyLoss()
        self.mode_criterion = nn.CrossEntropyLoss()
    
    def forward(self, numerical_outputs, mode_output, targets):
        """
        numerical_outputs: list of 5 tensors, each of shape (batch_size, 3)
        mode_output: tensor of shape (batch_size, 2)
        targets: tensor of shape (batch_size, 6)
        """
        total_loss = 0
        
        # 计算数值变化的损失
        for i, output in enumerate(numerical_outputs):
            loss = self.numerical_criterion(output, targets[:, i])
            total_loss += loss
        
        # 计算模式变化的损失
        mode_loss = self.mode_criterion(mode_output, targets[:, 5])
        total_loss += mode_loss
        
        return total_loss