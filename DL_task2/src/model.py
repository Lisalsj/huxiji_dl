import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config

class VentilatorPredictionModel(nn.Module):
    def __init__(self):
        super(VentilatorPredictionModel, self).__init__()
        
        # Shared feature extractor
        layers = []
        input_size = Config.INPUT_SIZE
        
        for hidden_size in Config.HIDDEN_SIZES:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(Config.DROPOUT_RATE)
            ])
            input_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Numerical output heads
        self.numerical_head = nn.Linear(Config.HIDDEN_SIZES[-1], Config.NUMERICAL_OUTPUTS)
        
        # Categorical output head (for SET_VENTMODE)
        self.categorical_head = nn.Linear(Config.HIDDEN_SIZES[-1], Config.VENTMODE_CLASSES)
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Numerical predictions
        numerical_outputs = self.numerical_head(features)
        
        # Categorical predictions
        categorical_outputs = self.categorical_head(features)
        
        return numerical_outputs, categorical_outputs

class CombinedLoss(nn.Module):
    def __init__(self, numerical_weight=1.0, categorical_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.numerical_weight = numerical_weight
        self.categorical_weight = categorical_weight
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, numerical_pred, categorical_pred, numerical_target, categorical_target):
        numerical_loss = self.mse_loss(numerical_pred, numerical_target)
        categorical_loss = self.ce_loss(categorical_pred, categorical_target)
        
        total_loss = (self.numerical_weight * numerical_loss + 
                     self.categorical_weight * categorical_loss)
        
        return total_loss, numerical_loss, categorical_loss