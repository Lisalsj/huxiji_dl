import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from src.model import create_model, MultiTaskLoss
from src.data_loader import DataProcessor
from src.config import Config

class Trainer:
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        self.model = create_model().to(self.device)
        self.criterion = MultiTaskLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc="Training"):
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            numerical_outputs, mode_output = self.model(batch_features)
            loss = self.criterion(numerical_outputs, mode_output, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct_predictions = [0] * 6  # 5个数值 + 1个模式
        total_predictions = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                numerical_outputs, mode_output = self.model(batch_features)
                loss = self.criterion(numerical_outputs, mode_output, batch_labels)
                
                total_loss += loss.item()
                
                # 计算准确率
                batch_size = batch_features.size(0)
                total_predictions += batch_size
                
                # 数值变化准确率
                for i, output in enumerate(numerical_outputs):
                    pred = torch.argmax(output, dim=1)
                    correct_predictions[i] += (pred == batch_labels[:, i]).sum().item()
                
                # 模式变化准确率
                mode_pred = torch.argmax(mode_output, dim=1)
                correct_predictions[5] += (mode_pred == batch_labels[:, 5]).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracies = [correct / total_predictions for correct in correct_predictions]
        
        return avg_loss, accuracies
    
    def train(self, train_loader, val_loader):
        """完整训练过程"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_accuracies = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracies: {[f'{acc:.4f}' for acc in val_accuracies]}")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model()
                print("Model saved!")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("Training completed!")
    
    def save_model(self):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, self.config.MODEL_SAVE_PATH)
    
    def load_model(self, model_path):
        """加载模型"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file {model_path} not found!")
            return False

def train_model(data_path):
    """训练模型的主函数"""
    # 初始化数据处理器
    data_processor = DataProcessor()

    # 加载和预处理数据
    print("Loading and preprocessing data...")
    df = data_processor.load_data(data_path)
    X, y = data_processor.preprocess_data(df)
    data_processor.print_label_distribution(y)
    # 划分数据集
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(X, y)
    
    # 标准化特征
    print("Normalizing features...")
    X_train_scaled, X_val_scaled, X_test_scaled = data_processor.normalize_features(
        X_train, X_val, X_test
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = data_processor.create_dataloaders(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    )
    
    print(f"Data split: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 初始化训练器并开始训练
    trainer = Trainer()
    trainer.train(train_loader, val_loader)
    
    # 在测试集上评估
    print("\nEvaluating on test set...")
    test_loss, test_accuracies = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracies: {[f'{acc:.4f}' for acc in test_accuracies]}")
    
    return trainer