import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from .model import VentilatorPredictionModel, CombinedLoss
from .data_loader import VentilatorDataLoader
from .config import Config
from .utils import set_random_seed


class Trainer:
    def __init__(self, csv_path):
        set_random_seed(Config.RANDOM_SEED)

        self.device = Config.DEVICE
        print(f"Using device: {self.device}")

        self.model = VentilatorPredictionModel().to(self.device)
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Load data - 注意：训练时不使用测试集
        print("Loading and preprocessing data...")
        self.data_loader = VentilatorDataLoader(csv_path, is_training=True, use_test_split=False)
        train_val_test = self.data_loader.create_data_loaders()

        self.train_loader = train_val_test[0]
        self.val_loader = train_val_test[1]
        self.test_loader = train_val_test[2]  # 这里是 None

        print(f"Data loaded successfully:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Validation batches: {len(self.val_loader)}")
        print(f"  Test set: {'Reserved for final evaluation' if self.test_loader is None else 'Available'}")

        # For tensorboard logging
        self.writer = SummaryWriter('runs/ventilator_prediction')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_numerical_loss = 0
        total_categorical_loss = 0

        for batch_idx, (features, targets_numerical, targets_categorical) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets_numerical = targets_numerical.to(self.device)
            targets_categorical = targets_categorical.to(self.device)

            # 调试：检查标签范围
            if batch_idx == 0:
                print(
                    f"Batch {batch_idx}: categorical target range [{targets_categorical.min().item()}, {targets_categorical.max().item()}]")

            self.optimizer.zero_grad()

            numerical_pred, categorical_pred = self.model(features)
            loss, num_loss, cat_loss = self.criterion(
                numerical_pred, categorical_pred, targets_numerical, targets_categorical
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_numerical_loss += num_loss.item()
            total_categorical_loss += cat_loss.item()

        return (total_loss / len(self.train_loader),
                total_numerical_loss / len(self.train_loader),
                total_categorical_loss / len(self.train_loader))

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_numerical_loss = 0
        total_categorical_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for features, targets_numerical, targets_categorical in data_loader:
                features = features.to(self.device)
                targets_numerical = targets_numerical.to(self.device)
                targets_categorical = targets_categorical.to(self.device)

                numerical_pred, categorical_pred = self.model(features)
                loss, num_loss, cat_loss = self.criterion(
                    numerical_pred, categorical_pred, targets_numerical, targets_categorical
                )

                total_loss += loss.item()
                total_numerical_loss += num_loss.item()
                total_categorical_loss += cat_loss.item()

                # Calculate accuracy for categorical predictions
                _, predicted = torch.max(categorical_pred.data, 1)
                total_samples += targets_categorical.size(0)
                correct_predictions += (predicted == targets_categorical).sum().item()

        accuracy = 100 * correct_predictions / total_samples
        avg_loss = total_loss / len(data_loader)
        avg_num_loss = total_numerical_loss / len(data_loader)
        avg_cat_loss = total_categorical_loss / len(data_loader)

        return avg_loss, avg_num_loss, avg_cat_loss, accuracy

    def train(self):
        best_val_loss = float('inf')

        print("=" * 60)
        print("🚀 Starting training...")
        print("📊 Test set is RESERVED and will NOT be used during training!")
        print("=" * 60)

        for epoch in range(Config.NUM_EPOCHS):
            # Training on train set
            train_loss, train_num_loss, train_cat_loss = self.train_epoch()

            # Validation on validation set only
            val_loss, val_num_loss, val_cat_loss, val_accuracy = self.evaluate(self.val_loader)

            # Logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

            print(f'Epoch [{epoch + 1}/{Config.NUM_EPOCHS}]')
            print(f'Train Loss: {train_loss:.4f} (Num: {train_num_loss:.4f}, Cat: {train_cat_loss:.4f})')
            print(
                f'Val Loss: {val_loss:.4f} (Num: {val_num_loss:.4f}, Cat: {val_cat_loss:.4f}) Acc: {val_accuracy:.2f}%')
            print('-' * 60)

            # Save best model based on validation loss only
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'feature_scaler': self.data_loader.scaler_features,
                    'target_scaler': self.data_loader.scaler_targets,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, Config.MODEL_PATH)
                print(f'✅ Saved best model with validation loss: {val_loss:.4f}')

        self.writer.close()
        print('🎉 Training completed!')
        print('📝 Test set remains untouched and ready for final evaluation!')


def train_model(csv_path):
    """Main training function"""
    trainer = Trainer(csv_path)
    trainer.train()