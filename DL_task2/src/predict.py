import torch
import pandas as pd
import numpy as np
from .model import VentilatorPredictionModel
from .data_loader import VentilatorDataLoader
from .config import Config
from .utils import save_predictions_json, map_int_to_ventmode


class Predictor:
    def __init__(self, model_path):
        self.device = Config.DEVICE
        self.model = VentilatorPredictionModel().to(self.device)

        # Load trained model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']

        self.model.eval()
        print(f"✅ Model loaded from {model_path}")
        print(f"📊 Model was trained until epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.4f}")

    def predict(self, csv_path, output_path, use_test_split=True):
        """
        Make predictions on new data

        Args:
            csv_path: Path to CSV file
            output_path: Path to save predictions
            use_test_split: If True, only use test portion (unseen data);
                           If False, use full dataset
        """
        # Load and preprocess data
        data_loader = VentilatorDataLoader(csv_path, is_training=False, use_test_split=use_test_split)
        data_loader.scaler_features = self.feature_scaler
        data_loader.scaler_targets = self.target_scaler

        test_loader = data_loader.create_data_loaders()

        print(f"🔮 Making predictions...")
        if use_test_split:
            print("📊 Using UNSEEN test data (20% of dataset)")
        else:
            print("📊 Using FULL dataset for prediction")

        all_predictions = []

        with torch.no_grad():
            for batch_idx, (features, _, _) in enumerate(test_loader):
                features = features.to(self.device)

                numerical_pred, categorical_pred = self.model(features)

                # Denormalize numerical predictions
                numerical_pred_cpu = numerical_pred.cpu().numpy()
                numerical_pred_denorm = self.target_scaler.inverse_transform(numerical_pred_cpu)

                # Get categorical predictions
                _, categorical_pred_classes = torch.max(categorical_pred, 1)
                categorical_pred_cpu = categorical_pred_classes.cpu().numpy()

                # Convert to required format
                for i in range(len(numerical_pred_denorm)):
                    prediction = {
                        'SET_SIMVRR': numerical_pred_denorm[i][0],
                        'SET_TRIGGERFLOW': numerical_pred_denorm[i][1],
                        'SET_OXYGEN': numerical_pred_denorm[i][2],
                        'SET_PEEP': numerical_pred_denorm[i][3],
                        'SET_PSUPP': numerical_pred_denorm[i][4],
                        'SET_VENTMODE': categorical_pred_cpu[i]
                    }
                    all_predictions.append(prediction)

        # Save predictions
        save_predictions_json(all_predictions, output_path)
        print(f'✅ Predictions saved to {output_path}')
        print(f'📈 Total predictions: {len(all_predictions)}')

        return all_predictions


def predict_from_csv(csv_path, model_path=Config.MODEL_PATH, output_path=Config.PREDICTIONS_PATH, use_unseen_data=True):
    """
    Main prediction function

    Args:
        csv_path: Path to CSV file
        model_path: Path to trained model
        output_path: Path to save predictions
        use_unseen_data: If True, only predict on test split (unseen data)
    """
    predictor = Predictor(model_path)
    return predictor.predict(csv_path, output_path, use_test_split=use_unseen_data)