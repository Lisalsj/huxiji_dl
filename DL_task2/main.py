import argparse
import sys
from src.train import train_model
from src.predict import predict_from_csv
from src.config import Config


def main():
    parser = argparse.ArgumentParser(description='Ventilator Parameter Prediction')
    parser.add_argument('mode', choices=['train', 'predict'], help='Mode: train or predict')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--model', default=Config.MODEL_PATH, help='Path to model file')
    parser.add_argument('--output', default=Config.PREDICTIONS_PATH, help='Path to output predictions file')
    parser.add_argument('--use-unseen', action='store_true', default=True,
                        help='Use only unseen test data for prediction (default: True)')
    parser.add_argument('--use-full', action='store_true', default=False,
                        help='Use full dataset for prediction (overrides --use-unseen)')

    args = parser.parse_args()

    if args.mode == 'train':
        print("🚀 Starting training...")
        print("📊 Test set will be reserved for final evaluation!")
        train_model(args.data)
        print("✅ Training completed!")

    elif args.mode == 'predict':
        print("🔮 Starting prediction...")

        # Determine whether to use unseen data
        use_unseen_data = not args.use_full  # If use_full is True, then use_unseen_data is False

        if use_unseen_data:
            print("📊 Will predict on UNSEEN test data only")
        else:
            print("📊 Will predict on FULL dataset")

        predict_from_csv(args.data, args.model, args.output, use_unseen_data)
        print("✅ Prediction completed!")


if __name__ == '__main__':
    main()