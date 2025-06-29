import argparse
import os
import sys
from src.train import train_model
from src.predict import predict_from_csv
from src.config import Config
from src.utils import create_sample_data, validate_data_format
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Ventilator Deep Learning Model')
    parser.add_argument('mode', choices=['train', 'predict', 'demo'], 
                       help='Mode: train the model, predict on new data, or run demo')
    parser.add_argument('--data', type=str, default='data.csv',
                       help='Path to the CSV data file')
    parser.add_argument('--model', type=str, default='model.pth',
                       help='Path to the model file')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Path to save predictions')
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.mode == 'train':
        print("Starting training mode...")
        
        # 检查数据文件是否存在
        if not os.path.exists(args.data):
            print(f"Error: Data file '{args.data}' not found!")
            print("Please provide a valid CSV file with the required format.")
            sys.exit(1)
        
        try:
            # 验证数据格式
            df = pd.read_csv(args.data)
            validate_data_format(df)
            
            # 开始训练
            trainer = train_model(args.data)
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)
    
    elif args.mode == 'predict':
        print("Starting prediction mode...")
        
        # 检查模型和数据文件是否存在
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found!")
            print("Please train the model first or provide a valid model file.")
            sys.exit(1)
        
        if not os.path.exists(args.data):
            print(f"Error: Data file '{args.data}' not found!")
            sys.exit(1)
        
        try:
            # 进行预测
            predictions = predict_from_csv(args.model, args.data, args.output)
            print(f"Predictions completed! Results saved to '{args.output}'")
            
            # 显示前几个预测结果
            print("\nFirst 3 predictions:")
            for i, pred in enumerate(predictions[:3]):
                print(f"Sample {pred['id']}: {pred['task1_output']}")
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)
    
    elif args.mode == 'demo':
        print("Running demo mode...")
        
        # 创建示例数据
        sample_data = create_sample_data()
        print("Sample data created:")
        for key, value in sample_data.items():
            print(f"  {key}: {value}")
        
        # 创建示例CSV文件进行演示
        demo_data = []
        for i in range(5):
            row = sample_data.copy()
            # 添加next_字段作为示例
            row.update({
                'next_SET_SIMVRR': row['SET_SIMVRR'] + (i % 3 - 1),  # -1, 0, 1, -1, 0
                'next_SET_TRIGGERFLOW': row['SET_TRIGGERFLOW'],
                'next_SET_OXYGEN': row['SET_OXYGEN'] - (i % 2),  # 减少氧气浓度
                'next_SET_PEEP': row['SET_PEEP'],
                'next_SET_PSUPP': row['SET_PSUPP'] + (i % 2),  # 增加支持压力
                'next_SET_VENTMODE': '5119' if i % 2 == 0 else '5120'  # 模式变化
            })
            demo_data.append(row)
        
        # 保存示例数据
        demo_df = pd.DataFrame(demo_data)
        demo_file = 'demo_data.csv'
        demo_df.to_csv(demo_file, index=False)
        print(f"\nDemo data saved to '{demo_file}'")
        
        # 如果模型存在，进行预测演示
        if os.path.exists(args.model):
            print("\nRunning prediction on demo data...")
            try:
                predictions = predict_from_csv(args.model, demo_file, 'demo_predictions.json')
                print("Demo predictions:")
                for pred in predictions:
                    print(f"  Sample {pred['id']}: {pred['task1_output']}")
            except Exception as e:
                print(f"Demo prediction failed: {e}")
        else:
            print(f"\nNo trained model found at '{args.model}'")
            print("Run 'python main.py train --data demo_data.csv' to train a model first.")

def print_usage():
    """打印使用说明"""
    print("""
Ventilator Deep Learning Model Usage:

1. Training:
   python main.py train --data your_data.csv
   
2. Prediction:
   python main.py predict --data new_data.csv --model model.pth --output predictions.json
   
3. Demo:
   python main.py demo
   
Data Format Requirements:
- CSV file with columns: Sex, Age, VTI, VTE, RATE, FSPN, MVSPN, PPEAK, PMEAN, PPLAT, MVLEAK, 
  Vent_Paw_Wave, Vent_Flow_Wave, Vent_Vol_Wave, SET_SIMVRR, SET_VENTMODE, SET_TRIGGERFLOW, 
  SET_OXYGEN, SET_PEEP, SET_PSUPP
- Plus corresponding next_ columns for prediction targets
- SET_VENTMODE should use the predefined mapping values

Example:
  python main.py train --data data.csv
  python main.py predict --data test_data.csv --model model.pth
    """)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()
    else:
        main()