# 深度学习呼吸机参数预测项目

这是一个基于 PyTorch 的深度学习项目，用于预测呼吸机的下一时刻参数设置。

## 项目结构

```
DL/
├── src/
│   ├── __init__.py          # 包初始化文件
│   ├── config.py            # 配置参数
│   ├── data_loader.py       # 数据加载和预处理
│   ├── model.py             # 神经网络模型
│   ├── predict.py           # 预测模块
│   ├── train.py             # 训练模块
│   └── utils.py             # 工具函数
├── main.py                  # 主程序入口
└── README.md               # 项目说明
```

## 功能特性

- **多输出预测**: 同时预测5个数值型参数和1个分类型参数
- **特征工程**: 自动提取变化趋势特征
- **数据标准化**: 对输入特征和目标变量进行标准化
- **模型架构**: 使用多层感知机(MLP)实现多任务学习
- **损失函数**: 结合MSE损失和交叉熵损失的混合损失函数

## 环境要求

```bash
pip install torch pandas scikit-learn numpy tensorboard
```

## 数据格式

输入CSV文件应包含以下列：
- 基础特征：Sex, Age, VTI, VTE, RATE, FSPN, MVSPN, PPEAK, PMEAN, PPLAT, MVLEAK, Vent_Paw_Wave, Vent_Flow_Wave, Vent_Vol_Wave
- 当前设置：SET_SIMVRR, SET_VENTMODE, SET_TRIGGERFLOW, SET_OXYGEN, SET_PEEP, SET_PSUPP
- 下一时刻设置：next_SET_SIMVRR, next_SET_VENTMODE, next_SET_TRIGGERFLOW, next_SET_OXYGEN, next_SET_PEEP, next_SET_PSUPP

## 使用方法

### 训练模型

```bash
python main.py train --data E:\aaaa_ubuntu_code\huxiji_vla\data_dl\filtered_lsj_data.csv
```

### 预测

```bash
python main.py predict --data E:\aaaa_ubuntu_code\huxiji_vla\data_dl\filtered_lsj_data.csv --model E:\aaaa_ubuntu_code\huxiji_vla\our_data_work_v0.1\DL_DL\DL_task2\model_out\model.pth --output E:\aaaa_ubuntu_code\huxiji_vla\our_data_work_v0.1\DL_DL\DL_task2\model_out\predict/predictions.json
```

## 模型架构

模型采用多层感知机架构：
- 输入层：26个特征（20个原始特征 + 6个趋势特征）
- 隐藏层：256 → 128 → 64，使用ReLU激活函数和Dropout正则化
- 输出层：
  - 数值型输出：5个参数的回归预测
  - 分类型输出：SET_VENTMODE的分类预测（11个类别）

## 数据分割

数据按时间顺序分割：
- 训练集：50%
- 测试集：20%
- 验证集：30%

## 输出格式

预测结果保存为JSON格式：

```json
[
  {
    "id": 0,
    "task2_output": "{\"SET_SIMVRR\": \"10\", \"SET_VENTMODE\": \"5120\", \"SET_TRIGGERFLOW\": \"2.0\", \"SET_OXYGEN\": \"40\", \"SET_PEEP\": \"4\", \"SET_PSUPP\": \"10\"}"
  }
]
```

## 配置参数

在 `src/config.py` 中可以调整以下参数：
- 模型架构参数：隐藏层大小、Dropout率
- 训练参数：学习率、批次大小、训练轮数
- 数据分割比例
- 设备配置（CPU/GPU）

## 监控训练

训练过程使用TensorBoard记录：

```bash
tensorboard --logdir runs/
```

## 注意事项

1. 确保CSV文件包含所有必需的列
2. SET_VENTMODE字段会自动映射为整数
3. 模型会自动计算变化趋势特征
4. 预测时需要提供训练好的模型文件

## 扩展功能

- 支持早停机制
- 模型检查点保存
- 详细的训练日志
- 可配置的损失函数权重