# 深度学习呼吸机数据分析项目

本项目使用 PyTorch 构建深度学习模型，用于预测呼吸机设置参数的变化。

## 项目结构

```
DL/
├── src/
│   ├── __init__.py          # 包初始化文件
│   ├── config.py            # 配置参数
│   ├── data_loader.py       # 数据加载和预处理
│   ├── model.py             # 模型定义
│   ├── predict.py           # 预测模块
│   ├── train.py             # 训练模块
│   └── utils.py             # 工具函数
├── main.py                  # 主程序入口
└── README.md               # 项目说明
```

## 功能特性

- **多任务学习**：同时预测5个数值参数变化和1个模式变化
- **数据预处理**：自动处理CSV数据，包括特征标准化和标签编码
- **模型训练**：支持早停、学习率调度等训练技巧
- **预测推理**：加载训练好的模型对新数据进行预测
- **结果输出**：预测结果保存为指定格式的JSON文件

## 数据格式

### 输入特征
模型使用以下字段作为输入特征（非`next_`开头的列）：
- `Sex`, `Age`, `VTI`, `VTE`, `RATE`, `FSPN`, `MVSPN`, `PPEAK`, `PMEAN`, `PPLAT`, `MVLEAK`
- `Vent_Paw_Wave`, `Vent_Flow_Wave`, `Vent_Vol_Wave`
- `SET_SIMVRR`, `SET_VENTMODE`, `SET_TRIGGERFLOW`, `SET_OXYGEN`, `SET_PEEP`, `SET_PSUPP`

### 预测目标
模型预测以下字段的变化：
1. **数值变化字段**（输出：`increase`, `decrease`, `no change`）：
   - `SET_SIMVRR`
   - `SET_TRIGGERFLOW` 
   - `SET_OXYGEN`
   - `SET_PEEP`
   - `SET_PSUPP`

2. **模式变化字段**（输出：`changed`, `no change`）：
   - `SET_VENTMODE`

### SET_VENTMODE 映射表
```python
{
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
```

### CSV 示例
```csv
Sex,Age,VTI,VTE,RATE,FSPN,MVSPN,PPEAK,PMEAN,PPLAT,MVLEAK,Vent_Paw_Wave,Vent_Flow_Wave,Vent_Vol_Wave,SET_SIMVRR,next_SET_SIMVRR,SET_VENTMODE,next_SET_VENTMODE,SET_TRIGGERFLOW,next_SET_TRIGGERFLOW,SET_OXYGEN,next_SET_OXYGEN,SET_PEEP,next_SET_PEEP,SET_PSUPP,next_SET_PSUPP
1,80,514.58,525.16,15.25,4.58,1.31,17.0,8.88,17.0,0.0,8.50,-0.11,175.66,10,10,5119,5119,2.0,2.0,30,30,5,5,12,12
```

## 安装依赖

```bash
pip install torch pandas numpy scikit-learn tqdm
```

## 使用方法

### 1. 训练模型

```bash
python main.py train --data E:\aaaa_ubuntu_code\huxiji_vla\data_dl\filtered_lsj_data.csv
```

训练参数说明：
- 数据集划分：50% 训练集，20% 测试集，30% 验证集
- 批量大小：32
- 学习率：0.001
- 最大轮数：100（支持早停）

### 2. 预测

```bash
python main.py predict --data E:\aaaa_ubuntu_code\huxiji_vla\data_dl\filtered_lsj_data.csv --output E:\aaaa_ubuntu_code\huxiji_vla\our_data_work_v0.1\DL\model_out\predictions.json --model E:\aaaa_ubuntu_code\huxiji_vla\our_data_work_v0.1\DL\model_out\model.pth
```

### 3. 运行演示

```bash
python main.py demo
```

演示模式会创建示例数据并展示完整的训练和预测流程。

## 输出格式

预测结果保存为JSON格式：

```json
{
  "id": 0,
  "task1_output": "SET_SIMVRR: no change; SET_TRIGGERFLOW: no change; SET_OXYGEN: decrease; SET_PEEP: no change; SET_PSUPP: no change; SET_VENTMODE: no change"
}
```

## 模型架构

使用多层感知机（MLP）实现多任务学习：

- **输入层**：20个特征
- **隐藏层**：128 → 64 → 32
- **输出层**：
  - 5个三分类器（数值变化预测）
  - 1个二分类器（模式变化预测）
- **正则化**：Dropout + Batch Normalization

## 配置说明

主要配置参数在 `src/config.py` 中：

```python
class Config:
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # 模型参数
    HIDDEN_DIM = 128
    
    # 数据划分
    TRAIN_RATIO = 0.5
    TEST_RATIO = 0.2
    VAL_RATIO = 0.3
```

## 注意事项

1. **数据质量**：确保CSV文件包含所有必需的列
2. **内存使用**：大数据集可能需要调整批量大小
3. **设备选择**：自动检测GPU可用性，优先使用CUDA
4. **模型保存**：训练完成后模型自动保存为`model.pth`

## 故障排除

### 常见问题

1. **"Missing required columns" 错误**
   - 检查CSV文件是否包含所有必需的列
   - 确认列名拼写正确

2. **内存不足错误**
   - 减小 `BATCH_SIZE` 参数
   - 使用更少的隐藏层节点

3. **模型收敛慢**
   - 调整学习率
   - 检查数据预处理是否正确
   - 增加训练轮数

### 性能优化

1. **训练速度**：
   - 使用GPU加速
   - 增大批量大小（在内存允许范围内）

2. **预测精度**：
   - 增加训练数据量
   - 调整模型架构
   - 使用交叉验证

## 扩展功能

项目支持以下扩展：

1. **自定义模型架构**：修改 `src/model.py`
2. **新增预测字段**：更新配置文件和数据处理逻辑
3. **不同的损失函数**：在 `MultiTaskLoss` 类中实现
4. **数据增强**：在 `DataProcessor` 中添加增强方法

## 许可证

本项目遵循 MIT 许可证。


# predict
``` python
python main.py predict --data E:\aaaa_ubuntu_code\huxiji_vla\data_dl\filtered_lsj_data.csv --output E:\aaaa_ubuntu_code\huxiji_vla\our_data_work_v0.1\DL\model_out\predictions.json --model E:\aaaa_ubuntu_code\huxiji_vla\our_data_work_v0.1\DL\model_out\model.pth
```

# train
``` python
python main.py train --data E:\aaaa_ubuntu_code\huxiji_vla\data_dl\filtered_lsj_data.csv
```