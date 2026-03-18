# MolBench 使用教程

本教程将指导您如何使用 MolBench 框架进行分子机器学习基准测试。MolBench 提供了一个统一的平台，支持多种数据集、模型和评估指标，帮助您快速比较和评估分子预测模型的性能。

## ✨ 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [CLI 使用](#cli-使用)
4. [Python API 使用](#python-api-使用)
5. [配置文件](#配置文件)
6. [自定义模型](#自定义模型)
7. [结果分析](#结果分析)
8. [故障排除](#故障排除)

## 📦 安装

### 环境要求

- Python >= 3.10
- 推荐使用 conda 或 venv 创建虚拟环境

### 安装步骤

1. **克隆仓库**：
   ```bash
   git clone https://github.com/hfsy0503/molbench.git
   cd molbench
   ```

2. **创建虚拟环境**（推荐）：
   ```bash
   conda create -n molbench python=3.10
   conda activate molbench
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **安装 MolBench**：
   ```bash
   pip install -e .
   ```

5. **验证安装**：
   ```bash
   molbench --help
   ```

## 🚀 快速开始

让我们从一个简单的例子开始：使用 ChemBERTa2 模型在 ESOL 数据集上进行水溶性预测。

### CLI 方式

```bash
molbench --dataset ESOL --model ChemBERTa2 --output results/
```

### Python 方式

```python
import molbench

# 配置
config = {
    "dataset": "ESOL",
    "models": [
        {
            "name": "ChemBERTa2",
            "params": {}
        }
    ],
    "output_dir": "results/"
}

# 运行
results = molbench.run(config)
print("测试完成！结果保存在 results/ 目录中")
```

## 🧪 CLI 使用

MolBench 提供了强大的命令行接口，支持多种参数配置。

### 基本语法

```bash
molbench [OPTIONS]
```

### 常用选项

- `--dataset DATASET`: 指定数据集 (e.g., ESOL, FreeSolv, BBBP)
- `--model MODEL`: 指定模型 (e.g., ChemBERTa2, GCN, SVM)
- `--config CONFIG_FILE`: 使用配置文件
- `--output OUTPUT_DIR`: 输出目录
- `--seed SEED`: 随机种子
- `--verbose`: 详细输出

### 示例

1. **单模型单数据集**：
   ```bash
   molbench --dataset ESOL --model ChemBERTa2
   ```

2. **多模型比较**：
   ```bash
   molbench --dataset BBBP --model ChemBERTa2 GCN SVM
   ```

3. **使用配置文件**：
   ```bash
   molbench --config examples/config.yaml
   ```

4. **指定输出目录和随机种子**：
   ```bash
   molbench --dataset FreeSolv --model MPNN --output my_results/ --seed 42
   ```

## 📄 Python API 使用

对于更灵活的使用，MolBench 提供了 Python API。

### 基本导入

```python
import molbench
```

### 简单运行

```python
config = {
    "dataset": "ESOL",
    "models": [
        {
            "name": "ChemBERTa2"
        }
    ]
}

results = molbench.run(config)
```

### 高级配置

```python
config = {
    "dataset": "BBBP",
    "models": [
        {
            "name": "GCN",
            "params": {
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.2
            }
        },
        {
            "name": "GAT",
            "params": {
                "hidden_dim": 64,
                "num_heads": 4
            }
        }
    ],
    "split": {
        "type": "scaffold",
        "test_size": 0.2,
        "val_size": 0.1
    },
    "metrics": ["auc", "accuracy", "precision", "recall", "f1"],
    "output_dir": "results/",
    "seed": 42
}

results = molbench.run(config)
```

### 结果处理

```python
# results 是一个字典，包含所有模型的结果
for model_name, model_results in results.items():
    print(f"模型: {model_name}")
    print(f"测试集 AUC: {model_results['test']['auc']:.4f}")
    print(f"验证集准确率: {model_results['val']['accuracy']:.4f}")
    print("---")
```

## 📁 配置文件

对于复杂的实验，建议使用 YAML 配置文件。

### 示例配置文件

创建 `config.yaml`：

```yaml
dataset: ESOL
models:
  - name: ChemBERTa2
    params: {}
  - name: GCN
    params:
      hidden_dim: 128
      num_layers: 3
split:
  type: random
  test_size: 0.2
  val_size: 0.1
metrics:
  - rmse
  - mae
  - r2
output_dir: results/
seed: 42
```

### 运行配置文件

```bash
molbench --config config.yaml
```

或在 Python 中：

```python
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

results = molbench.run(config)
```

## 📊 自定义模型

MolBench 支持添加自定义模型。您需要实现一个适配器类。

### 模型适配器结构

```python
from molbench.core.adapters.base import BaseModelAdapter

class MyCustomModel(BaseModelAdapter):
    def __init__(self, **params):
        super().__init__(**params)
        # 初始化您的模型
        self.model = MyModel(**params)

    def fit(self, X, y):
        # 训练模型
        self.model.fit(X, y)

    def predict(self, X):
        # 预测
        return self.model.predict(X)

    def predict_proba(self, X):
        # 概率预测（分类任务）
        return self.model.predict_proba(X)
```

### 注册自定义模型

在您的代码中：

```python
from molbench.core.adapters import register_adapter

register_adapter('my_model', MyCustomModel)
```

然后在配置中使用：

```python
config = {
    "dataset": "ESOL",
    "models": [
        {
            "name": "my_model",
            "params": {
                "param1": value1,
                "param2": value2
            }
        }
    ]
}
```

## 📈 结果分析

MolBench 会自动生成详细的结果报告。

### 指标说明

- **回归任务**: RMSE, MAE, R²
- **分类任务**: AUC, Accuracy, Precision, Recall, F1

### 可视化

MolBench 自动生成各种图表，包括：
- 混淆矩阵
- ROC 曲线
- 预测 vs 真实值散点图
- 特征重要性图

## 🔧 故障排除

### 常见问题

1. **导入错误**：
   - 确保所有依赖都已安装
   - 检查 Python 版本 (>= 3.10)

2. **内存不足**：
   - 对于大型数据集，考虑使用更小的批次大小
   - 使用 `--batch_size` 参数

3. **模型训练失败**：
   - 检查模型参数是否正确
   - 查看详细错误日志 (`--verbose`)

4. **CUDA 相关错误**：
   - 确保 PyTorch 和 CUDA 版本兼容
   - 设置 `CUDA_VISIBLE_DEVICES` 环境变量

### 获取帮助

- 查看 [API 文档](api.md)
- 在 GitHub 上提交 issue
- 查看示例代码

---

恭喜！您已经完成了 MolBench 的基本使用教程。现在您可以开始使用 MolBench 进行分子机器学习的研究了。如有问题，请随时查看文档或提交 issue。