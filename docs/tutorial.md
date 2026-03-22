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

3. **安装 MolBench 及依赖**

   ```bash
    # 1. 先安装 PyTorch
    pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # 2. 再安装 PyTorch Geometric
    pip install torch-geometric

    # 3. 最后安装其他依赖
    pip install -r requirements.txt
   ```

4. **验证安装**：
   ```bash
   molbench --help
   ```

## 🚀 快速开始

让我们从简单的评测任务运行开始。

### 方式一：生成模板，然后运行

```bash
molbench -c my_config.yaml -o results/
```

### 方式二：使用 Python API

```python
import molbench

# 运行评测
results = molbench.run(
    df=df,                    # 包含 SMILES 和标签的 DataFrame
    smiles_col='smiles',      # SMILES 列名
    target_cols=['solubility'], # 目标列
    task_type='regression',   # 或 'binary'
    sklearn_models={'XGBoostRegressor': config}
)

print(results)
```

## 🧪 CLI 使用

MolBench 提供了强大的命令行接口，支持多种参数配置。

### 基本语法

```bash
molbench [OPTIONS]
```

### 常用选项

| 选项 | 说明 |
|------|------|
|-h, --help|	显示帮助信息|
|-t {basic,multi_model,advanced}, --template {basic,multi_model,advanced}|	查看指定模板的内容|
|-c CONFIG, --config CONFIG|	使用配置文件运行实验|
|-g {basic,multi_model,advanced}, --generate {basic,multi_model,advanced}|	生成配置模板|
|-o OUTPUT, --output OUTPUT	|输出目录|
|--list-templates|	列出所有可用的模板|

### 示例

1. **交互式运行（适合初次使用）**

```bash
python -m molbench
```

2. **采用内置模板运行**：
```bash
# 列出可用模板
molbench --list-templates

# 单模型快速评测
molbench -t basic

# 进阶：超参数优化
molbench -t advanced
```

3. **生成自定义配置文件**：
```bash
molbench -g advanced -o config.yaml
```

4. **运行实验**：
```bash
molbench -c config.yaml -o results/
```

## 📄 Python API 使用

MolBench 的核心是 `run_benchmark` 函数，提供了完整的评测流程。

### 基本导入

```python
import pandas as pd
from molbench.core import run_benchmark
```

### 准备数据
```python
# 加载数据（需包含 SMILES 列和目标列）
df = pd.read_csv('your_data.csv')
# 示例格式：
# | smiles | solubility |
# |--------|------------|
# | CCO    | -1.2       |
```

### 运行评测
```python
results = run_benchmark(
    # 数据参数
    df=df,
    file_base='experiment_1',
    smiles_col='smiles',
    feature_cols=['smiles'],      # 其他特征列（可选）
    target_cols=['solubility'],   # 支持多个目标列
    task_type='regression',       # 或 'binary'
    
    # 模型配置
    sklearn_models={
        'XGBoostRegressor': {
            "search_space": {
                "lr": {"type": "real", "bounds": [0.01, 0.3]},
                "max_depth": {"type": "int", "bounds": [3, 10]}
            },
            "fixed_params": {"n_estimators": 100}
        }
    },
    graph_models={
        'GCN': {
            "fixed_params": {"hidden_dim": 64, "num_layers": 3}
        }
    },
    text_models={
        'ChemBERTa-77M-MLM': {
            "fixed_params": {"model_path": "DeepChem/ChemBERTa-77M-MLM"}
        }
    },
    
    # 特征化参数（用于 sklearn 模型）
    featurizer_name='ecfp',      # 或 'morgan', 'rdkit', 'coulomb'
    featurizer_params={'radius': 2, 'nBits': 1024},
    
    # 评测参数
    split_method='random',       # 或 'scaffold'
    split_seed=42,
    n_iter=3,                    # 贝叶斯优化迭代次数
    cache_enabled=True,
    verbose=True
)

# 查看结果
print(results)
```

### 结果结构

```python
# results 是一个字典，包含所有模型的结果
for model_name, model_results in results.items():
    print(f"模型: {model_name}")
    print(f"测试集 ROC_AUC: {model_results['test_roc_auc']:.4f}")
    print(f"验证集 PRC_AUC: {model_results['val_pr_auc']:.4f}")
    print("---")
```

## 📁 配置文件

对于复杂的实验，建议使用 YAML 配置文件。

### 示例配置文件

创建 `config.yaml`：

```yaml
dataset:
  name: ESOL                    
  path: molbench/core/data/datasets/delaney-processed.csv
  task_type: regression
  smiles_col:
   - smiles
  target_cols:
   - ESOL predicted log solubility in mols per litre
  split:
    method: random
    train_ratio: 0.7
    val_ratio: 0.2
    test_ratio: 0.1
    seed: 42 

# 模型配置
models:
  - name: RandomForestRegressor
    type: sklearn
    protocol: sklearn
    
    hyperopt_config: test/random_forest_regressor.json
featurizer:
  name: ecfp                    
  params:
    radius: 2
    n_bits: 2048

# 贝叶斯优化配置
optimization:
  n_iter: 30  # 全局默认
  
# 评测配置
evaluation:
  extra_metrics:
    - Pearson_r
    - F1_macro
  
  # 可视化
  visualization: true
  save_plots: true
  output_dir: ./results

# 系统配置
system:
  cache: true                   # 启用磁盘缓存
  cache_dir: ./.molbench_cache
  verbose: true
  n_jobs: 1   
```

### 运行配置文件

```bash
molbench -c config.yaml -o results/
```

或在 Python 中：

```python
import yaml
from molbench.core import run_benchmark

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 需要将配置转换为 run_benchmark 接受的格式
results = run_benchmark(
    df=df,
    file_base=config.get('dataset', 'exp'),
    smiles_col=config['smiles_col'],
    target_cols=config['target_cols'],
    task_type=config['task_type'],
    sklearn_models=config.get('sklearn_models', {}),
    graph_models=config.get('graph_models', {}),
    text_models=config.get('text_models', {}),
    featurizer_name=config.get('featurizer', 'ecfp'),
    featurizer_params=config.get('featurizer_params', {}),
    split_method=config.get('split_method', 'random'),
    split_seed=config.get('split_seed', 42),
    n_iter=config.get('n_iter', 3)
)
```

## 📊 自定义模型

MolBench 支持添加自定义模型。您需要实现一个适配器类。

### 模型适配器结构

```python
from molbench.core import BenchModel

# 编写你自己的模型适配器：
class CustomModel(BenchModel):          # 可自定义类名
    """用户自建模型"""
    def __init__(self, lr=0.01, n_estimators=100):   # 写入超参
        self.params = dict(lr=lr, n_estimators=n_estimators)
        # 模型实例化(此处为举例)
        from sklearn.ensemble import AdaBoostRegressor
        self.model = AdaBoostRegressor(learning_rate=lr, n_estimators=n_estimators)

    def fit(self, X, y): 
        self.model.fit(X, y)
        return self

    def predict(self, X): 
        return self.model.predict(X)

    def get_params(self, deep=True): 
        return self.params.copy()

    def get_task_type(self): 
        return 'regression'   
```

### 注册自定义模型

在您的代码中：

```python
from molbench.core import register_model

register_model(CustomModel, task_type='regression',
                save_dir=r'molbench\core\hyper_parameters\test', # 可自行更改配置文件保存地址
                protocol='bench')
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

1. **导入错误**：`ModuleNotFoundError`
   - 确保所有依赖都已安装：`pip install -r requirements.txt`
   - 检查 Python 版本 (>= 3.10)

2. **内存不足**：
   - 减小 `batch_size` 参数
   - 使用更简单的模型或更小的特征维度

3. **模型训练失败**：
   - 检查实际任务 `task_type` 与模型json配置文件中的 `task_type` 是否匹配
   - 检查模型参数是否正确

4. **数据加载失败**：
   - 检查 CSV 文件格式
   - 确认 SMILES 列包含有效分子

### 获取帮助

- 查看 [API 文档](api.md)
- 运行 `molbench --help` 查看 CLI 帮助
- 查看示例配置：`molbench --list-templates`
- 在 GitHub 上提交 issue

---

恭喜！您已经完成了 MolBench 的基本使用教程。现在您可以开始使用 MolBench 进行分子机器学习的研究了。如有问题，请随时查看文档或提交 issue。