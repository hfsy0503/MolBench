# MolBench API 文档

本文档详细介绍了 MolBench 框架的 API 接口。如果您是第一次使用 MolBench，建议先阅读[教程](tutorial.md)了解基本概念和使用方法。

## 目录

1. [核心 API](#核心-api)
2. [命令行接口](#命令行接口)
3. [模型适配器](#模型适配器)
4. [数据处理](#数据处理)
5. [特征化](#特征化)
6. [评估指标](#评估指标)
7. [配置格式](#配置格式)
8. [工具函数](#工具函数)

## 核心 API

### molbench.run(config)

运行基准测试的主要函数。

**参数：**
- `config` (dict): 配置字典，包含数据集、模型、评估等设置

**返回值：**
- `dict`: 包含所有模型结果（results, best models, task_type）的字典

**示例：**

```python
import molbench

config = {
    "dataset": "ESOL",
    "models": [{"name": "ChemBERTa2"}],
    "output_dir": "results/"
}

results = molbench.run(config)
```

详细配置选项请参考[配置格式](#配置格式)部分。

## 命令行接口

### molbench 命令

MolBench 的主要命令行入口。

**基本语法：**
```bash
molbench [OPTIONS]
```

**主要选项：**

| 选项 | 类型 | 描述 | 示例 |
|------|------|------|------|
| `-t, --template` | str | 使用内置模板 | `molbench -t basic` |
| `-c, --config` | path | 指定配置文件 | `molbench -c config.yaml` |
| `-g, --generate-config` | str | 生成配置模板 | `molbench -g basic` |
| `-o, --output` | path | 输出路径（仅生成配置时可使用） | `molbench -g basic -o my_config.yaml` |
| `--list-templates` | flag | 列出可用模板 | `molbench --list-templates` |

**示例：**

```bash
# 使用内置模板
molbench --template basic

# 使用配置文件
molbench --config examples/config.yaml

# 生成配置模板
molbench --generate-config advanced --output my_config.yaml
```

## 模型适配器

MolBench 提供了以下内置模型适配器：

| 基类/适配器 | 功能 |
| -------------------- |------------------------------------------------------------- |
| `BenchModel`         | 所有模型适配器的抽象基类，定义统一接口（fit, predict, predict\_proba, save, load） |
| `SklearnModel`     | scikit-learn 模型封装|
| `BenchGNN`         | 图神经网络模型封装（PyTorch Geometric）|
| `TextModel`,`HFTestModel`,`DeepChemTextCNN`     | Transformer 文本模型封装（HuggingFace）|
| `CustomModel` | 自定义模型封装 |
**统一接口:** `fit()`,`predict()`,`predict_proba()`,`save()`,`load()`

### 自定义模型适配器

要添加自定义模型，请继承 `BenchModel` 类：

```python
from molbench.core.adapters.base import BenchModel

class CustomModel(BenchModel):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = MyModel(**params)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_task_type(self):
        return 'regressor'  # 或 'binary', 'multiclass'
```

然后注册生成对应的模型配置文件（.json）：

```python
from molbench.core import register_model
from my_module import CustomModel

register_model(
    CustomModel, 
    task_type='binary',
    save_dir='molbench/core/hyper_parameters/test',
    search_space={'lr': {'type': 'real', 'bounds': [1e-5, 1e-3]}}
)
```

## 数据处理

### 数据加载

| 函数 | 功能 | 交互方式 |
|-----|-------------------------------------------------------|------|
| `load_dataset(source)` | 从文件加载数据 | 非交互，支持 CSV/TSV/Excel |
| `load_file()` | 交互式加载数据 | 交互式 |
| `select_task_columns(df)` | 交互式选择列 | 交互式 |
| `load_data(df, task_type, feature_col, target_col)` | 数据预处理 | 非交互 |


### 数据标准化
| 函数 | 功能 |
|------|------|
| `standardization(X_train, X_val, X_test)` | 使用StandardScaler对特征进行标准化 |

### 数据分割
| 函数 | 功能 | 适用场景 |
|----|------|----------|
| `split_data(df,…)` | 选择分割策略 | 交互模式 |
| `split_data_non_interactive(df,…)` | 指定分割策略 | 配置文件 |
| `random_split(df,…)` | 随机分割 | 通用 |
| `scaffold_split(df,…)` | 基于分子骨架分割 | 化学数据 |
| `stratified_split(df,…)` | 分层分割 | 确保每个集合包含所有提供的标签范围 |
| `time_split(df,…)` | 时序分割 | 时序数据 |
| `stratified_scaffold_split(df,…)` | 分层+骨架结合分割 | 需兼顾结构与分布的任务 |

### 使用模式
| 场景      | 调用链 |
| ------- |------------------------------------------------------------------------------ |
| 交互式 CLI | `load_file()` → `select_task_columns()` → `split_data()` → `standardization()` |
| 非交互脚本   | `load_dataset()` → `split_data_non_interactive()` → `standardization()`|


## 特征化

| 函数 | 功能 |
| ------------------------------------------- |---------- |
| `get_featurizer(name, **params)`            | 获取指定特征化器实例 |
| `cached_transform(featurizer, smiles_list)` | 带缓存的特征转换   |

### 支持的特征化方法
| 名称                | 说明                  |
| ----------------- | ------------------- |
| `ecfp` / `morgan` | Morgan 分子指纹（ECFP）   |
| `rdkit2d`         | RDKit 2D 分子描述符      |
| `maccs`           | MACCS 结构指纹          |
| `mol2vec`         | Mol2Vec 无监督嵌入       |
| `coulomb_matrix`  | Coulomb Matrix 描述符  |
| `pcfp`            | PubChem Fingerprint |
| `graph`           | 图结构特征（用于 GNN）       |


## 评估模块

| 任务类型  | 指标 |
| ---------- | ---------------------------------------------- |
| **回归** | `r2`, `rmse`, `mae`                            |
| **分类** | `roc-auc`, `prc-auc`, `precision` |

### 评估工具
| 函数/类 | 功能 |
| ---------------------------------------- | ------------------------------------------------------ |
| `get_score(model, X)`                    | 获取模型预测分数（支持 probability/decision\_function/predict 回退） |
| `get_calibrated_models(best_models)`     | 对离散型分类器进行概率校准                                          |
| `evaluation(task_type, extra_metrics)`  | 评估类，计算任务相关指标|

## 可视化模块
| 函数/类 | 功能 |
| ---------------------------------------- | ------------------------------------------------------ |
| `visualizer` | 可视化类，支持回归/二分类/多分类 |
| `plot(...)`| 绘制回归散点图、分类 ROC-AUC 曲线  |
| `compare_models(...)` | 一键生成对比图表（雷达图、柱状图等） |

## 配置格式

MolBench 使用 YAML 格式的配置文件。配置字典包含以下主要部分：

### 基本配置

```yaml
dataset: ESOL                    # 数据集名称
output_dir: results/             # 输出目录
seed: 42                        # 随机种子
verbose: true                   # 详细输出
```

### 模型配置

```yaml
models:
  - name: ChemBERTa-77M-MTR     # 模型名称
    params: {}                  # 模型参数
  - name: GCN
    params:
      hidden_dim: 128
      num_layers: 3
      dropout: 0.2
```

### 数据分割配置

```yaml
split:
  type: random                  # 分割类型: random/scaffold
  test_size: 0.2               # 测试集比例
  val_size: 0.1                # 验证集比例
```

### 特征化配置

```yaml
featurizer:
  name: ecfp                  # 特征化器名称
  params:
    radius: 2
    n_bits: 2048
```

### 评估配置

```yaml
metrics:
  - rmse
  - mae
  - r2
```

### 超参数优化配置

```yaml
optimization:
  enabled: true
  method: bayesian              # 优化方法
  n_iter: 30                    # 迭代次数
  param_space:                  # 参数空间
    hidden_dim: [64, 128, 256]
    dropout: [0.1, 0.2, 0.3]
```
> 注：超参数配置也可通过 `hyper_parameters/` 目录下的 JSON 文件管理

## 工具函数/类

| 类别  | 函数/类  | 功能   |
| :-------- | :---------------------------------------------- | :----------- |
| **缓存**    | `cached_transform(X, task_id, transform_fn, **params)` | sklearn 特征缓存 |
|           | `CachedGraphConverter` | GNN 图数据缓存包装器 |
|           | `enable_graph_cache(model, task_id)`  | 为 GNN 启用图缓存 |
|           | `disable_graph_cache(model)`| 禁用图缓存 |
|           | `clear_cache()` | 清空所有缓存 |
|           | `cache_stats()` | 显示缓存统计       |
| **模型管理**  | `UnifiedModelSelector`  | 模型配置加载与选择  |
|           | `register_model(cls, task_type, save_dir,...)`| 注册自定义模型到框架 |
| **贝叶斯优化** | `optimization(X_train, X_val,...)` | 超参数贝叶斯优化 |

## 示例

完整的使用示例请参考[教程](tutorial.md)中的代码示例。

## 注意事项

1. 所有路径参数都支持相对路径和绝对路径
2. 配置文件中的参数会覆盖默认值
3. 模型参数应与具体模型适配器兼容
4. 大数据集建议启用缓存以提高性能

如有疑问，请查看[教程](tutorial.md)或提交 GitHub issue。