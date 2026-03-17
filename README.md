# MolBench - 分子机器学习基准测试框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

MolBench 是一个专为分子机器学习设计的基准测试框架，支持回归、二分类和多分类任务，提供统一的数据集、模型适配器、超参数优化、评估指标与结果可视化功能。

## 📁 项目结构
```plain
MolBench/
├── LICENSE                    
├── MANIFEST.in                
├── setup.py                   
├── README.md                  
├── requirements.txt           
├── .gitignore                 
├── molbench/                  
│   ├── __init__.py           
│   ├── __main__.py           
│   ├── cli.py                # 命令行接口
│   ├── run_from_config.py    # 配置文件模式入口
│   ├── configs/              # 配置管理
│   ├── core/                 # 核心功能
│   │   ├── runner_engine.py  # 评测主引擎
│   │   ├── runner.py         # 交互模式运行器
│   │   ├── adapters/         # 模型适配器
│   │   ├── data/             # 数据加载与拆分
│   │   ├── featurizers/      # 分子特征化方法
│   │   ├── hyper_parameters/ # 模型超参数配置
│   │   ├── evaluation/       # 评估指标与可视化
│   │   └── utils/            # 工具函数（优化/缓存/日志）
│   └── results/              # 评测结果存储
├── tests/                    
└── docs/                     
```
<<<<<<< HEAD


## ✨特性

- **多任务支持**：回归（r²）、二分类/多分类（ROC-AUC、Accuracy）
- **多模态输入**：SMILES 字符串 + 可选表格特征
- **丰富特征化**：ECFP、MACCS、Mol2Vec、RDKit 2D 描述符、Coulomb Matrix 等
- **模型适配器**：统一接口支持 sklearn、XGBoost、GNN（PyG）、Transformer（HuggingFace）
- **智能优化**：贝叶斯超参数搜索
- **灵活配置**：YAML/JSON 配置文件驱动，支持交互式与批量模式

## 📦安装

### 环境要求

- Python >= 3.10
- 支持的操作系统：Linux, macOS, Windows

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/hfsy0503/molbench.git
   cd molbench
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 安装 MolBench：
   ```bash
   pip install -e .
   ```

## 🧪 使用方法

### 命令行使用

#### 交互模式

```bash
molbench -m molbench
```

#### 配置文件模式
```bash
molbench --config config.yaml
```

#### Python API
```python
import molbench

config = {
    "dataset": "ESOL",
    "task_type": "regression",
    "models": [
        {"name": "RandomForestRegressor"},
        {"name": "XGBRegressor"}
    ],
    "featurizer": "ecfp",
    "optimization": {
        "n_calls": 50,
        "cv_folds": 5
    }
}
results = molbench.run(config)
print(results.summary())
```


## 📊 支持的数据集

- 物理化学：ESOL、Freesolv、Lipophilicity
- 量子化学：QM7、QM8、QM9
- 生理毒性：Tox21、ToxCast、ClinTox、BBBP、SIDER
- 生物物理：HIV、BACE、MUV

## 🚀 支持的模型

### 传统机器学习（sklearn + XGBoost + LGBM）
- 回归（30+）：RandomForest、XGBRegressor、SVR、LGBM 等
- 分类（20+）：LogisticRegression、XGBClassifier、SVC 等
### 图神经网络（PyTorch Geometric）
- GCN、GAT、GraphSAGE、GIN、MPNN、SchNet 等
### 预训练语言模型（Transformers）
- ChemBERTa-77M-MTR
- ChemBERTa-77M-MLM
- MoLFormer


## 🤝 贡献

欢迎贡献！请查看我们的贡献指南 CONTRIBUTING.md 了解详情。



## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📚 引用

如果您在研究中使用 MolBench，请引用：

```
@software{molbench2026,
  title={MolBench: A Benchmarking Framework for Molecular Machine Learning},
  author={MolBench Contributors},
  url={https://github.com/hfsy0503/molbench},
  year={2026}
}
```

## 🙏 联系方式

- 项目主页: https://github.com/hfsy0503/molbench
- 问题反馈: https://github.com/hfsy0503/molbench/issues
- 邮箱: 2310753@mail.nankai.edu.cn
=======
>>>>>>> 769821a (Inital README)


## ✨特性

- **多任务支持**：回归（r²）、二分类/多分类（ROC-AUC、Accuracy）
- **多模态输入**：SMILES 字符串 + 可选表格特征
- **丰富特征化**：ECFP、MACCS、Mol2Vec、RDKit 2D 描述符、Coulomb Matrix 等
- **模型适配器**：统一接口支持 sklearn、XGBoost、GNN（PyG）、Transformer（HuggingFace）
- **智能优化**：贝叶斯超参数搜索
- **灵活配置**：YAML/JSON 配置文件驱动，支持交互式与批量模式

## 📦安装

### 环境要求

- Python >= 3.10
- 支持的操作系统：Linux, macOS, Windows

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/hfsy0503/molbench.git
   cd molbench
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 安装 MolBench：
   ```bash
   pip install -e .
   ```

## 🧪 使用方法

### 命令行使用

#### 交互模式

```bash
molbench -m molbench
```

#### 配置文件模式
```bash
molbench --config config.yaml
```

#### Python API
```python
import molbench

config = {
    "dataset": "ESOL",
    "task_type": "regression",
    "models": [
        {"name": "RandomForestRegressor"},
        {"name": "XGBRegressor"}
    ],
    "featurizer": "ecfp",
    "optimization": {
        "n_calls": 50,
        "cv_folds": 5
    }
}
results = molbench.run(config)
print(results.summary())
```


## 📊 支持的数据集

- 物理化学：ESOL、Freesolv、Lipophilicity
- 量子化学：QM7、QM8、QM9
- 生理毒性：Tox21、ToxCast、ClinTox、BBBP、SIDER
- 生物物理：HIV、BACE、MUV

## 🚀 支持的模型

### 传统机器学习（sklearn + XGBoost + LGBM）
- 回归（30+）：RandomForest、XGBRegressor、SVR、LGBM 等
- 分类（20+）：LogisticRegression、XGBClassifier、SVC 等
### 图神经网络（PyTorch Geometric）
- GCN、GAT、GraphSAGE、GIN、MPNN、SchNet 等
### 预训练语言模型（Transformers）
- ChemBERTa-77M-MTR
- ChemBERTa-77M-MLM
- MoLFormer


## 🤝 贡献

欢迎贡献！请查看我们的贡献指南 CONTRIBUTING.md 了解详情。



## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📚 引用

如果您在研究中使用 MolBench，请引用：

```
@software{molbench2026,
  title={MolBench: A Benchmarking Framework for Molecular Machine Learning},
  author={MolBench Contributors},
  url={https://github.com/hfsy0503/molbench},
  year={2026}
}
```

## 🙏 联系方式

- 项目主页: https://github.com/hfsy0503/molbench
- 问题反馈: https://github.com/hfsy0503/molbench/issues
- 邮箱: 2310753@mail.nankai.edu.cn

