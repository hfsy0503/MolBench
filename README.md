molbench/
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── setup.py
├── requirements.txt
├── .gitignore
├── __init__.py 
├── __main__.py                 # molbench包入口点
├── cli.py                      # 统一运行入口
├── run_from_config.py          # 配置文件模式运行入口
├── configs/                    # 运行文件配置管理模块
│   ├── __init__.py
│   ├── cfg_generator.py        # 配置文件生成器
│   ├── cfg_parser.py           # 配置文件解析器
│   └── templates/ 
│       ├── basic.yaml          # 单模型评测配置文件
│       └── advanced.yaml       # 多模型评测配置文件
├── core/                       # 核心功能模块
│   ├── __init__.py
│   ├── runner_engine.py        # 评测核心引擎
│   ├── runner.py               # 交互模式运行入口
│   ├── adapters/               # 模型适配器
│   │   ├── __init__.py
│   │   ├── base.py             # 模型基类
│   │   ├── custom_model.py     # 自定义模型封装
│   │   ├── sklearn_adapter.py  # Sklearn模型封装
│   │   ├── gnn_adapter.py      # GNN模型封装
│   │   ├── smiles_to_graph.py  # 将SMILES转成Data对象
│   │   └── text_adapter.py     # Transformer模型封装
│   ├── data/                   # 数据处理
│   │   ├── __init__.py 
│   │   ├── datasets/           # 存储数据集文件
│   │   ├── data_loader.py      # 数据集加载+标准化
│   │   └── data_splitter.py    # 数据拆分策略
│   ├── featurizers/            # 特征化处理
│   │   ├── __init__.py         # 统一暴露接口
│   |   ├── base.py             # 特征化基类
│   |   ├── feat_registry.py    # 特征化方法注册
│   |   ├── ecfp.py             # ECFP分子指纹
│   |   ├── coulomb_matrix.py   # Coulomb Matrix描述符
│   |   ├── maccs.py            # MACCS keys指纹
│   |   ├── mol2vec.py          # Mol2vec无监督嵌入
│   |   ├── pcfp.py             # PubChem Fingerprint指纹
│   │   └── rdkit2d.py          # RDkit 2D描述符
│   ├── hyper_parameters/       # 模型超参数搜索配置文件
│   |   ├── regression_models/  # 传统回归模型配置文件
│   |   ├── binary_models/      # 传统分类模型配置文件
│   |   ├── new/                # 新兴模型配置文件
│   │   ├── test/               # 测试配置文件
│   │   └── configs_manager.py  # 模型配置文件调取与格式转化
│   ├── evaluation/             # 评估模块
│   │   ├── __init__.py 
│   │   ├── metrics.py          # 计算评估指标
│   │   └── visualization.py    # 结果可视化
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── bayesian_opt.py     # 贝叶斯优化
│       ├── model_register.py   # 模型登记
│       ├── model_selector.py   # 模型选择与加载
│       ├── cache.py            # 数据缓存
│       ├── logger.py           # 日志
│       └── error handler.py    # 错误处理
├── tests/                      # 测试文件
├── results/                    # 评测结果存储
├── examples/                   # 使用示例
└── docs/                       # 文档


