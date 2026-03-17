"""
molbench/configs/parser.py - 配置解析为 runner_engine 变量
"""
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# hyper_parameters 目录
HYPERPARAM_DIR = Path(__file__).parent.parent / "core" / "hyper_parameters"

class ConfigParser:
    """配置解析器：YAML -> runner_engine 所需变量"""
    
    @classmethod
    def load_for_runner(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载配置并解析为 runner_engine.run_benchmark() 需要的所有变量
        
        Returns
        -------
        dict
            包含所有必需参数的字典
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必需字段
        cls._validate(config)
        
        # 解析为 runner_engine 变量
        runner_vars = {}
        
        # 数据集相关
        ds_config = config.get('dataset', {})
        runner_vars['file_base'] = ds_config.get('name', 'unknown')
        runner_vars['task_type'] = ds_config.get('task_type', 'regression')
        
        # 数据加载（路径处理）
        data_path = ds_config.get('path')
        if data_path:
            # 统一路径分隔符
            data_path = Path(str(data_path).replace('\\', '/'))
            # 如果是相对路径，基于配置文件或项目根目录解析
            if not data_path.is_absolute():
                # 尝试多种基础路径
                bases = [
                    config_path.parent,  # 配置文件所在目录
                    Path.cwd(),          # 当前工作目录
                    HYPERPARAM_DIR.parent.parent,  # molbench 项目根目录
                ]
                for base in bases:
                    full_path = base / data_path
                    if full_path.exists():
                        data_path = full_path
                        break
        
        runner_vars['data_path'] = str(data_path) if data_path else None
        
        # 列配置
        runner_vars['smiles_col'] = 'smiles'  # 默认，可从配置扩展
        runner_vars['feature_cols'] = []      # 默认空，从数据推断
        runner_vars['target_cols'] = ds_config.get('target_cols', [])
        
        # 如果没有指定 target_cols，自动检测（单任务）
        if not runner_vars['target_cols']:
            # 在 load_dataset 后填充
            runner_vars['target_cols'] = None  # 标记为待推断
        
        # 划分配置
        split_config = ds_config.get('split', {})
        runner_vars['split_method'] = split_config.get('method', 'random')
        runner_vars['split_seed'] = split_config.get('seed', 42)
        # train/val/test ratio 在 split_data 中使用
        
        # 贝叶斯优化配置
        opt_config = config.get('optimization',{})
        global_n_iter = opt_config.get('n_iter', 30)
        runner_vars['n_iter'] = global_n_iter

        # 模型配置
        models_config = config.get('models', [])
        json_configs = cls._build_json_configs(models_config, global_n_iter)
        
        # 分类模型（与 runner.py 相同逻辑）
        runner_vars['text_models'] = {}
        runner_vars['graph_models'] = {}
        runner_vars['sklearn_models'] = {}
        
        TEXT_MODELS = ['HFTextModel', 'DeepChemTextCNN']
        GRAPH_MODELS = ['BenchGNN', 'GCN', 'GAT', 'GIN', 'MPNN',
                        'GraphConv','TransformerConv', 'GraphSAGE']
        
        for name, cfg in json_configs.items():
            model_type = cfg.get('model', '')
            if any(t in str(model_type) for t in TEXT_MODELS):
                runner_vars['text_models'][name] = cfg
            elif any(t in model_type for t in GRAPH_MODELS):
                runner_vars['graph_models'][name] = cfg
            else:
                runner_vars['sklearn_models'][name] = cfg
        
        # 特征化配置
        featurizer_config = config.get('featurizer', {})
        runner_vars['featurizer_name'] = featurizer_config.get('name', 'ecfp')
        runner_vars['featurizer_params'] = featurizer_config.get('params', {})

        # 评测配置
        eval_config = config.get('evaluation', {})
        runner_vars['extra_metrics'] = eval_config.get('extra_metrics', False)
        runner_vars['output_dir'] = eval_config.get('output_dir', './results')
        runner_vars['visualization'] = eval_config.get('visualization', True)
        
        # 系统配置
        sys_config = config.get('system', {})
        runner_vars['cache_enabled'] = sys_config.get('cache', True)
        runner_vars['verbose'] = sys_config.get('verbose', True)
        runner_vars['n_jobs'] = sys_config.get('n_jobs', 1)
        
        return runner_vars
    
    @classmethod
    def _validate(cls, config: Dict[str, Any]) -> None:
        """验证配置完整性"""
        if 'dataset' not in config:
            raise ValueError("配置缺少 'dataset' 部分")
        
        if 'models' not in config or not config['models']:
            raise ValueError("配置缺少 'models' 或为空列表")
        
        for i, model in enumerate(config['models']):
            if 'name' not in model:
                raise ValueError(f"第 {i+1} 个模型缺少 'name'")
            if 'hyperopt_config' in model and 'hyperopt' in model:
                raise ValueError(f"模型 {model['name']}: 不能同时指定 hyperopt_config 和 hyperopt")
    
    @classmethod
    def _build_json_configs(cls, models: List[Dict], global_n_iter: int = 30) -> Dict[str, Dict]:
        """
        构建 optimization 需要的 json_configs 格式
        """
        json_configs = {}
        
        for model in models:
            name = model['name']
            cfg = {
                'model': name,
                'type': model.get('type', 'sklearn'),
                'protocol': model.get('protocol', 'sklearn'),
            }
            
            # 处理超参配置
            model_n_iter = model.get('n_iter', None) # 模型特定的 n_iter
            if 'hyperopt_config' in model:
                # 已有外部 JSON，读取并引用
                hyperopt = cls._load_hyperopt_json(model['hyperopt_config'])
                cfg['fixed_params'] = hyperopt.get('fixed_params', {}) or {}
                cfg['skopt_space'] = hyperopt.get('skopt_space') or []
                cfg['n_iter'] = model_n_iter or hyperopt.get('n_iter') or global_n_iter
            elif 'hyperopt' in model:
                # 内联配置
                hyperopt = model['hyperopt']
                cfg['fixed_params'] = hyperopt.get('fixed_params', {}) or {}
                cfg['skopt_space'] = hyperopt.get('space') or []
                cfg['n_iter'] = model_n_iter or hyperopt.get('n_iter') or global_n_iter
            else:
                # 无超参优化，使用默认参数
                cfg['fixed_params'] = model.get('params', {}) or {}
                cfg['skopt_space'] = []
                cfg['n_iter'] = 1
            
            json_configs[name] = cfg
        
        return json_configs
    
    @classmethod
    def _load_hyperopt_json(cls, json_path: str) -> Dict[str, Any]:
        """加载 hyper_parameters/ 下的 JSON"""
        if not json_path.endswith('.json'):
            json_path += '.json'
        
        # 尝试多个位置
        paths_to_try = [
            HYPERPARAM_DIR / json_path,
            HYPERPARAM_DIR / 'test' / json_path,
            HYPERPARAM_DIR / 'regression_models' / json_path,
            HYPERPARAM_DIR / 'binary_models' / json_path,
        ]
        
        for full_path in paths_to_try:
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        raise FileNotFoundError(
            f"超参配置未找到: {json_path}\n"
            f"搜索路径: {[str(p) for p in paths_to_try]}"
        )
    
    @classmethod
    def load(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """兼容方法，等价于 load_for_runner"""
        return cls.load_for_runner(config_path)