#!/usr/bin/env python
"""
molbench/core/runner.py - 交互模式入口
调用 runner_engine 确保与配置模式结果一致
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

from molbench.core.runner_engine import run_benchmark, TEXT_MODELS, GRAPH_MODELS
from molbench.core.data import load_file, select_task_columns
from molbench.core.utils import select_task_type, UnifiedModelSelector

def main():
    """交互式模式"""
    
    print("="*50)
    print("MolBench 交互式模式")
    print("="*50)
    
    # 1. 交互式输入
    df, file_base = load_file()
    smiles_col, feature_cols, target_cols = select_task_columns(df)
    task_type = select_task_type()
    
    # 2. 模型选择
    if task_type == 'regression':
        model_dir = r'molbench\core\hyper_parameters\test'
    elif task_type == 'binary':
        model_dir = r'molbench\core\hyper_parameters\test'
    else:
        raise ValueError(f'未知任务类型: {task_type}')
    
    model_selector = UnifiedModelSelector(models_dir=model_dir)
    json_configs = model_selector.load_models(selection='interactive')
    
    # 3. 分类模型
    text_models, graph_models, sklearn_models = {}, {}, {}
    for name, cfg in json_configs.items():
        model_type = cfg.get('model', '')
        if any(t in str(model_type) for t in TEXT_MODELS):
            text_models[name] = cfg
        elif any(t in model_type for t in GRAPH_MODELS):
            graph_models[name] = cfg
        else:
            sklearn_models[name] = cfg
    
    print(f"\n检测到模型: GNN={list(graph_models)}, Sklearn={list(sklearn_models)}, Text={list(text_models)}")
    
    # 4. 额外参数
    from molbench.core.evaluation import select_extra_metrics
    extra_metrics = select_extra_metrics(task_type)
    
    featurizer_name = None
    featurizer_params = {}
    # 5. 特征化器
    if sklearn_models:
        from molbench.core.featurizers import get_featurizer
        featurizer = get_featurizer()
        featurizer_name = type(featurizer).__name__.lower().replace('featurizer', '')
        featurizer_params = {k: getattr(featurizer, k, None) 
                            for k in ['radius', 'n_bits', 'max_atoms', 'use_chirality'] 
                            if hasattr(featurizer, k)}
    
    # 6. 调用核心引擎
    results = run_benchmark(
        df=df,
        file_base=file_base,
        smiles_col=smiles_col,
        feature_cols=feature_cols,
        target_cols=target_cols,
        task_type=task_type,
        
        graph_models=graph_models,
        sklearn_models=sklearn_models,
        text_models=text_models,
        
        featurizer_name=featurizer_name,
        featurizer_params=featurizer_params,
        
        extra_metrics=extra_metrics,
        split_method='random',  # 交互模式默认
        split_seed=42,
        
        cache_enabled=True,
        verbose=True,
        interactive=True
    )
    
    return results


if __name__ == "__main__":
    main()