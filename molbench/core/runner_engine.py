"""
molbench/core/runner_engine.py - 评测核心引擎
适配两种模式：命令行交互式 + yaml文件配置式
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# 路径设置
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molbench.core.data import (load_file, load_data, select_task_columns, standardization, 
                                split_data, split_data_non_interactive)
from molbench.core.featurizers import get_featurizer
from molbench.core.utils import (
    cached_transform, enable_graph_cache, cache_stats,
    select_task_type, UnifiedModelSelector, optimization, 
    show_results
)
from molbench.core.evaluation import visualizer, select_extra_metrics

# 定义结果目录
RESULTS_DIR = Path('molbench') / 'results'

# 常量定义（两种模式共用）
TEXT_MODELS = ['HFTextModel', 'DeepChemTextCNN']
GRAPH_MODELS = ['BenchGNN', 'GCN', 'GAT', 'GIN', 'MPNN',
                'GraphConv','TransformerConv', 'GraphSAGE']

def run_benchmark(
    # 数据集参数
    df: pd.DataFrame,
    file_base: str,
    smiles_col: str,
    feature_cols: List[str],
    target_cols: List[str],
    task_type: str,
    
    # 模型参数
    graph_models: Dict[str, Any],
    sklearn_models: Dict[str, Any],
    text_models: Dict[str, Any],
    
    # 特征化参数
    featurizer_name: str,
    featurizer_params: Dict[str, Any],
    
    # 评测参数
    extra_metrics: Any = None,
    split_method: str = 'random',
    split_seed: int = 42,

    # 贝叶斯优化参数
    n_iter: int = 30,
    
    # 系统参数
    cache_enabled: bool = True,
    verbose: bool = True,
    interactive: bool = False,
    
    # 回调函数（用于交互模式更新进度）
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    评测核心引擎：执行完整的模型训练和评估
    
    此函数与模式无关（交互式/配置式都调用它），确保结果完全一致
    
    Parameters
    ----------
    ...（见上文）
    
    Returns
    -------
    dict
        完整结果，包含所有模型、指标、排名
    """
    
    # 初始化缓存
    if cache_enabled:
        print("✓ 磁盘缓存已启用")
    
    all_results_global = {}
    all_best_models_global = {}
    
    # 遍历所有目标列
    for target_col in target_cols:
        task_id = f"{file_base}_{target_col}"
        
        if target_col not in df.columns:
            raise ValueError(f"列 '{target_col}' 不存在")
        
        if verbose:
            print(f"处理目标列: {target_col}")
        
        col_num = df.columns.get_loc(target_col)
        
        # 数据清洗
        X_raw, y_raw = load_data(df, task_type, smiles_col, target_col)
        
        # 数据划分
        if interactive:
            # 交互式：手动选择
            from molbench.core.data import split_data
            train_idx, val_idx, test_idx = split_data(
                df=df.loc[X_raw.index],
                task_type=task_type,
                smiles_col=smiles_col,
                stratify_col=target_cols[0] if target_cols else None,
            )
        else:
            # 非交互式：从配置读取
            from molbench.core.data import split_data_non_interactive
            train_idx, val_idx, test_idx = split_data_non_interactive(
                df=df.loc[X_raw.index],
                task_type=task_type,
                smiles_col=smiles_col,
                stratify_col=target_cols[0] if target_cols else None,
                method=split_method,
                random_state=split_seed
            )
        
        y_train, y_val, y_test = y_raw.loc[train_idx], y_raw.loc[val_idx], y_raw.loc[test_idx]
        
        # SMILES 数据
        X_train_smiles = X_raw.loc[train_idx].tolist()
        X_val_smiles = X_raw.loc[val_idx].tolist()
        X_test_smiles = X_raw.loc[test_idx].tolist()
        
        # 特征化（sklearn）
        X_train_scaled = X_val_scaled = X_test_scaled = None
        selector_sklearn = None
        
        if sklearn_models:
            if verbose:
                print(f"\n--- 准备 Sklearn 特征 ({featurizer_name}) ---")
            
            featurizer = get_featurizer(featurizer_name, **featurizer_params)
            
            X_feat = cached_transform(
                X_raw,
                task_id=task_id,
                transform_fn=lambda: featurizer.transform(X_raw).astype(float),
                name=featurizer_name,
                **featurizer_params
            )
            X_feat = pd.DataFrame(X_feat, index=X_raw.index)
            
            # 合并其他特征
            other_cols = [c for c in feature_cols if c != smiles_col]
            if other_cols:
                X_other = df.loc[X_raw.index, other_cols].astype(float)
                X_full = pd.concat([X_feat, X_other], axis=1)
            else:
                X_full = X_feat
            
            # 划分和标准化
            X_train, X_val, X_test = X_full.loc[train_idx], X_full.loc[val_idx], X_full.loc[test_idx]
            X_train_scaled, X_val_scaled, X_test_scaled = standardization(X_train, X_val, X_test)
            
            if verbose:
                print(f"  特征矩阵: {X_train_scaled.shape}")
        
        # 训练所有模型
        all_best_models = {}
        all_results = {}
        rank_metric = 'val_r2' if task_type == 'regression' else 'val_roc_auc'
        data_map = {}
        
        # GNN 模型
        if graph_models:
            if verbose:
                print(f"\n--- 训练 GNN 模型: {list(graph_models.keys())} ---")
            
            # 启用图缓存
            for name in graph_models.keys():
                # 临时启用，实际在 optimization 内部处理
                pass
            
            best_models_gnn, results_gnn, metric_gnn, selector_gnn, data_cache = optimization(
                X_train_smiles, X_val_smiles, y_train, y_val, task_type,
                json_configs=graph_models,
                X_test=X_test_smiles,
                y_test=y_test,
                extra_metrics=extra_metrics,
                cache_task_id=task_id if cache_enabled else None,
                n_iter= n_iter
            )
            
            all_best_models.update(best_models_gnn)
            all_results.update(results_gnn)
            rank_metric = metric_gnn
            
            # 构建 data_map
            for name in graph_models.keys():
                if name in data_cache.get('graphs', {}):
                    graph_data = data_cache['graphs'][name]
                    data_map[name] = {
                        'X': graph_data['X_val'],
                        'y_val': graph_data['y_val'],
                        'X_test': graph_data.get('X_test'),
                        'y_test': graph_data.get('y_test'),
                        'selector': None,
                        'is_graph': True,
                    }

        # Sklearn 模型
        if sklearn_models:
            if verbose:
                print(f"\n--- 训练 Sklearn 模型: {list(sklearn_models.keys())} ---")
            
            best_models_sklearn, results_sklearn, metric_sklearn, selector_sklearn, _ = optimization(
                X_train_scaled, X_val_scaled, y_train, y_val, task_type,
                json_configs=sklearn_models,
                X_test=X_test_scaled,
                y_test=y_test,
                extra_metrics=extra_metrics,
                n_iter=n_iter,
            )
            
            all_best_models.update(best_models_sklearn)
            all_results.update(results_sklearn)
            rank_metric = metric_sklearn
            
            # 特征选择处理
            if selector_sklearn is not None:
                X_val_selected = selector_sklearn.transform(X_val_scaled)
                X_test_selected = selector_sklearn.transform(X_test_scaled)
            else:
                X_val_selected, X_test_selected = X_val_scaled, X_test_scaled
            
            for name in sklearn_models.keys():
                data_map[name] = {
                    'X': X_val_selected,
                    'y_val': y_val,
                    'X_test': X_test_selected,
                    'y_test': y_test,
                    'selector': selector_sklearn,
                    'is_graph': False,
                }
        
        if text_models:
            if verbose:
                print(f"\n--- 训练 Text 模型: {list(text_models.keys())} ---")

            best_models_text, results_text, metric_text, _, _ = optimization(
                X_train_smiles, X_val_smiles, y_train, y_val, task_type,
                json_configs=text_models,
                X_test=X_test_smiles,
                y_test=y_test,
                extra_metrics=extra_metrics,
                cache_task_id=task_id if cache_enabled else None,
                n_iter=n_iter,
            )

            # ========== 添加详细调试 ==========
            print(f"\n🔥 optimization 返回结果:")
            print(f"  best_models_text keys: {list(best_models_text.keys())}")
            print(f"  results_text keys: {list(results_text.keys())}")
            
            for name, model in best_models_text.items():
                print(f"\n  🔥 模型 {name}:")
                print(f"    类型: {type(model)}")
                if model is not None:
                    print(f"    是否有 predict: {hasattr(model, 'predict')}")
                    print(f"    predict 是否可调用: {callable(getattr(model, 'predict', None))}")
                    print(f"    是否有 forward: {hasattr(model, 'forward')}")
            
            # 过滤掉 None 模型
            best_models_text = {k: v for k, v in best_models_text.items() if v is not None}
            
            # 确保每个模型都有可调用的 predict 方法
            for name in list(best_models_text.keys()):
                model = best_models_text[name]
                if not hasattr(model, 'predict') or not callable(model.predict):
                    print(f"  ⚠️ 模型 {name} 的 predict 方法无效，移除")
                    del best_models_text[name]
                    if name in results_text:
                        del results_text[name]
            
            print(f"🔥 过滤后 best_models_text keys: {list(best_models_text.keys())}")
            # ========== 调试结束 ==========
            
            all_best_models.update(best_models_text)
            all_results.update(results_text)
            rank_metric = metric_text
            
            # Text 模型直接使用 SMILES
            for name in text_models.keys():
                data_map[name] = {
                    'X': X_val_smiles,      # 验证集 SMILES
                    'y_val': y_val,
                    'X_test': X_test_smiles,  # 测试集 SMILES
                    'y_test': y_test,
                    'selector': None,      # 文本模型不需要特征选择
                    'is_graph': False,
                    'is_text': True,       # 标记为文本模型
                }

        # 结果排名
        if not all_best_models:
            raise ValueError("没有模型成功训练")
        
        model_rank = show_results(all_results, rank_metric)
        
        if verbose:
            print(f"\n模型排名:\n{model_rank}")
        
        # 最佳模型最终预测
        best_model_name = str(model_rank.index[0])
        best_model = all_best_models[best_model_name]
        
        if best_model_name not in data_map:
            raise ValueError(f"{best_model_name} 不在 data_map 中")
        
        model_data = data_map[best_model_name]
        
        # 对齐数据
        y_val_input = model_data.get('y_val', y_val)
        y_test_input = model_data.get('y_test', y_test)
        X_test_input = model_data['X_test']
        X_val_input = model_data['X']
        
        # 预测
        y_test_pred = best_model.predict(X_test_input)
        
        if verbose:
            print(f"  ✅ {best_model_name} 预测: {len(y_test_pred)} 样本")
        
        # 保存结果
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_name = RESULTS_DIR / f'{file_base}_{col_num}_test_results.csv'
        model_rank.to_csv(out_name, index_label='model')
        
        # 可视化
        vs = visualizer(task_type=task_type, save_dir=RESULTS_DIR,
                        selector=model_data.get('selector'), data_map=data_map)
        if task_type == 'regression':
            vs.plot('auto', y_test_input, y_test_pred, file_name=file_base, col_idx=col_num)
        elif task_type == 'binary':
            vs.plot('val', all_best_models, None, y_val_input, file_name=file_base, col_idx=col_num)
            vs.plot('test', best_model, X_test_input, y_test_input, best_model_name,
                   file_name=file_base, col_idx=col_num)
        
        # 累计全局结果
        all_results_global[target_col] = all_results
        all_best_models_global[target_col] = all_best_models
    
    # 最终缓存统计
    if cache_enabled:
        cache_stats()
    
    return {
        'results': all_results_global,
        'best_models': all_best_models_global,
        'task_type': task_type,
    }