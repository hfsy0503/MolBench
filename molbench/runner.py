"""
molbench 主运行脚本
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'  # 禁用 SSL 验证（临时）
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from core.utils import cached_transform
from core import (
    load_file, load_data, select_task_columns, select_task_type,
    standardization, split_data, get_featurizer,
    UnifiedModelSelector, optimization, show_results,
    select_extra_metrics, visualizer, BenchGNN
)

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)


# 主函数
def main():
    # 加载数据
    df, file_base = load_file()
    smiles_col, feature_cols, target_cols= select_task_columns(df)

    # 选择任务类型
    task_type= select_task_type()

    # 模型选择器
    if task_type == 'regression':
        model_dir = r'molbench\core\hyper_parameters\test'
    elif task_type == 'binary':
        model_dir = r'molbench\core\hyper_parameters\test'
    # elif task_type == 'multiclass':
    #     model_dir = 'hyper_parameters/multiple'
    else:
        raise ValueError(f'未知任务类型{task_type}')
    
    TEXT_MODELS = ['HFTextModel', 'DeepChemTextCNN']
    GRAPH_MODELS = ['BenchGNN', 'GCN', 'GAT', 'GIN','TransformerConv','GraphSAGE']
    has_text_model = False
    has_graph_model = False

    model_selector = UnifiedModelSelector(models_dir=model_dir)
    json_configs = model_selector.load_models(selection='interactive')

    # 分类模型: 对不同类别的模型区别处理
    text_models = {}
    graph_models = {}
    sklearn_models = {}
    for name, cfg in json_configs.items():
        model_type = cfg.get('model', '')
        if any(t in str(model_type) for t in TEXT_MODELS):
            has_text_model = True
            text_models[name] = cfg
        elif any(t in model_type for t in GRAPH_MODELS):
            has_graph_model = True
            graph_models[name] = cfg
        else:
            sklearn_models[name] = cfg
    print(f"检测到模型: GNN Models={list(graph_models.keys())}, "
          f"Sklearn Models={list(sklearn_models.keys())}, "
          f"Text Models={list(text_models.keys())}")
    data_map = {}
    
    # 选择额外评价指标
    extra_metrics = select_extra_metrics(task_type)

    for target_col in target_cols:
        task_id = f"{file_base}_{target_col}"
        if target_col not in df.columns:
            raise ValueError(f"指定的列 '{target_col}' 不存在于数据集中。")
        print(f"\n处理目标列: {target_col}")
        
        # 获取当前列的序号
        col_num = df.columns.get_loc(target_col)

        # 加载&清洗数据集
        X_raw, y_raw = load_data(df, task_type, smiles_col, target_col)

        # 为每一列数据采取适合的划分方式，仅当前任务列用于分层
        train_idx, val_idx, test_idx = split_data(
            df.loc[X_raw.index], task_type=task_type, smiles_col=smiles_col, stratify_col=target_col)
        y_train, y_val, y_test = y_raw.loc[train_idx], y_raw.loc[val_idx], y_raw.loc[test_idx]

        # 准备两种数据
        # 1. SMILES 数据（GNN + Text）
        X_train_smiles = X_raw.loc[train_idx].tolist()
        X_val_smiles = X_raw.loc[val_idx].tolist()
        X_test_smiles = X_raw.loc[test_idx].tolist()

        # 2. 特征化数据（sklearn模型）
        if sklearn_models: 
            featurizer = get_featurizer()
            X_feat = cached_transform(
                X_raw,
                task_id=task_id,
                transform_fn=lambda: featurizer.transform(X_raw).astype(float),
                name=type(featurizer).__name__.lower().replace('featurizer', ''),
                **{k: getattr(featurizer, k, None) 
                   for k in ['radius', 'n_bits', 'max_atoms', 'use_chirality'] 
                   if hasattr(featurizer, k)}
            )
            X_feat = pd.DataFrame(X_feat, index=X_raw.index)

            # 其他特征（若有）拼到 SMILES 后面
            other_cols = [c for c in feature_cols if c != smiles_col]
            if other_cols:
                X_other = df.loc[X_raw.index, other_cols].astype(float)
                X_full = pd.concat([X_feat, X_other], axis=1)
            else:
                X_full = X_feat

            # 划分+标准化
            X_train, X_val, X_test = X_full.loc[train_idx], X_full.loc[val_idx], X_full.loc[test_idx]
            X_train_scaled, X_val_scaled, X_test_scaled = standardization(X_train, X_val, X_test)
            
        all_best_models = {}
        all_results ={}
        rank_metric = 'val_r2' if task_type == 'regression' else 'val_roc_auc'

        # 分类训练模型(贝叶斯优化+验证集预测)
        if graph_models:
            print(f"训练 GNN 模型: {list(graph_models.keys())}")
        
            best_models_gnn, results_gnn, metric_gnn, selector_gnn, data_cache= optimization(
                X_train_smiles, X_val_smiles, y_train, y_val, task_type,
                json_configs=graph_models,
                X_test=X_test_smiles,
                y_test=y_test,
                extra_metrics=extra_metrics,
                cache_task_id=task_id
                )
            all_best_models.update(best_models_gnn)
            all_results.update(results_gnn)
            rank_metric = metric_gnn

            for name in graph_models.keys():
                if name in data_cache.get('graphs', {}):
                    graph_data = data_cache['graphs'][name]
                    data_map[name] = {
                        'X': graph_data['X_val'],      # 图对象列表
                        'y_val': graph_data['y_val'],
                        'X_test': graph_data.get('X_test'),  # 测试集图对象
                        'y_test': graph_data.get('y_test'),
                        'selector': None,
                        'is_graph': True
                    }
                else:
                    # 回退
                    data_map[name] = {
                        'X': X_val_smiles,
                        'y_val': y_val,
                        'X_test': X_test_smiles,
                        'y_test': y_test,
                        'selector': None,
                        'is_graph': False,
                        'model_type': 'gnn'
                    }

        if sklearn_models:
            print(f"训练 Sklearn 模型: {list(sklearn_models.keys())}")

            best_models_sklearn, results_sklearn, metric_sklearn, selector_sklearn, data_cache_sklearn= optimization(
                X_train_scaled, X_val_scaled, y_train, y_val, task_type,
                json_configs=sklearn_models,
                X_test=X_test_scaled,
                y_test=y_test,
                extra_metrics=extra_metrics)
            all_best_models.update(best_models_sklearn)
            all_results.update(results_sklearn)
            rank_metric = metric_sklearn

            if selector_sklearn is not None:
                print(f"  应用特征选择: 1024 → {selector_sklearn.k} 维")
                X_val_selected = selector_sklearn.transform(X_val_scaled)
                X_test_selected = selector_sklearn.transform(X_test_scaled)
            else:
                X_val_selected = X_val_scaled
                X_test_selected = X_test_scaled

            for name in sklearn_models.keys():
                data_map[name] = {
                    'X': X_val_selected,
                    'y_val': y_val,
                    'X_test': X_test_selected,
                    'y_test': y_test,
                    'selector': selector_sklearn,
                    'is_graph': False,
                    'model_type': 'sklearn'
                }
        
        if not all_best_models:
            raise ValueError("没有模型成功训练")

        # 合并结果
        model_rank = show_results(all_results, rank_metric)
        print("模型运行结果比较: \n", model_rank)
        
        # 打印出结果后，使用最好的模型进行最终预测
        best_model_name = str(model_rank.index[0])
        best_model = all_best_models[best_model_name] # 获取最好的模型
    
        if best_model_name not in data_map:
            raise ValueError(f"{best_model_name} 不在 data_map 中，无法获取预处理数据")
        model_data = data_map[best_model_name]

        if 'y_val' in model_data and model_data['y_val'] is not None:
            y_val_input = model_data['y_val']
        else:
            y_val_input = y_val
        if 'y_test' in model_data and model_data['y_test'] is not None:
            y_test_input = model_data['y_test']
        else:
            y_test_input = y_test

        X_test_input = model_data['X_test']
        X_val_input = model_data['X']
        selector_for_vs = model_data.get('selector')

        if len(X_test_input) != len(y_test_input):
            print(f"⚠️ 长度不匹配: X_test={len(X_test_input)}, y_test={len(y_test_input)}")
            if 'test_indices' in model_data:
                y_test_input = y_test[model_data['test_indices']]
                print(f"  用 test_indices 对齐: y_test 现在 {len(y_test_input)} 个")
            else:
                raise ValueError(f"无法对齐: 缺少 test_indices")
            
        if len(X_val_input) != len(y_val_input):
            print(f"  ⚠️ 验证集长度不匹配: X_val={len(X_val_input)}, y_val={len(y_val_input)}")
            if 'val_idx' in model_data:
                y_val_input = y_val[model_data['val_idx']]
                print(f"  用 val_idx 对齐: y_val 现在 {len(y_val_input)} 个")
        try:
            # 预测
            y_test_pred = best_model.predict(X_test_input)
            print(f"  ✅ {best_model_name} 预测成功: {len(y_test_pred)} 个样本")
        except Exception as e:
            print(f"  ⚠️ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        out_name = f'{file_base}_{col_num}_test_results.csv'
        model_rank.to_csv(out_name, index_label='model')

        # 可视化结果
        vs = visualizer(task_type, selector=selector_for_vs, data_map=data_map)
        if task_type == 'regression':
            vs.plot('auto', y_test_input, y_test_pred, file_name=file_base, col_idx=col_num)
        elif task_type == 'binary':
            # 1. 验证集画图
            best_name, _ = vs.plot('val', all_best_models, None, y_val_input, 
                                    file_name= file_base, col_idx=col_num)
            # 2. 测试集画图
            vs.plot('test', best_model, X_test_input, y_test_input, best_model_name,
                        file_name=file_base, col_idx=col_num)

if __name__ == "__main__":
    main()
    from molbench.core.utils import cache_stats
    cache_stats()