"""
简化版贝叶斯优化
- 固定 n_iter=30（或配置化），所有模型相同
- 无早停，确保公平对比
- 随机种子固定，可复现

如需更高质量调参，建议：
1. 增加 n_iter（如 20-50）
2. 或改用两阶段：贝叶斯找参 + 交叉验证评估
"""
from molbench.core.evaluation import evaluation
from molbench.core.utils import UnifiedModelSelector
from molbench.core.adapters import BenchModel
import pandas as pd
import numpy as np
import warnings
import time
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

def set_seed(seed = 42):
    import random, numpy as np, torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

def optimization(X_train, X_val, y_train, y_val, task_type, 
                 model_selection=None, 
                 json_configs=None,
                 calibrate_probability=True,
                 X_test=None,
                 y_test=None,
                 extra_metrics=False,
                 cache_task_id=None,
                 n_iter=30):
    """
    统一的优化函数
    
    参数:
    - X_train, y_train: 训练数据
    - X_val, y_val: 验证数据（用于模型选择）
    - X_test, y_test: 测试数据（可选，用于评估泛化能力）
    - model_selection: 
        None/"all": 使用所有模型
        list: 使用指定模型, 如 ['RandomForest', 'SVM']
        "interactive": 交互式选择
    """
    data_cache = {
        'X_train': X_train,  # 原始输入（SMILES或特征）
        'X_val': X_val,
        'X_test': X_test,
        'graphs': {}  # 存储GNN转换后的图对象
    }
    n_iter = 1
    try:
        print(f"\n🔍 数据类型检查:")
        print(f"  X_train 类型: {type(X_train)}, 形状: {getattr(X_train, 'shape', 'N/A')}")
        print(f"  X_val 类型: {type(X_val)}, 形状: {getattr(X_val, 'shape', 'N/A')}")
        assert len(X_train) == len(y_train), f"X_train({len(X_train)}) != y_train({len(y_train)})"
        assert len(X_val) == len(y_val), f"X_val({len(X_val)}) != y_val({len(y_val)})"
    
        # 如果是文本列表（SMILES），跳过所有特征选择
        is_text_data = (
            isinstance(X_train, list) and len(X_train) > 0 and isinstance(X_train[0], str)
        ) or (
            hasattr(X_train, 'dtype') and X_train.dtype.kind in ['U', 'S', 'O']  # 字符串类型
        )
        
        if is_text_data:
            print("  检测到文本数据（SMILES），跳过特征选择")
            X_train_selected = X_train
            X_val_selected = X_val
            selector = None
        else:
            # 📊 特征选择：处理高维数据的 n<p 问题
            print("\n🔧 应用特征选择 (SelectKBest k=200)...")
            if (hasattr(X_train, 'ndim') and X_train.ndim == 2 and
                hasattr(X_train, 'shape') and X_train.shape[1] > 200):
                score_func = f_regression if task_type == 'regression' else f_classif
                selector = SelectKBest(score_func=score_func, k=200)
                selector.fit(X_train, y_train)          # 仅 fit 一次
                print(f"  ✓ 特征维度: {X_train.shape[1]} → {selector.k}")
            else:
                selector = None
                dim_info = getattr(X_train, 'shape', f'list(len={len(X_train)})')
                print(f"  ✓ 跳过特征选择 (输入： {dim_info})")
            
            if selector is not None:
                X_train_selected = selector.transform(X_train)
                X_val_selected = selector.transform(X_val)  # 使用同一个 selector！
                X_test_selected = selector.transform(X_test) if X_test is not None else None
                print(f"  训练集形状: {X_train_selected.shape}")
                print(f"  验证集形状: {X_val_selected.shape}")
            else:
                X_train_selected = X_train
                X_val_selected = X_val
                X_test_selected = X_test
                selector = None

        if not is_text_data and hasattr(X_val_selected, 'shape'):
            if X_train_selected.shape[1] != X_val_selected.shape[1]:
                raise ValueError(
                    f"维度不匹配！训练集: {X_train_selected.shape}, "
                    f"验证集: {X_val_selected.shape}"
                )

        # 加载模型配置
        if json_configs is None:
            model_selector = UnifiedModelSelector('model_configs')
            json_configs = model_selector.load_models(model_selection)
        else:
            print("使用传入的json_configs...")
        
        if not json_configs:
            raise ValueError("没有成功加载任何模型配置")
        
        # 转换为skopt格式
        from molbench.core.hyper_parameters.configs_manager import HyperparamConfigManager
        config_manager = HyperparamConfigManager()
        models_config = config_manager.json_to_skopt(json_configs)
        print(f"加载的模型配置:")
        for name, config in json_configs.items():
            print(f"  {name}: model={config.get('model')}, protocol={config.get('protocol')}")

        scoring_map ={
            'regression': 'r2',
            'binary': 'roc_auc',
            'multiclass': 'roc_auc_ovr'
        }
        scoring_metric = scoring_map.get(task_type, 'accuracy')

        if is_text_data:
            # 文本数据：列表拼接
            X_train_selected = X_train
            X_val_selected = X_val
            X_test_selected = X_test
            X_combined = X_train_selected + X_val_selected
            print(f"  文本数据合并: {len(X_train_selected)} + {len(X_val_selected)} = {len(X_combined)} 个样本")
        else:
            # 数值数据：np.vstack
            X_combined = np.vstack([X_train_selected, X_val_selected])
            print(f"  数值数据合并: {X_combined.shape}")

        # y 数据合并（始终是数组）
        y_train_arr = np.asarray(y_train).ravel()
        y_val_arr = np.asarray(y_val).ravel()
        y_combined = np.concatenate([y_train_arr, y_val_arr])
        print(f"  y 合并后: {y_combined.shape}")

        # 使用 PredefinedSplit 替代 KFold
        from sklearn.model_selection import PredefinedSplit
        # -1 表示训练集，0 表示验证集
        test_fold = np.concatenate([
            np.full(len(X_train_selected), -1),
            np.full(len(X_val_selected), 0)
        ])
        cv_obj = PredefinedSplit(test_fold=test_fold)
    
        # 贝叶斯优化
        best_models={}
        results={}
        first_metric_name = 'val_r2' if task_type == 'regression' else 'val_roc_auc'
        models_need_calibration = [
            'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier', 
            'GaussianProcessClassifier', 'QuadraticDiscriminantAnalysis']
        
        converter_cache = {}
        for name, config in models_config.items():
            print(f"\n 正在训练{name}...") # 打印当前正在训练的模型名称。
            start_time = time.time()  # 记录训练开始时间 
            
            model_n_iter = config.get('n_iter') or n_iter
            try:
                ModelClass = config['model']
                fixed_params = config['fixed_params']
                skopt_space = config['skopt_space']
                protocol = config['protocol']
                is_gnn = (
                        hasattr(ModelClass, 'fit_graphs') or 'GNN' in name 
                        or name in ['GAT', 'GCN', 'GIN', 'GraphConv', 'MPNN', 'TransformerConv'])
                is_text = (
                    'BERT' in name or hasattr(ModelClass, '_tokenize') or 
                    name in ['SciBERT', 'ChemBERTa2-MLM', 'ChemBerTa2-MTR'])
                
                if protocol == 'bench': 
                    # 检查是否实现协议
                    required_methods = ['fit', 'predict', 'save', 'load', 'get_params', 'get_task_type']
                    missing = [m for m in required_methods if not hasattr(ModelClass, m)]
                    if missing:
                        raise ValueError(f"模型 {name} 缺少必要方法：{missing}")
                    
                    if is_gnn and name not in converter_cache:
                        temp_model = ModelClass(**config.get('fixed_params', {}))

                        if cache_task_id:
                            from molbench.core.utils import enable_graph_cache
                            enable_graph_cache(temp_model, cache_task_id)
                            print(f" ✓ GNN磁盘缓存已启用: {cache_task_id}")

                        converter = temp_model.graph_converter

                        X_train_g, train_idx = converter.batch_convert(list(X_train_selected), fit_scaler=True)
                        y_train_f = np.asarray(y_train).ravel()[train_idx]
                        
                        # 转换验证集
                        X_val_g, val_idx = converter.batch_convert(list(X_val_selected), fit_scaler=False)
                        y_val_f = np.asarray(y_val).ravel()[val_idx]
                        
                        converter_cache[name] = {
                            'model': temp_model,  # 保存带缓存的模型实例
                            'X_train': X_train_g,
                            'y_train': y_train_f,
                            'X_val': X_val_g,
                            'y_val': y_val_f,
                            'train_idx': train_idx,
                            'val_idx': val_idx,
                        }
                        data_cache['graphs'][name] = converter_cache[name]
                        print(f"  [data_cache] 已缓存 {name} 的图数据: "
                            f"train={len(X_train_g)}, val={len(X_val_g)}")
                    
                    if not skopt_space:
                        if is_gnn:
                            best_model = converter_cache[name]['model']
                            best_model.fit_graphs(
                                converter_cache[name]['X_train'], 
                                converter_cache[name]['y_train'])
                        else:
                            best_model = ModelClass(**fixed_params)
                            best_model.fit(X_train_selected, y_train)
                    else:
                        from skopt import Optimizer
                        if isinstance(skopt_space, dict):
                            dimensions = list(skopt_space.values())  
                            param_names = list(skopt_space.keys())    
                        else:
                            dimensions = skopt_space  # 已经是列表
                            param_names = [d.name for d in dimensions]

                        optimizer = Optimizer(dimensions, random_state=42)
                        
                        best_score = -np.inf
                        best_model = None
                        best_params = None

                        if is_gnn:
                            cached = converter_cache[name]
                            if len(X_val_g) == 0:
                                raise ValueError("验证集无有效图")

                        for i in range(model_n_iter): 
                            suggested = optimizer.ask()
                            params_dict = dict(zip(param_names, suggested))
                            
                            print(f"\n  Iteration {i+1}/{model_n_iter}: {params_dict}")
                            # 创建模型并训练（只在训练集上）
                            model = ModelClass(**fixed_params)
                            model.set_params(**params_dict)

                            if is_gnn:
                                model.graph_converter = cached['model'].graph_converter 
                            
                            try:
                                if is_gnn:
                                    model.fit_graphs(cached['X_train'], cached['y_train'])
                                    y_pred = model.predict_graphs(cached['X_val'])
                                    score_y_true = cached['y_val']
                                else:
                                    model.fit(X_train_selected, y_train)
                                    y_pred = model.predict(X_val_selected)
                                    score_y_true = y_val

                                if task_type == 'regression':
                                    from sklearn.metrics import r2_score
                                    score = r2_score(score_y_true, y_pred)
                                else:
                                    from sklearn.metrics import roc_auc_score
                                    score = roc_auc_score(score_y_true, y_pred)
                                print(f"  Validation score: {score:.4f}")
                                
                                optimizer.tell(suggested, -score)  # 最小化问题
                                
                                if score > best_score:
                                    best_score = score
                                    best_model = model
                                    best_params = params_dict
                                    print(f"  *** New best model! ***")
                                    
                            except Exception as e:
                                print(f"训练失败: {e}")
                                optimizer.tell(suggested, 0.0)  # 告诉优化器这次很差
                        
                        print(f"\nBest validation score: {best_score:.4f}")
                        print(f"Best params: {best_params}")
                        final_model = best_model

                else: # sklearn 模型
                    ModelClass = config['model']
                    fixed_params = config.get('fixed_params', {}) or {}
                    skopt_space = config.get('skopt_space', {})

                    if isinstance(ModelClass, str):
                        raise ValueError(f"sklearn 模型 {name} 的 model 是字符串 '{ModelClass}'应该是类。")
                    
                    from sklearn.model_selection import PredefinedSplit
                    if not skopt_space:
                        best_model = ModelClass(**fixed_params)
                        best_model.fit(X_train_selected, y_train)
                        best_score = None
                    else:
                        from skopt import Optimizer  # 改用与 GNN 相同的 Optimizer
                        
                        if isinstance(skopt_space, dict):
                            dimensions = list(skopt_space.values())
                            param_names = list(skopt_space.keys())
                        else:
                            dimensions = skopt_space
                            param_names = [d.name for d in dimensions]

                        optimizer = Optimizer(dimensions, random_state=42)
                        
                        best_score = -np.inf
                        best_model = None
                        best_params = None

                        for i in range(model_n_iter):
                            suggested = optimizer.ask()
                            params_dict = dict(zip(param_names, suggested))
                            
                            print(f"\n  Iteration {i+1}/{model_n_iter}: {params_dict}")
                            
                            model = ModelClass(**fixed_params)
                            model.set_params(**params_dict)
                            
                            try:
                                model.fit(X_train_selected, y_train)
                                
                                # 关键：在 X_val 上评估（与 GNN 一致！）
                                if hasattr(model, 'predict_proba'):
                                    y_score = model.predict_proba(X_val_selected)
                                    if y_score.ndim > 1 and y_score.shape[1] == 2:
                                        y_score = y_score[:, 1]
                                else:
                                    y_score = model.predict(X_val_selected)
                                
                                if task_type == 'regression':
                                    from sklearn.metrics import r2_score
                                    score = r2_score(y_val, y_score)
                                else:
                                    from sklearn.metrics import roc_auc_score
                                    score = roc_auc_score(y_val, y_score)
                                
                                print(f"  Validation score: {score:.4f}")
                                optimizer.tell(suggested, -score)
                                
                                if score > best_score:
                                    best_score = score
                                    best_model = model
                                    best_params = params_dict
                                    best_models[name] = model
                                    print(f"  *** New best model! ***")
                                    
                            except Exception as e:
                                print(f"  训练失败: {str(e)[:100]}")
                                optimizer.tell(suggested, 0.0)
                        
                        print(f"\nBest validation score: {best_score:.4f}")
                        print(f"Best params: {best_params}")
                        
                        # 恢复固定参数（BayesSearchCV 的 set_params 可能会覆盖）
                        if fixed_params:
                            # 特殊处理：如果有 kernel 字符串，需要转换为对象
                            if 'kernel' in fixed_params and isinstance(fixed_params['kernel'], str):
                                kernel_str = fixed_params['kernel']
                                try:
                                    from sklearn.gaussian_process.kernels import (
                                        ConstantKernel, RBF, Matern, WhiteKernel, RationalQuadratic, ExpSineSquared
                                    )
                                    kernel_obj = eval(kernel_str, {
                                        'ConstantKernel': ConstantKernel,
                                        'RBF': RBF,
                                        'Matern': Matern,
                                        'WhiteKernel': WhiteKernel,
                                        'RationalQuadratic': RationalQuadratic,
                                        'ExpSineSquared': ExpSineSquared,
                                    })
                                    fixed_params = fixed_params.copy()
                                    fixed_params['kernel'] = kernel_obj
                                except Exception as e:
                                    print(f"    ⚠️  无法转换 kernel: {e}")
                    final_model = best_model

                # 概率校准
                final_model = best_model
                if (task_type != 'regression' and calibrate_probability and 
                    not isinstance(final_model, BenchModel) and
                    any(model_name in name for model_name in models_need_calibration)):
                    
                    print(f"对 {name} 进行概率校准...")
                    calibrated_model = calibrate_model(best_model, X_train_selected, y_train, X_val_selected, y_val)
                    final_model = calibrated_model
                else:
                    final_model = best_model
                
                # 计算验证集指标
                ev = evaluation(task_type, extra_metrics=extra_metrics)

                if is_gnn:
                    if name in converter_cache:
                        cached = converter_cache[name]
                        converter = cached['model'].graph_converter
                        X_train_g = cached['X_train']
                        y_train_f = cached['y_train']
                        X_val_g = cached['X_val']
                        y_val_f = cached['y_val']
                        train_idx = cached['train_idx']
                        val_idx = cached['val_idx']
                        # 直接使用缓存的 X_val_g 和 y_val_f
                        val_metrics = ev.plot(final_model, X_val_g, y_val_f, dataset_type='val')
                    else:
                        raise RuntimeError(f"GNN模型 {name} 的 converter_cache 不存在")
                else:
                    # 其它模型：使用原始数据
                    val_metrics = ev.plot(final_model, X_val_selected, y_val, dataset_type='val')

                # 计算测试集指标
                test_metrics = {}
                if X_test is not None and y_test is not None:
                    if is_gnn:
                        converter = converter_cache[name]['model'].graph_converter
                        X_test_graphs, test_indices = converter.batch_convert(list(X_test), fit_scaler=False)
                        y_test_arr = np.asarray(y_test).ravel()
                        
                        data_cache['graphs'][name]['X_test'] = X_test_graphs
                        data_cache['graphs'][name]['test_idx'] = test_indices
                        
                        if len(test_indices) != len(y_test_arr):
                            y_test_aligned = y_test_arr[test_indices]
                        else:
                            y_test_aligned = y_test_arr
                        data_cache['graphs'][name]['y_test'] = y_test_aligned
                            
                        test_metrics = ev.plot(final_model, X_test_graphs, y_test_aligned, dataset_type='test')
                    else:
                        test_metrics = ev.plot(final_model, X_test_selected, y_test, dataset_type='test')

                # 合并指标
                all_metrics = {**val_metrics, **test_metrics}
                
                # 计算泛化间隙（仅适用于回归）
                if task_type == 'regression' and 'test_r2' in all_metrics and 'val_r2' in all_metrics:
                    all_metrics['r2_gap'] = all_metrics['test_r2'] - all_metrics['val_r2']

                # 获取用于排序的指标名称（基于验证集 R²）
                if task_type == 'regression':
                    first_metric_name = 'val_r2'
                else:
                    first_metric_name = 'val_roc_auc'
                    
                end_time = time.time()  # 训练结束时间
                training_time = end_time - start_time  # 计算训练耗时

                results[name] = {**all_metrics, 'training time': training_time}
                best_models[name] = final_model # 记录最佳模型

                print(f" {name} 训练成功！")
                print(f"验证集指标: {val_metrics}")
                if test_metrics:
                    print(f"测试集指标: {test_metrics}")
                    if 'r2_gap' in all_metrics:
                        print(f"泛化间隙 (test_r2 - val_r2): {all_metrics['r2_gap']:.4f}")
                print(f"训练耗时: {training_time:.2f} 秒")
            except NotFittedError as e: 
                print(f"{name} 训练失败: {str(e)}")
                continue  # 跳过当前模型，进入下一个模型的训练
            except Exception as e: 
                print(f" {name} 训练失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue           
        return best_models, results, first_metric_name, selector, data_cache
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return {}, {}, None, None, data_cache

def show_results(results, first_metric_name):
    if not results:                       # 空结果直接返回空表
        print('⚠️  results 为空，无指标可排序')
        return pd.DataFrame()
    results_data = pd.DataFrame(results).T
    results_data = results_data.round(4)
    results_data = results_data.sort_values(first_metric_name, ascending=False)
    
    # 优化列顺序：优先展示验证集/测试集指标和泛化间隙
    if 'val_r2' in results_data.columns:
        # 回归任务：突出验证集、测试集和泛化间隙
        col_order = []
        if 'val_r2' in results_data.columns:
            col_order.append('val_r2')
        if 'test_r2' in results_data.columns:
            col_order.append('test_r2')
        if 'r2_gap' in results_data.columns:
            col_order.append('r2_gap')
        # 添加其他列
        other_cols = [c for c in results_data.columns if c not in col_order and c != 'training time']
        col_order.extend(other_cols)
        if 'training time' in results_data.columns:
            col_order.append('training time')
        results_data = results_data[col_order]
    
    return results_data

def calibrate_model(base_model, X_train, y_train, X_val, y_val):
    # 模型概率校准
    from sklearn.calibration import CalibratedClassifierCV
    try:
        # 使用预拟合方法
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method='isotonic',
            cv='prefit'  # 使用已经训练好的模型
        )
        calibrated_model.fit(X_val, y_val)  # 在验证集上校准
        return calibrated_model
        
    except ValueError:
        # 使用交叉验证方法（如果预拟合失败）
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model.__class__(**base_model.get_params()),
            method='isotonic',
            cv=3
        )
        calibrated_model.fit(X_train, y_train)
        return calibrated_model