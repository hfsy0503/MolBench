import json
import os
from importlib import import_module
from skopt.space import Integer, Real, Categorical

class HyperparamConfigManager:
    def __init__(self, config_dir='hyperparam_configs'):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

    def _convert_search_space(self, space_dict):
        """JSON 搜索空间 -> skopt Dimension 对象"""
        if not space_dict:
            return [],{}
        
        if isinstance(space_dict, list):
            return space_dict, {}
        
        converted = []
        fixed = {}

        for param_name, spec in space_dict.items():
            if not isinstance(spec, dict):
                fixed[param_name] = spec
                continue

            ptype = spec.get('type')
            bounds = spec.get('bounds', [0, 1])
            prior = spec.get('prior', 'uniform')
            
            if ptype == 'integer':
                converted.append(Integer(int(bounds[0]), int(bounds[1]), name=param_name))
            elif ptype == 'real':
                converted.append(Real(float(bounds[0]), float(bounds[1]), 
                                            prior=prior, name=param_name))
            elif ptype == 'categorical':
                converted.append(Categorical(spec['categories'], name=param_name))
            else:
                fixed[param_name] = spec
        
        return converted, fixed
    
    def json_to_skopt(self, json_configs):
        """
        统一转换：支持 BenchModel 和旧协议 sklearn JSON
        输入：list[dict] 或 dict{name: cfg}
        输出：dict{name: {'model': class/instance, 'params': {...}, ...}}
        """
        skopt_config = {}
        
        # 归一化为 list
        if isinstance(json_configs, dict):
            json_configs = [{'name': k, **v} for k, v in json_configs.items()]
        
        for cfg in json_configs:
            name = cfg.get('name', cfg.get('model', 'unknown'))
            try:
                protocol = cfg.get('protocol', 'sklearn')
                # ===== 新协议：BenchModel =====
                if cfg.get('protocol') == 'bench':
                    model_cls = self._load_bench_model(cfg['module'], cfg['model'])
                    fixed_params = cfg.get('fixed_params', {})
                    search_space = cfg.get('search_space', {})
                    
                    skopt_space, extra_fixed = self._convert_search_space(search_space)
                    fixed_params.update(extra_fixed)

                    skopt_config[name] = {
                        'model': model_cls,                           # 类对象
                        'module':cfg.get('module'),
                        'model_name':cfg['model'],
                        'fixed_params': fixed_params,                 # 实例化时的固定参数
                        'search_space': search_space,
                        'skopt_space': skopt_space,                  # 贝叶斯优化的搜索空间
                        'task_type': cfg.get('task_type', 'regression'),
                        'protocol': 'bench'
                    }
                
                # ===== sklearn 字符串模型 =====
                else:
                    model_cls = self._load_sklearn_model(cfg['model'])
                    
                    fixed_params = cfg.get('fixed_params', {}) or {}
                    search_space = cfg.get('search_space') or cfg.get('params', {}) or []
                    if search_space is None:
                        search_space = []
                    try:
                        if search_space:
                            skopt_space, extra_fixed = self._convert_search_space(search_space)
                            fixed_params.update(extra_fixed)
                        else:
                            skopt_space = []
                            extra_fixed = {}
                    except Exception as conv_e:
                        print(f"⚠️ 搜索空间转换失败 {name}: {conv_e}")
                        skopt_space = []
                    
                    skopt_config[name] = {
                        'model': model_cls,                           # 类对象
                        'fixed_params': fixed_params,
                        'search_space': search_space,
                        'skopt_space': skopt_space,
                        'task_type': cfg.get('task_type', 'regression'),
                        'protocol': 'sklearn',
                        'n_iter': cfg.get('n_iter', 10)
                    }
                
                print(f"✓ 转换: {name} ({protocol})")
                
            except Exception as e:
                print(f"✗ 转换失败 {name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return skopt_config
    
    # ---------- 内部工具 ----------
    def _load_bench_model(self, module_path, class_name):
        """动态加载 bench 模型类"""
        module = import_module(module_path)
        cls = getattr(module, class_name)
        
        required_methods = ['fit', 'predict', 'save', 'load', 'get_params', 'get_task_type']
        missing = [m for m in required_methods if not hasattr(cls, m)]
        if missing:
            raise TypeError(f"{class_name} 缺少必要方法: {missing}")
        try:
            import torch.nn as nn
            if issubclass(cls, nn.Module):
                print(f"注意: {class_name} 是 PyTorch 模型")
            else:
                print(f"注意: {class_name} 不是 PyTorch 模型")
        except ImportError:
            pass  # 未安装 PyTorch
        return cls
    
    def _load_sklearn_model(self, class_name):
        """动态加载 sklearn 模型类"""
        # 常见模块尝试顺序
        modules = [
            'sklearn.ensemble', 'sklearn.linear_model', 'sklearn.svm',
            'sklearn.neighbors', 'sklearn.tree', 'sklearn.naive_bayes',
            'sklearn.neural_network', 'sklearn.gaussian_process',
            'sklearn.kernal_ridge', 'sklearn.cross_decomposition',
            'xgboost', 'lightgbm'
        ]
        
        for mod_name in modules:
            try:
                mod = import_module(mod_name)
                return getattr(mod, class_name)
            except (ImportError, AttributeError):
                continue
        raise ImportError(f"找不到模型: {class_name}")
    
    def _convert_param_space(self, params_config):
        """把 JSON 中的参数空间定义转换为 skopt 的搜索空间对象"""
        space = {}
        fixed = {}
        for name, cfg in params_config.items():
            if isinstance(cfg, dict) and 'type' in cfg:
                ptype = cfg.get('type')
                low, high = cfg.get('bounds', [0, 1])[:2]
            
                if ptype == 'integer':
                    space[name] = Integer(int(low), int(high))
                elif ptype == 'real':
                    prior=cfg.get('prior', 'uniform')
                    space[name] = Real(float(low), float(high), prior=prior)
                elif ptype == 'categorical':
                    space[name] = Categorical(cfg['categories'])
            else:
                fixed[name] = cfg  # 固定值
        return space, fixed
    
    def _process_fixed_params(self, fixed):
        """处理固定参数（字符串转 Python 对象）"""
        result = {}
        for k, v in fixed.items():
            if isinstance(v, str):
                v_lower = v.lower()
                if v_lower == 'none':
                    v = None
                elif v_lower == 'true':
                    v = True
                elif v_lower == 'false':
                    v = False
                else:
                    try:
                        v = float(v) if '.' in v else int(v)
                    except ValueError:
                        pass  # 保持字符串
            result[k] = v
        return result
    
    def instantiate_model(self, config, use_search_defaults=False):
        """实例化模型"""
        model_cls = config['model']
        fixed_params = config.get('init_params', {})
        search_space = config.get('search_space', {})
        
        # 从搜索空间提取默认值（下界）
        if use_search_defaults and search_space:
            for param_name, space_def in search_space.items():
                if space_def['type'] == 'integer':
                    fixed_params[param_name] = int(space_def['bounds'][0])
                elif space_def['type'] == 'real':
                    fixed_params[param_name] = float(space_def['bounds'][0])
                elif space_def['type'] == 'categorical':
                    fixed_params[param_name] = space_def['categories'][0]
        
        # 过滤掉搜索空间定义，保留纯粹的固定参数用于实例化
        clean_params = {k: v for k, v in fixed_params.items() if not isinstance(v, dict)}
        print(f"实例化 {model_cls.__name__} with params: {clean_params}")
        return model_cls(**clean_params)