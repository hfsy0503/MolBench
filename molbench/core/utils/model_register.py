import json
import os
import inspect
import importlib
from skopt.space import Real, Integer, Categorical

def register_model(model_cls, task_type: str, save_dir: str, 
                   search_space: dict = None, fixed_params: dict = None,
                   protocol: str = 'bench', **defaults):
    """
    把用户模型登记到框架扫描目录
    model_cls : 模型类（如BenchGNN， BenchModel子类）
    task_type : 'regression' | 'binary' | 'multiclass'
    save_dir  : 超参 JSON 存放目录，如 hyper_parameters/test
    search_space : 可选，贝叶斯优化的搜索空间定义，格式如：
        {
            "hidden": {"type": "integer", "bounds": [32, 256]},
            "lr": {"type": "real", "bounds": [1e-4, 1e-2], 'prior': 'log-uniform'},
        }
    fixed_params : 固定不变的超参，如 model_name
    protocol  : 'bench' | 'custom'，决定存放子目录
    **defaults  : 其他默认超参
    """
    os.makedirs(save_dir, exist_ok=True)

    # 默认超参 = 类签名默认值
    sig = inspect.signature(model_cls.__init__)
    sig_defaults = {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in sig.parameters.items()
        if k not in ('self', 'args', 'kwargs')}
    
    # 合并超参优先级：用户传入 > 固定值 > 默认值
    final_fixed = {**sig_defaults, **defaults, **(fixed_params or {})}  
    model_name = final_fixed.get('model_name', model_cls.__name__) # 模型的真实名称
    class_name = model_cls.__name__  # 注册用的类名（默认为模型类名）

    # 模块路径处理
    module = model_cls.__module__
    if module == '__main__':
        file_path = inspect.getfile(model_cls)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # core/utils/../.. -> core/
        rel_path = os.path.relpath(file_path, root)
        module = rel_path.replace(os.sep, '.').replace('.py', '')

    cfg = {
        "model": class_name,
        "module": module,
        "task_type": task_type,
        'protocol': protocol,
        "search_space": search_space or {},
        "fixed_params": final_fixed  # 贝叶斯优化以此为初始空间
    }
    
    out_file = os.path.join(save_dir, f"{model_name}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
    print(f"[register] 已生成 {out_file}")
    print(f"  - 搜索空间: {list(search_space.keys()) if search_space else 'None'}")
    print(f"  - 固定参数: {list(final_fixed.keys())}")

def load_bench_model(module_path: str, class_name: str):
    """
    根据字符串模块名和类名动态导入并返回 BenchModel 子类
    例：load_bench_model('user_models.my_xgb', 'MyXGB') -> <class 'MyXGB'>
    """
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 {module_path}.{class_name} : {e}") from e

    # 简单校验（必须是 BenchModel 的子类）
    from adapters.base import BenchModel
    if not inspect.isclass(cls) or not issubclass(cls, BenchModel):
        raise TypeError(f"{class_name} 必须是 BenchModel 的子类")
    return cls