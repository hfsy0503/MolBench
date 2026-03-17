"""
core.featurizers - 分子特征化模块
提供多种分子表示方法：指纹、描述符、嵌入等
"""

from .base import BaseFeaturizer
from .ecfp import ECFPFeaturizer
from .rdkit2d import RDKit2DFeaturizer
from .maccs import MACCSFeaturizer
from .pcfp import PCFPFeaturizer
from .coulomb_matrix import CoulombFeaturizer
from .mol2vec import Mol2VecFeaturizer

featurizer_list = {
    'ECFP': ECFPFeaturizer,
    'RDKit 2D': RDKit2DFeaturizer,
    'MACCCS': MACCSFeaturizer,
    'PubChem Fingerprint': PCFPFeaturizer,
    'Coulomb Matrix': CoulombFeaturizer,
    'Mol2Vec': Mol2VecFeaturizer,
}
IDX_TO_NAME ={
    '1':'ECFP',
    '2':'MACCS',
    '3':'RDKit 2D',
    '4':'PubChem Fingerprint(PCFP)',
    '5':'Coulomb Matrix',
    '6':'Mol2Vec',
    '7':'Custom',
    '8':'HELP'
}
def get_featurizer(name: str = 'HELP',**kwargs) -> BaseFeaturizer:
    """
    获取特征化器
    - 如果 name 是有效特征化器名称，直接使用（非交互）
    - 如果 name='HELP' 或无效，进入交互模式
    """
    # 标准化名称
    name_stripped = name.strip()
    
    # 检查是否是有效名称（非 HELP 且在列表中）
    is_valid_name = (
        name_stripped != 'HELP' 
        and name_stripped in featurizer_list
    )
    
    # 尝试不区分大小写匹配
    if not is_valid_name and name_stripped != 'HELP':
        name_lower = name_stripped.lower()
        for key in featurizer_list:
            if key.lower() == name_lower:
                name_stripped = key  # 使用正确的名称
                is_valid_name = True
                break
    
    # 有效参数：使用非交互模式
    if is_valid_name:
        if kwargs.get('verbose', True):
            print(f"📦 使用特征化器: {name_stripped}")
            if kwargs:
                print(f"   参数: {kwargs}")
        return featurizer_list[name_stripped](**{k: v for k, v in kwargs.items() if k != 'verbose'})
    
    # 无效参数：进入交互模式
    print(f"⚠️ 未知特征化器 '{name_stripped}'，进入交互模式选择...")
    return get_featurizer_interactive(**kwargs)

def get_featurizer_interactive(**kwargs) -> BaseFeaturizer:
    """交互式选择: 字符串 -> 描述符实例"""
    _print_descriptor_menu()

    while True:
        raw = input("请输入想选择的描述符名称(按 q 退出): ").strip()
        if raw.lower() =='q':
            raise SystemExit("已退出选择。")
        
        if raw in IDX_TO_NAME: # 用户输入的是序号
            name = IDX_TO_NAME[raw]
        else: # 用户输入的是字符串
            name = raw
            
        # 特殊选择
        if name =='Custom':
            plug_user_descriptor()
            continue
        if name =='HELP':
            _print_descriptor_menu()
            continue
        # 正常选择
        if name not in featurizer_list:
            print(f"暂不支持描述符'{name}'。当前可用：{list(featurizer_list.keys())}")
            continue
        return featurizer_list[name](**kwargs)

def _print_descriptor_menu():
    menu = """
    可用描述符：
    1. ECFP
    2. MACCS
    3. RDKit 2D
    4. PubChem Fingerprint(PCFP)
    5. Colomb Matrix
    6. Mol2Vec
    7. 自定义描述符：选择已写好的 .py 文件
    8. HELP: 重新显示本菜单
    """
    print(menu)

def plug_user_descriptor():
    """
    1. 用户选择自己写好的 .py 文件
    2. 复制到 featurizers/
    3. 动态 import 并挂到 featurizer_list
    4. 即录即用，无需重启解释器
    """
    from pathlib import Path
    import shutil
    import importlib
    import json
    import os

    pkg_path = Path(__file__).parent          # featurizers 目录
    user_file = input(" 请选择你的描述符 .py 文件路径（直接回车取消）：\n").strip('"\'')
    if not user_file:
        return

    src = Path(user_file)
    if not src.is_file() or src.suffix != '.py':
        print("❌ 必须选择 .py 文件！")
        return

    # 简单安全检查：不允许 import os/system 等危险模块
    with src.open(encoding='utf-8') as f:
        code = f.read()
    for bad in ['os.system', 'subprocess', '__import__']:
        if bad in code:
            print("文件包含危险语句，已拒绝加载。")
            return

    # 1. 复制到 featurizers/（不重名则直接拷）
    dst_name = src.name  # 保留原文件名
    dst = pkg_path / dst_name
    if dst.exists():
        print("⚠️ 文件已存在，将覆盖旧版本！")
    shutil.copy(src, dst)

    # 2. 动态 import（无需重启 Python）
    module_name = dst.stem # 去掉 .py
    try:
        spec = importlib.util.spec_from_file_location(module_name, dst)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        return

    # 3. 约定俗成：类名 = 文件名 + "Featurizer"
    class_name = module_name + "Featurizer"
    if not hasattr(mod, class_name):
        print(f"❌ 模块内找不到类 '{class_name}'！")
        return

    cls = getattr(mod, class_name)
    if not issubclass(cls, BaseFeaturizer):
        print("❌ 类必须继承 BaseFeaturizer！")
        return

    # 4. 挂到注册表（lambda 保留默认参数）
    from . import featurizer_list
    featurizer_list[module_name] = lambda **kw: cls(**kw)

    # 5. 持久化：写进 user_custom.json，下次启动自动加载
    custom_json = pkg_path / 'user_custom.json'
    customs = json.loads(custom_json.read_text()) if custom_json.is_file() else []
    customs.append({
        "name": module_name,
        "module": module_name,
        "class": class_name,
        "params": {}          # 用户可在 get_featurizer 里覆盖
    })
    custom_json.write_text(json.dumps(customs, indent=2, ensure_ascii=False))

    print(f"✅ 描述符 '{module_name}' 已挂载，现在可以选择：")
    print(f"get_featurizer('{module_name}')")

__all__ = [
    'BaseFeaturizer',
    'ECFPFeaturizer',
    'CoulombFeaturizer',
    'MACCSFeaturizer',
    'Mol2VecFeaturizer',
    'PCFPFeaturizer',
    'RDKit2DFeaturizer',

    'get_featurizer',
    'get_featurizer_interactive',
    '_print_descriptor_menu',
    'plug_user_descriptor()',

    'featurizer_list',
    'IDX_TO_NAME',
]