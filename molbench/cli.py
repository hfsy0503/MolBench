#!/usr/bin/env python
"""
molbench/cli.py - 统一命令行入口
支持: 直接运行、配置文件、生成配置
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from molbench.configs.cfg_parser import ConfigParser
from molbench.configs.cfg_generator import ConfigGenerator

def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog='molbench',
        description='MolBench - 分子机器学习基准测试框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方式:
  1. 使用内置模板（推荐）:
     python -m molbench -t basic
  
  2. 指定配置文件:
     python -m molbench -c configs/basic.yaml
     python -m molbench -c basic.yaml  （自动搜索configs目录）
  
  3. 生成配置模板:
     python -m molbench -g basic
  
  4. 交互式模式:
     python -m molbench
        """
    )
    
    # 模式选择（互斥）
    mode_group = parser.add_mutually_exclusive_group()
    
    # 内置模板模式
    mode_group.add_argument('-t', '--template',
                           choices=['basic', 'multi_model', 'advanced'],
                           help='使用内置配置模板（无需指定路径）')
    
    # 配置文件模式
    mode_group.add_argument('-c', '--config', 
                           help='YAML配置文件路径（支持自动搜索configs目录）')
    
    # 生成配置模式
    mode_group.add_argument('-g', '--generate-config',
                           choices=['basic', 'multi_model', 'advanced'],
                           help='生成配置文件模板')
    
    # 生成配置时的输出路径
    parser.add_argument('-o', '--output', 
                       help='生成配置的输出路径')
    
    # 信息查询
    parser.add_argument('--list-templates', action='store_true',
                       help='列出可用配置模板')
    
    return parser

def resolve_config_path(config_arg):
    """
    解析配置文件路径，支持自动搜索configs目录
    """
    config_path = Path(config_arg)
    
    # 如果直接存在，直接使用
    if config_path.exists():
        return config_path
    
    # 尝试 configs/ 子目录（多个位置）
    search_paths = [
        Path.cwd() / "configs",                         # 当前工作目录/configs
        Path(__file__).parent / "configs",              # molbench包内configs
        Path(__file__).parent / "configs"/"templates",
        Path.cwd(),                                     # 当前目录
    ]
    
    for search_dir in search_paths:
        candidate = search_dir / config_path.name
        if candidate.exists():
            print(f"ℹ️  自动定位配置: {candidate}")
            return candidate
    
    return None

def run_with_config(config_path):
    """
    使用指定配置文件运行评测
    """
    print(f"📄 配置模式: {config_path}")
    
    # 解析配置
    try:
        cfg = ConfigParser.load_for_runner(config_path)
    except Exception as e:
        print(f"❌ 配置解析错误: {e}")
        return 1
    
    # 导入依赖
    try:
        from molbench.core.runner_engine import run_benchmark, TEXT_MODELS, GRAPH_MODELS
        from molbench.core.data import load_dataset
        from molbench.core.utils import UnifiedModelSelector
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return 1
    
    # 加载数据
    print(f"\n📊 加载数据集: {cfg['file_base']}")
    try:
        if cfg.get('data_path'):
            df = load_dataset(cfg['data_path'])
        else:
            df = load_dataset(cfg['file_base'])
    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return 1
    
    # 推断 target_cols
    if cfg['target_cols'] is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cfg['target_cols'] = [c for c in numeric_cols if c not in ['id', 'ID', 'mol_id']]
        print(f"  自动检测目标列: {cfg['target_cols']}")
    
    # 运行评测
    print(f"\n🚀 启动评测...")
    try:
        results = run_benchmark(
            df=df,
            file_base=cfg['file_base'],
            smiles_col=cfg['smiles_col'],
            feature_cols=cfg['feature_cols'],
            target_cols=cfg['target_cols'],
            task_type=cfg['task_type'],
            graph_models=cfg['graph_models'],
            sklearn_models=cfg['sklearn_models'],
            text_models=cfg['text_models'],
            featurizer_name=cfg['featurizer_name'],
            featurizer_params=cfg['featurizer_params'],
            extra_metrics=cfg['extra_metrics'],
            split_method=cfg['split_method'],
            split_seed=cfg['split_seed'],
            cache_enabled=cfg['cache_enabled'],
            verbose=cfg['verbose'],
            n_iter=cfg['n_iter'],
        )
        
        print(f"\n{'='*50}")
        print("✅ 评测完成")
        print(f"{'='*50}")
        return 0
        
    except Exception as e:
        print(f"\n❌ 评测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 列出模板
    if args.list_templates:
        templates = ConfigGenerator.list_templates()
        print("可用配置模板:")
        for t in templates:
            print(f"  - {t}")
        return 0
    
    # 模式1: 生成配置
    if args.generate_config:
        ConfigGenerator.interactive(
            template=args.generate_config,
            output_path=args.output
        )
        return 0
    
    # 模式2: 使用内置模板
    if args.template:
        config_path = Path(__file__).parent / "configs" / f"{args.template}.yaml"
        if not config_path.exists():
            print(f"❌ 内置模板不存在: {config_path}")
            print(f"   请检查 molbench/configs/ 目录")
            return 1
        return run_with_config(config_path)
    
    # 模式3: 配置文件运行（支持自动搜索）
    if args.config:
        config_path = resolve_config_path(args.config)
        if config_path is None:
            print(f"❌ 配置文件不存在: {args.config}")
            print(f"   已尝试以下位置:")
            print(f"     - {Path(args.config).absolute()}")
            print(f"     - {Path.cwd() / 'configs' / args.config}")
            print(f"     - {Path(__file__).parent / 'configs' / args.config}")
            print(f"\n💡 提示:")
            print(f"   1. 使用完整路径: python -m molbench -c configs/basic.yaml")
            print(f"   2. 使用内置模板: python -m molbench -t basic")
            return 1
        
        return run_with_config(config_path)
    
    # 模式4: 交互式模式（默认）
    print("启动交互式模式...")
    try:
        from molbench.core.runner import main as interactive_main
        interactive_main()
        return 0
    except ImportError:
        print("❌ 无法启动交互式模式")
        import traceback
        traceback.print_exc()
        print("💡 提示: 使用 -c 指定配置文件 或 -t 使用内置模板")
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())