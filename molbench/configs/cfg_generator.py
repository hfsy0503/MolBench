"""
molbench/configs/cfg_generator.py - 交互式配置生成
"""
import yaml
from pathlib import Path
from typing import Optional
from .cfg_parser import ConfigParser

class ConfigGenerator:
    """交互式配置生成器"""
    
    TEMPLATES = {
        'basic': '单模型快速评测',
        'advanced': '超参数优化 + 交叉验证',
    }
    
    @classmethod
    def interactive(cls, template: Optional[str] = None, 
                    output_path: Optional[str] = None) -> str:
        """
        交互式生成配置文件
        
        Parameters
        ----------
        template : str, optional
            模板名称，None则交互选择
        output_path : str, optional
            输出路径，None则交互输入
            
        Returns
        -------
        str
            生成的配置文件路径
        """
        print("=" * 60)
        print("MolBench 配置生成器")
        print("=" * 60)
        
        # 选择模板
        if template is None:
            print("\n可用模板:")
            for name, desc in cls.TEMPLATES.items():
                print(f"  - {name}: {desc}")
            template = input("\n选择模板 [basic]: ").strip() or "basic"
        
        # 加载模板
        template_path = ConfigParser.DEFAULT_CONFIG_DIR / f"{template}.yaml"
        if not template_path.exists():
            raise FileNotFoundError(f"模板不存在: {template}")
        
        with open(template_path) as f:
            config = yaml.safe_load(f)
        
        # 交互式修改关键参数
        print(f"\n基于模板: {template}")
        print("-" * 60)
        
        config['dataset']['name'] = input(
            f"数据集名称 [{config['dataset']['name']}]: "
        ).strip() or config['dataset']['name']
        
        # 可以添加更多交互...
        
        # 保存
        if output_path is None:
            default_name = f"molbench_{config['dataset']['name']}.yaml"
            output_path = input(f"\n保存路径 [{default_name}]: ").strip() or default_name
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n✓ 配置文件已生成: {output_path.absolute()}")
        print(f"\n运行方式:")
        print(f"  python -m molbench --config {output_path}")
        
        return str(output_path)
    
    @classmethod
    def from_command_line(cls, dataset: str, models: list, **kwargs) -> str:
        """
        从命令行参数快速生成配置
        
        Usage:
            ConfigGenerator.from_command_line(
                dataset='ESOL',
                models=['RandomForest', 'GCN'],
                split='scaffold'
            )
        """
        config = {
            'dataset': {
                'name': dataset,
                'split': {
                    'method': kwargs.get('split', 'random'),
                    'seed': kwargs.get('seed', 42),
                }
            },
            'models': [],
            'featurizer': {
                'name': kwargs.get('featurizer', 'ecfp'),
                'params': {'radius': 2, 'n_bits': 2048}
            },
            'evaluation': {
                'output_dir': kwargs.get('output_dir', './results')
            },
            'system': {
                'cache': True,
                'verbose': True,
            }
        }
        
        # 添加模型
        for model_name in models:
            config['models'].append({
                'name': model_name,
                'type': cls._infer_type(model_name),
                'protocol': 'bench',
                'params': {},
            })
        
        # 保存
        output_path = Path(f"molbench_{dataset}.yaml")
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(output_path)
    
    @staticmethod
    def _infer_type(model_name: str) -> str:
        """从模型名推断类型"""
        gnn_models = ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'TransformerConv', 'MPNN', 'GraphConv']
        text_models = ['HFTextModel', 'DeepChemTextCNN']
        
        if model_name in gnn_models:
            return 'gnn'
        elif model_name in text_models:
            return 'text'
        else:
            return 'sklearn'