import sys, os
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..', '..', '..'))
molbench_path = os.path.join(project_root, 'molbench')
if molbench_path not in sys.path:
    sys.path.insert(0, molbench_path)

from text_adapter import HFTextModel, DeepChemTextCNN
from adapters.gnn_adapter import BenchGNN

# 文字模型注册表
TEXT_MODEL_REGISTRY = {
    # HuggingFace 模型
    'MoLFormer': "ibm-research/MoLFormer-XL-both-10pct",
    'ChemBERTa-10M': {
        'class': HFTextModel,
        'params': {
            'model_path': 'DeepChem/ChemBERTa-10M-MTR',
            'lr': 2e-5,
            'max_len': 128,
            'epochs': 3,
        },
        'search_space': {
            'lr': {"type": "real", "bounds": [1e-5, 5e-5], "prior": "log-uniform"},
            'epochs': {"type": "integer", "bounds": [3, 20]},
            'max_len': {"type": "integer", "bounds": [64, 256]},
            'batch_size': {"type": "integer", "bounds": [16, 32]},
            'weight_decay': {"type": "real", "bounds": [0.0, 0.1]},
        }
    },
    'ChemGPT-1.2B': {
        'class': HFTextModel,
        'params': {
            'model_path': 'ncfrey/ChemGPT-1.2B',
            'lr': 1e-5,  # 大模型小学习率
            'max_len': 256,
            'epochs': 3,
            'trust_remote_code': True,  # 需要
        },
        'search_space': {
            'lr': {"type": "real", "bounds": [1e-6, 1e-5], "prior": "log-uniform"},
            'epochs': {"type": "integer", "bounds": [1, 5]},
            'max_len': {"type": "integer", "bounds": [128, 512]},
            'batch_size': {"type": "integer", "bounds": [4, 8]},
            'warmup_ratio': {"type": "real", "bounds": [0.1, 0.2]},
        }
    },
    'BioGPT':{
        'class': HFTextModel,
        'params': {
            'model_path': 'microsoft/biogpt',
            'lr': 2e-5,
            'max_len': 512,
            'epochs': 3,
            'trust_remote_code': False
        },
        'requirements': ['sacremoses'],
        'search_space': {
            'lr': {"type": "real", "bounds": [5e-6, 2e-5], "prior": "log-uniform"},
            'epochs': {"type": "integer", "bounds": [2, 10]},
            'max_len': {"type": "integer", "bounds": [256, 512]},
            'batch_size': {"type": "integer", "bounds": [8, 16]},
            'warmup_ratio': {"type": "real", "bounds": [0.05, 0.15]},
        }
    },
    'SciBERT':{
        'class': HFTextModel,
        'params': {
            'model_path': 'allenai/scibert_scivocab_uncased',
            'lr': 2e-5,
            'max_len': 256,
            'epochs': 3,
            'trust_remote_code': False
        },
        'search_space': {
            'lr': {"type": "real", "bounds": [1e-5, 5e-5], "prior": "log-uniform"},
            'epochs': {"type": "integer", "bounds": [3, 15]},
            'max_len': {"type": "integer", "bounds": [128, 256]},
            'batch_size': {"type": "integer", "bounds": [16, 32]},
            'weight_decay': {"type": "real", "bounds": [0.0, 0.1]},
        }
    },
    'DeepChem-TextCNN': {
        'class': DeepChemTextCNN,
        'params': {
            'model_path': 'deepchem-textcnn',  # 标识用
            'n_embedding': 128,
            'kernel_sizes': [3, 4, 5],
            'filter_sizes': [100, 200, 100],
            'dropout': 0.5,
            'lr': 1e-3,
            'epochs': 100,
        },
        'search_space': {
            'lr': {"type": "real", "bounds": [5e-4, 2e-3], "prior": "log-uniform"},
            'epochs': {"type": "integer", "bounds": [50, 300]},
            'n_embedding': {"type": "integer", "bounds": [64, 256]},
            'dropout': {"type": "real", "bounds": [0.2, 0.5]},
        }
    },
}

GNN_MODEL_REGISTRY = {
    'GraphSAGE': {
        'class': BenchGNN,
        'params': {
            'out_dim': 1,
            'batch_size': 32,
            'aggr': 'mean',
            'num_layers': 3,
            'random_state': 42,
            'use_edge_features': False,
        },
        'search_space': {
            'lr': {"type": "real", "bounds": [1e-4, 1e-2], "prior": "log-uniform"},
            'epochs': {"type": "integer", "bounds": [10, 100]},
            'hidden_dim': {"type": "integer", "bounds": [32, 128]},
            'dropout': {"type": "real", "bounds": [0.0, 0.5]},
        }
    }
    
}    
if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
    from core.utils.model_register import register_model
    
    for name, config in GNN_MODEL_REGISTRY.items():
        search_space = config['search_space']
        register_model(config['class'], 
                    task_type='binary',
                    save_dir=r'molbench\core\hyper_parameters\test',
                    fixed_params={'model_name': name, 
                                  **{k: v for k, v in config['params'].items()}},
                    search_space=search_space,
                    protocol='bench',
                )
