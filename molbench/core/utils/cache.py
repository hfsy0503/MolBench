"""
磁盘缓存
适配 GraphConverter.batch_convert 和 sklearn 特征化
"""

import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

_CACHE_DIR = Path("./.molbench_cache")
_CACHE_DIR.mkdir(exist_ok=True)

def _get_key(data_id: str, op: str, **params) -> str:
    """生成缓存键"""
    content = json.dumps({'id': data_id, 'op': op, 'p': params}, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:12]

# sklearn 特征缓存
def cached_transform(X_raw: pd.Series, task_id: str, 
                     transform_fn: Callable, **params) -> np.ndarray:
    """sklearn 特征缓存"""
    cache_key = _get_key(task_id, 'feat', **params)
    cache_file = _CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_file.exists():
        print(f"  ✓ 特征缓存命中 [{cache_key}]")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    result = transform_fn()
    
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"  ✓ 特征已缓存 [{cache_key}] {result.shape}")
    
    return result

# GNN 图缓存
class CachedGraphConverter:
    """
    带磁盘缓存的 GraphConverter 包装器
    关键修复：保持原始SMILES顺序，正确返回 (graphs, valid_indices)
    """
    
    def __init__(self, base_converter, task_id: str, cache_enabled: bool = True):
        self.base = base_converter
        self.task_id = task_id
        self.cache_enabled = cache_enabled
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __getattr__(self, name):
        """透传所有其他属性"""
        return getattr(self.base, name)
    
    def batch_convert(self, smiles_list: List[str], fit_scaler: bool = True, 
                      **kwargs) -> Tuple[List[Data], np.ndarray]:
        """
        带缓存的批量转换
        1. 使用原始 smiles_list 顺序（不排序），保持 valid_indices 正确
        2. 缓存键基于内容哈希而非排序后的列表
        3. 返回格式与原始 GraphConverter 完全一致: (graphs, valid_indices)
        """
        print(f"DEBUG cache.batch_convert: input type = {type(smiles_list)}")
        print(f"DEBUG cache.batch_convert: input len = {len(smiles_list) if hasattr(smiles_list, '__len__') else 'N/A'}")
        if isinstance(smiles_list, str):
            print(f"ERROR: smiles_list is string: {smiles_list[:100]}")
            raise ValueError("smiles_list should be list, not string")
        if len(smiles_list) > 0:
            print(f"DEBUG cache.batch_convert: first item = {smiles_list[0] if isinstance(smiles_list[0], str) else type(smiles_list[0])}")
        
        if not self.cache_enabled or len(smiles_list) == 0:
            return self.base.batch_convert(smiles_list, fit_scaler, **kwargs)
        
        # 使用原始顺序计算缓存键（基于内容哈希，不排序）
        # 这样相同内容不同顺序会有不同缓存，但保证了 valid_indices 正确
        content_hash = hashlib.md5(
            ''.join(smiles_list).encode()  # 按原始顺序拼接
        ).hexdigest()[:16]
        
        cache_key = _get_key(
            self.task_id, 
            'graph',
            content_hash=content_hash,  # 原始顺序的内容哈希
            model_type=self.base.model_type,
            use_edge_features=self.base.use_edge_features,
            fit_scaler=fit_scaler,
            **{k: v for k, v in kwargs.items() if k in ['min_nodes', 'min_edges']}
        )
        cache_file = _CACHE_DIR / f"{cache_key}.graph.pkl"
        
        # 尝试加载缓存
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                
                # 验证：检查缓存的 smiles 是否与当前请求匹配（顺序无关）
                if (isinstance(cached, dict) and 
                    'graphs' in cached and 
                    'valid_indices' in cached and
                    len(cached['graphs']) > 0):
                    
                    # 额外验证：检查 smiles 集合是否相同（允许不同顺序）
                    cached_smiles_set = set(cached.get('smiles_list', []))
                    current_smiles_set = set(smiles_list)
                    
                    if cached_smiles_set == current_smiles_set:
                        self._cache_hits += 1
                        print(f"  ✓ 图缓存命中 [{cache_key}] {len(cached['graphs'])} graphs "
                              f"(累计命中{self._cache_hits}次)")
                        
                        # 关键：返回缓存的 valid_indices，对应缓存时的顺序
                        # 但我们需要重新对齐到当前 smiles_list 的顺序
                        return self._realign_cached_graphs(
                            cached, smiles_list
                        )
                    
            except Exception as e:
                print(f"  ⚠️ 缓存加载失败: {e}")
        
        # 缓存未命中，执行原始转换
        self._cache_misses += 1
        print(f"  ⚡ 图转换 [{cache_key}] {len(smiles_list)} molecules "
              f"(累计未命中{self._cache_misses}次)")
        
        # 调用原始转换（保持原始顺序）
        graphs, valid_indices = self.base.batch_convert(
            smiles_list, fit_scaler, **kwargs
        )
        
        # 保存到缓存（保存原始 smiles 顺序）
        try:
            cache_data = {
                'graphs': graphs,
                'valid_indices': valid_indices,
                'smiles_list': smiles_list,  # 保存原始顺序，用于验证
                'fit_scaler': fit_scaler,
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            size_mb = cache_file.stat().st_size / 1024**2
            print(f"  ✓ 图已缓存 [{cache_key}] {len(graphs)} graphs ({size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"  ⚠️ 缓存保存失败: {e}")
        
        return graphs, valid_indices
    
    def _realign_cached_graphs(self, cached: dict, current_smiles: List[str]) -> Tuple[List[Data], np.ndarray]:
        """
        将缓存的图按当前 smiles_list 顺序重新对齐
        确保返回的 valid_indices 对应 current_smiles 的顺序
        """
        cached_smiles = cached['smiles_list']
        cached_graphs = cached['graphs']
        cached_indices = cached['valid_indices']
        
        # 构建映射：smiles -> (graph, original_index)
        smile_to_graph = {}
        for idx, (smile, graph) in enumerate(zip(cached_smiles, cached_graphs)):
            smile_to_graph[smile] = (graph, idx)
        
        # 按当前顺序重新排列
        realigned_graphs = []
        realigned_indices = []
        
        for i, smile in enumerate(current_smiles):
            if smile in smile_to_graph:
                graph, _ = smile_to_graph[smile]
                realigned_graphs.append(graph)
                realigned_indices.append(i)  # 当前顺序中的索引
        
        return realigned_graphs, np.array(realigned_indices, dtype=int)

# 便捷函数 
_original_converter = None

def enable_graph_cache(bench_gnn, task_id: str):
    """
    为 BenchGNN 实例启用图缓存
    
    修复：正确保存原始 converter，支持多次启用/禁用
    """
    global _original_converter
    
    if not hasattr(bench_gnn, 'graph_converter'):
        print("⚠️ 模型无 graph_converter，跳过缓存")
        return
    
    # 获取原始 converter（避免重复包装）
    current = bench_gnn.graph_converter
    if isinstance(current, CachedGraphConverter):
        print(f"✓ GNN图缓存已启用: {task_id} (已包装)")
        # 更新 task_id
        current.task_id = task_id
        return
    
    # 保存原始 converter
    _original_converter = current
    
    # 替换为带缓存的版本
    bench_gnn.graph_converter = CachedGraphConverter(
        _original_converter,
        task_id=task_id,
        cache_enabled=True
    )
    
    print(f"✓ GNN图缓存已启用: {task_id}")

def disable_graph_cache(bench_gnn):
    """恢复原始 converter"""
    global _original_converter
    if _original_converter and hasattr(bench_gnn, 'graph_converter'):
        bench_gnn.graph_converter = _original_converter
        print("✓ 图缓存已禁用")

def clear_cache():
    """清空所有缓存"""
    import shutil
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR)
        _CACHE_DIR.mkdir()
    print("✓ 缓存已清空")

def cache_stats():
    """显示缓存统计"""
    files = list(_CACHE_DIR.glob("*.pkl"))
    total_mb = sum(f.stat().st_size for f in files) / 1024**2
    feat_files = len([f for f in files if '.feat.' not in f.name])
    graph_files = len([f for f in files if '.graph.' in f.name])
    print(f"缓存统计: {len(files)} 文件 ({total_mb:.1f} MB)")
    print(f"  - 特征缓存: {feat_files} 个")
    print(f"  - 图缓存: {graph_files} 个")