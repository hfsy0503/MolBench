from rdkit import Chem
from torch_geometric.data import Data
import numpy as np
import random, os
import torch
try:
    from torch_geometric.utils import from_smiles
    PYG_FROM_SMILES_AVAILABLE = True
except ImportError:
    PYG_FROM_SMILES_AVAILABLE = False

def set_seed(seed=42):
    """固定所有随机性，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有 GPU
    
    # 确保确定性行为（可能牺牲一点速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置 Python hash 种子（影响某些数据结构的顺序）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ 已固定随机种子: {seed}")

set_seed(42)  

class GraphConverter:
    """
    把 SMILES 转成 PyG 的 Data 对象
    支持两种模式：
    1. 快速模式：使用 PyG 内置 from_smiles（无边特征，9维节点特征）
    2. 完整模式：使用 RDKit 自定义（有边特征，133维节点特征）
    """
    BACKEND_MAP = {
    'pyg_fast': 'pyg',
    'pyg_full': 'pyg',
    'graphconv': 'deepchem',
    'mpnn': 'deepchem'
}
    def __init__(self, model_type: str = 'pyg', use_edge_features: bool = False, 
                 handle_errors: bool = True,
                 node_feat_dim: int = 133,
                 normalize: bool = True):
        """use_edge_features: 是否采用完整模式，添加边特征
           handle_errors: 是否捕获无效 SMILES 错误，返回空图
           node_feat_dim: 节点特征维度，快速模式：9（PyG）；完整模式：133（RDKit）
           normalize: 是否对数值特征进行归一化（仅完整模式）"""
        self.model_type = model_type
        self.backend = self.BACKEND_MAP.get(model_type, 'pyg')
        self.handle_errors = handle_errors
        self.node_feat_dim = node_feat_dim
        self.normalize = normalize
        self.scaler = None 

        if self.backend == 'deepchem':
            self._init_deepchem()
            self.use_edge_features = True
        else:
            # PyG 模式
            if not PYG_FROM_SMILES_AVAILABLE:
                print("⚠️  无法使用 from_smiles，已自动切换到完整模式")
                self.use_edge_features = True
            else:
                self.use_edge_features = use_edge_features

    def _init_deepchem(self):
        """初始化 DeepChem 的图转换器"""
        try:
            from deepchem.feat import ConvMolFeaturizer, WeaveFeaturizer
        except ImportError:
            raise ImportError("使用 DeepChem 模式需要安装 deepchem: pip install deepchem")
        
        if self.model_type == 'graphconv':
            self.featurizer = ConvMolFeaturizer()
        elif self.model_type == 'mpnn':
            self.featurizer = WeaveFeaturizer()
        else:
            raise ValueError(f"未知的 DeepChem 模型类型: {self.model_type}")
            
    def _remove_edge_attr(self, data: Data):
        # 统一删除重复的 edge_attr（如果存在）
            if 'edge_attr' in data:
                del data['edge_attr']  # 从 Data 字典删除
            if hasattr(data, 'edge_attr'):
                delattr(data, 'edge_attr')  # 从属性删除
            if hasattr(data, '_store') and 'edge_attr' in data._store:
                del data._store['edge_attr']  # 从内部存储删除

    def _with_edge_features(self, smiles: str, add_self_loops: bool = True) -> Data:
        """
        使用 RDKit 实现完整转化（带边特征）
        
        Args:
            smiles: SMILES 字符串
            add_self_loops: 如果分子没有化学键，是否添加自环边（默认True）
        """
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            if not self.handle_errors:
                raise ValueError(f"无效 SMILES: {smiles}")
            return self._create_empty_graph("RDKit failed to parse SMILES")
        
        try:
            # 1. 构建节点特征
            atom_features = []
            for atom in mol.GetAtoms():
                feat = self._atom_to_features(atom)
                atom_features.append(feat)
            
            if len(atom_features) == 0:
                return self._create_empty_graph("分子没有原子")
                
            x = torch.tensor(np.array(atom_features, dtype=np.float32), dtype=torch.float32)

            # 2. 构建边索引和边特征
            edge_index = []
            edge_attr = []
            
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index.extend([[i, j], [j, i]])  # 无向图
                
                if self.use_edge_features:
                    bond_feat = self._bond_to_features(bond)
                    edge_attr.extend([bond_feat, bond_feat])  # 对称边特征
            
            n_atoms = x.shape[0]
            
            # 关键修改：处理无边的情况（孤立节点或单原子分子）
            if len(edge_index) == 0:
                if add_self_loops and n_atoms > 0:
                    # 策略1：添加自环边，让图卷积可以处理
                    print(f"⚠️  分子 {smiles[:30]}... 无化学键，添加自环边（{n_atoms}个节点）")
                    edge_index = [[i, i] for i in range(n_atoms)]
                    
                    if self.use_edge_features:
                        # 自环边的虚拟特征 [0,0,0,0] 表示"无化学键"
                        edge_attr = [[0, 0, 0, 0] for _ in range(n_atoms)]
                else:
                    # 策略2：如果不允许自环，返回空图（会被过滤）
                    return self._create_empty_graph("分子无化学键且不允许自环")
            
            # 转换为 tensor
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            if self.use_edge_features:
                edge_attr = torch.tensor(np.array(edge_attr, dtype=np.float32), dtype=torch.float32)
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)
            else:
                edge_attr = None
        
            # 3. 创建 Data 对象
            data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_attr=edge_attr,
                num_nodes=n_atoms
            )
            data.is_valid = True
            data.has_real_bonds = len(mol.GetBonds()) > 0  # 标记是否真的有化学键
            return data
            
        except Exception as e:
            if not self.handle_errors:
                raise ValueError(f"处理 SMILES 时出错: {smiles}") from e
            return self._create_empty_graph(f"Error during feature extraction: {e}")

    def _deepchem_to_graph(self, smiles: str, add_self_loops: bool = True) -> Data:
        """
        DeepChem 特征化 → PyG Data
        """
        import torch_geometric.data as pyg_data
        
        # 特征化
        mol_list = self.featurizer.featurize([smiles])
        if len(mol_list) == 0 or mol_list[0] is None:
            raise ValueError("DeepChem 特征化失败")
        
        mol = mol_list[0]
        
        # 提取原子特征
        if self.model_type == 'graphconv':
            # ConvMol
            atom_features = mol.get_atom_features()
            x = torch.tensor(atom_features, dtype=torch.float32)
            n_atoms = x.size(0)
            
            if n_atoms == 0:
                return self._create_empty_graph("DeepChem: 无原子")
            
            # 构建 edge_index
            adj_list = mol.get_adjacency_list()
            edges = []
            for src, neighbors in enumerate(adj_list):
                for dst in neighbors:
                    edges.append([src, dst])
            
            # 关键修改：处理无边的情况
            if len(edges) == 0:
                if add_self_loops:
                    print(f"⚠️  DeepChem 分子 {smiles[:30]}... 无邻接边，添加自环边")
                    edges = [[i, i] for i in range(n_atoms)]
                else:
                    return self._create_empty_graph("DeepChem: 无邻接边")
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            data = pyg_data.Data(
                x=x,
                edge_index=edge_index,
                num_nodes=n_atoms
            )
            
        elif self.model_type == 'mpnn':
            # WeaveMol
            atom_features = mol.get_atom_features()
            x = torch.tensor(atom_features, dtype=torch.float32)
            n_atoms = x.size(0)
            
            if n_atoms == 0:
                return self._create_empty_graph("DeepChem MPNN: 无原子")
            
            # Weave 的边处理
            if n_atoms > 1:
                # 创建全连接边（Weave 的特性）
                rows, cols = [], []
                for i in range(n_atoms):
                    for j in range(n_atoms):
                        if i != j:
                            rows.append(i)
                            cols.append(j)
                edge_index = torch.tensor([rows, cols], dtype=torch.long)
            else:
                # 单原子，添加自环
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
            data = pyg_data.Data(
                x=x,
                edge_index=edge_index,
                num_nodes=n_atoms
            )
        
        data.is_valid = True
        data.has_real_bonds = True  # DeepChem 通常处理的是有键的分子
        return data


    def smiles_to_graph(self, smiles: str, add_self_loops: bool = True) -> Data:
        """
        把 SMILES 转成图数据，返回 PyG 的 Data 对象
        
        Args:
            smiles: SMILES 字符串
            add_self_loops: 对于无化学键的分子，是否添加自环边（默认True）
        """
        try:
            if self.backend == 'deepchem':
                data = self._deepchem_to_graph(smiles, add_self_loops=add_self_loops)
            else:
                if self.use_edge_features:
                    data = self._with_edge_features(smiles, add_self_loops=add_self_loops)
                else:  # 快速模式，直接用 PyG 内置函数
                    if PYG_FROM_SMILES_AVAILABLE:
                        data = from_smiles(smiles)
                        self._remove_edge_attr(data)
                        
                        # 检查并添加自环边（针对 from_smiles 可能返回无边图的情况）
                        if add_self_loops and hasattr(data, 'edge_index'):
                            if data.edge_index.shape[1] == 0 and data.x.shape[0] > 0:
                                n_atoms = data.x.shape[0]
                                print(f"⚠️  from_smiles 结果无边，添加自环边: {smiles[:30]}...")
                                self_loops = torch.arange(n_atoms, dtype=torch.long).unsqueeze(0).repeat(2, 1)
                                data.edge_index = self_loops
                        
                        # 确保有 num_nodes 和 is_valid
                        if not hasattr(data, 'num_nodes'):
                            data.num_nodes = data.x.shape[0]
                        data.is_valid = True
                    else:
                        data = self._with_edge_features(smiles, add_self_loops=add_self_loops)
            
            return data
            
        except Exception as e:
            if not self.handle_errors:
                raise
            print(f"⚠️  转换 SMILES 失败: {smiles[:30]}..., 原因: {e}")
            return self._create_empty_graph(str(e))

    def _create_empty_graph(self, reason: str = "") -> Data:
        """
        创建空图，作为无效 SMILES 的占位符
        注意：这个图会被过滤掉，不会进入训练
        """
        dim = 9 if not self.use_edge_features else getattr(self, 'node_feat_dim', 9)
        
        # 创建1个节点+1条自环边，避免完全为空导致错误
        # 但标记为 is_valid=False，让上层过滤
        data = Data(
            x=torch.zeros((1, dim), dtype=torch.float), 
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # 自环边
            num_nodes=1
        )
        data.is_valid = False
        data._invalid_reason = reason
        data.has_real_bonds = False
        return data

    def batch_convert(self, smiles_list: list[str], fit_scaler: bool = True, 
                    min_nodes: int = 1, min_edges: int = 1) -> tuple[list[Data], list[int]]:
        """
        批量转换 SMILES 列表，返回有效图和原始索引映射
        
        Args:
            smiles_list: SMILES 列表
            fit_scaler: 是否拟合标准化器
            min_nodes: 最小节点数（默认1）
            min_edges: 最小边数（默认1，自环边也算）
        
        Returns:
            (valid_graphs, valid_indices): 有效图列表和对应的原始索引
        """
        graphs = []
        valid_indices = []
        
        for i, smi in enumerate(smiles_list):
            try:
                graph = self.smiles_to_graph(smi, add_self_loops=True)  # 默认添加自环
                
                # 严格验证
                is_valid = (
                    getattr(graph, 'is_valid', True) and
                    hasattr(graph, 'x') and graph.x is not None and
                    graph.x.shape[0] >= min_nodes and
                    hasattr(graph, 'edge_index') and graph.edge_index is not None and
                    graph.edge_index.shape[1] >= min_edges  # 至少 min_edges 条边
                )
                
                if is_valid:
                    graphs.append(graph)
                    valid_indices.append(i)
                else:
                    reason = getattr(graph, '_invalid_reason', 'Unknown')
                    print(f"⚠️  过滤掉无效图 (索引 {i}): {smi[:30]}..., 原因: {reason}")
                    
            except Exception as e:
                print(f"⚠️  转换 SMILES 时出错 (索引 {i}): {smi[:30]}..., 原因: {e}")
                # 不添加到 graphs，索引也不会进入 valid_indices

        print(f"批量转换完成: {len(smiles_list)} 个 SMILES -> {len(graphs)} 个有效图 "
            f"({len(smiles_list) - len(graphs)} 个无效)")
        
        if len(graphs) == 0:
            raise ValueError("所有 SMILES 都转换失败，请检查数据格式")
        
        # 标准化处理（仅对有效图）
        if self.normalize:
            all_node_feats = torch.cat([g.x for g in graphs], dim=0).numpy()
            if fit_scaler or self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                scaled_feats = self.scaler.fit_transform(all_node_feats)
            else:
                scaled_feats = self.scaler.transform(all_node_feats)
            
            # 为每个图设置标准化后的特征
            idx = 0
            for g in graphs:
                end_idx = idx + g.x.shape[0]
                g.x = torch.tensor(scaled_feats[idx:end_idx], dtype=torch.float32)
                idx = end_idx
        
        return graphs, np.array(valid_indices)  # 返回索引映射，方便同步过滤标签
    
    def invalid_filter(self, data_list: list[Data]) -> tuple[list[Data],list[int]]:
        """过滤掉无效图"""
        valid_graphs = []
        valid_indices = []
        for i, data in enumerate(data_list):
            # 检查：有节点、有x特征、标记为有效
            is_valid = (getattr(data, 'is_valid', True) and 
                    hasattr(data, 'x') and 
                    data.x is not None and 
                    data.x.shape[0] > 0)
            if is_valid:
                valid_graphs.append(data)
                valid_indices.append(i)
        return valid_graphs, valid_indices
        
    def _atom_to_features(self, atom) -> np.ndarray:
        """把 RDKit 原子转成 one-hot 特征向量"""
        features = []
        # 原子序数
        atom_type = atom.GetAtomicNum()
        features.extend(self._one_hot(atom_type, self.ATOM_FEATURES['atomic_num']))
        # 度数
        degree = atom.GetDegree()
        features.extend(self._one_hot(degree, self.ATOM_FEATURES['degree']))
        # 形式电荷
        charge = atom.GetFormalCharge()
        features.extend(self._one_hot(charge, self.ATOM_FEATURES['formal_charge']))
        # 芳香性
        features.append(int(atom.GetIsAromatic()))
        # 杂化类型
        hyb = atom.GetHybridization()
        features.extend(self._one_hot(hyb, self.ATOM_FEATURES['hybridization']))
        # ... 可添加更多原子特征
        return np.array(features, dtype=np.float32)

    def _bond_to_features(self, bond) -> np.ndarray:
        """把 RDKit 键转成特征向量"""
        features = []
        bond_type = bond.GetBondType()
        bt_one_hot = [0.0, 0.0, 0.0, 0.0]  # SINGLE, DOUBLE, TRIPLE, AROMATIC
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bt_one_hot[0] = 1.0
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bt_one_hot[1] = 1.0
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bt_one_hot[2] = 1.0
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bt_one_hot[3] = 1.0
        features.extend(bt_one_hot)
        features.append(int(bond.IsInRing()))
        features.append(int(bond.GetIsConjugated()))
        # ... 可添加更多边特征
        return np.array(features, dtype=np.float32)
    
    def _one_hot(self, value: int, allowable_values: list[int]) -> list[int]:
        """把一个值转成 one-hot 向量"""
        if value not in allowable_values:
            value = allowable_values[-1]  # 超出范围的值归到最后一类
        return [int(value == v) for v in allowable_values]
    
