import torch, torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from molbench.core.adapters.smiles_to_graph import GraphConverter

class BenchGNN(nn.Module, BaseEstimator):
    """引入多种 GNN，继承自 nn.Module，适配 BenchModel 接口"""
    Supported_GNNs = ['GCN', 'GIN', 'GAT', 'TransformerConv', 
                      'GraphConv', 'MPNN', 'GraphSAGE']  
    BACKEND_MAP = {
        'GCN': 'pyg',
        'GIN': 'pyg',
        'GAT': 'pyg',
        'TransformerConv': 'pyg',
        'GraphConv': 'deepchem',
        'MPNN': 'deepchem',
        'GraphSAGE' :'pyg'
    }
    _PARAM_SPECS = {
        'hidden_dim': (64, int),
        'out_dim': (1, int),
        'model_name': ('GCN', str),
        'lr': (1e-3, float),
        'epochs': (200, int),
        'task_type': ('binary', str),
        'num_layers': (2, int),
        'num_heads': (4, int),
        'batch_size': (8, int),
        'dropout': (0.0, float),
        'random_state': (42, int),
        'use_edge_features': (False, bool),
        'concat': (True, bool),           # TransformerConv: 是否拼接多头输出
        'beta': (True, bool),             # TransformerConv: 是否使用门控残差
        'aggr': ('mean', str),
        'node_feat_dim': (39, int),       # MPNN: 节点特征维度
        'edge_feat_dim': (10, int),       # MPNN: 边特征维度
        'num_step_message_passing': (6, int),   # MPNN: 消息传递步数
    }

    def __init__(self, **kw):
        super().__init__()
        for param, (default, typ) in self._PARAM_SPECS.items():
            val = kw.get(param, default)
            if isinstance(val, dict):
                print(f"警告: 参数 {val} 是 dict，使用默认值 {default}")
                val = default
            try: # 强制类型转换
                setattr(self, param, typ(val))
            except Exception as e:
                print(f"错误: 无法将参数 {param} 的值 {val} 转换为 {typ}: {e}, 使用默认值 {default}")
                setattr(self, param, typ(default))
                 
        # 验证模型名称
        if self.model_name not in self.Supported_GNNs:
            raise ValueError(f"不支持的 GNN 模型: {self.model_name}. 可选: {self.Supported_GNNs}")

        self.backend = self.BACKEND_MAP[self.model_name]

        self._set_seed(self.random_state)  # 设置随机种子，确保可复现

        self.params = {name: getattr(self, name) for name in self._PARAM_SPECS.keys()} # 构建参数字典，供 get_params 使用
        self._extra_params = {k: v for k, v in kw.items() if k not in self._PARAM_SPECS}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.backend == 'deepchem':
            self.use_edge_features = True  # DeepChem 模式强制完整特征

        # 图转换器
        self.graph_converter = GraphConverter(
            model_type=self._get_converter_model_type(),
            use_edge_features=self.use_edge_features)
    
        # 网络结构
        self.in_dim = None  # 输入维度，训练时根据数据自动推断
        self.convs = None  # 延迟初始化
        self.head = None   # 最后输出层
        self.bns = None     # BatchNorm 层（仅 DeepChem 模式）
        self.pool = global_mean_pool  # 图级任务的池化方法
        self._needs_rebuild = False
        self.is_fitted_ = False
        self.concat = True

        self.to(self.device)

        # 损失函数
        if self.task_type == 'regression':
            self.loss_fn = torch.nn.MSELoss()
        elif self.task_type == 'binary':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:  # multiclass
            self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def _set_seed(self, seed):
        import random, numpy as np, torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 确定性算法（可能降低速度，但确保可复现）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_converter_model_type(self):
        """映射到 GraphConverter 的 model_type"""
        mapping = {
            'GCN': 'pyg_fast',
            'GIN': 'pyg_fast',
            'GAT': 'pyg_fast',
            'TransformerConv': 'pyg_fast',
            'GraphConv': 'graphconv',
            'MPNN': 'mpnn',
            'GraphSAGE': 'pyg_fast'
        }
        return mapping[self.model_name]
    
    def _build_layers(self, in_dim: int, num_classes: int = None):
        """根据输入维度和任务类型构建 GNN 层和输出层"""
        if self.backend == 'pyg':
            self._build_pyg_layers(in_dim, num_classes)
        else:
            self._build_deepchem_layers(in_dim, num_classes)

    def _build_pyg_layers(self, in_dim: int, num_classes: int= None):
        """构建 GNN 层"""
        out_dim = 1 if self.task_type != 'multiclass' else (num_classes or self.out_dim)
        self.convs = nn.ModuleList()  # 延迟初始化

        if self.model_name in ['GAT', 'TransformerConv']:
            adjusted_hidden = (self.hidden_dim // self.num_heads) * self.num_heads
            if adjusted_hidden != self.hidden_dim:
                self.hidden_dim = adjusted_hidden
                print(f"Adjusted hidden_dim to {adjusted_hidden} to be divisible by num_heads={self.num_heads}")

        hidden = self.hidden_dim
        current_dim = in_dim

        for i in range(self.num_layers):
            in_ch = current_dim if i == 0 else hidden

            if self.model_name == 'GCN':
                from torch_geometric.nn import GCNConv
                conv = GCNConv(in_ch, hidden)
            elif self.model_name == 'GIN':
                from torch.nn import Sequential, Linear, ReLU
                from torch_geometric.nn import GINConv
                mlp = Sequential(Linear(in_ch, hidden), ReLU(), Linear(hidden, hidden))
                conv = GINConv(mlp)
            elif self.model_name == 'GAT':
                from torch_geometric.nn import GATConv
                head_dim = hidden // self.num_heads
                conv = GATConv(
                    in_ch, 
                    head_dim,
                    heads=self.num_heads, 
                    concat=True, 
                    dropout=self.dropout)
                if not self.concat:
                    self.concat = True
            elif self.model_name == 'TransformerConv':
                from torch_geometric.nn import TransformerConv
                head_dim = hidden // self.num_heads
                conv = TransformerConv(
                    in_ch, 
                    head_dim,
                    heads=self.num_heads, 
                    concat=True, 
                    beta=self.beta,
                    dropout=self.dropout)
                if not self.concat:
                    self.concat = True
            elif self.model_name == 'GraphSAGE':
                from torch_geometric.nn import SAGEConv
                conv = SAGEConv(
                    in_channels=in_ch,
                    out_channels=hidden,
                    aggr=self.aggr,
                    normalize=False,  
                    root_weight=True, 
                    bias=True
                )
            else:
                raise ValueError(f"未知模型名称 {self.model_name}, 可选项: {self.Supported_GNNs}")
        
            self.convs.append(conv.to(self.device))
            current_dim = hidden
        self.head = nn.Linear(hidden, out_dim).to(self.device)  

    def _build_deepchem_layers(self, in_dim: int, num_classes: int = None):
        """构建 DeepChem 特征的处理层"""
        # DeepChem 特征已经是向量，用简单 MLP 或 GCN 处理
        from torch_geometric.nn import GCNConv, NNConv
        hidden = self.hidden_dim
        
        self.convs = nn.ModuleList()  # 延迟初始化
        self.bns = nn.ModuleList()  # BatchNorm 层
        
        for i in range(self.num_layers):
            in_ch = hidden if i > 0 else in_dim
            
            if self.model_name == 'MPNN':
                edge_nn = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, in_ch*hidden))
                conv = NNConv(in_ch, hidden, edge_nn, aggr=self.aggr)
            elif self.model_name == 'GraphConv':
                from torch_geometric.nn import GraphConv
                conv = GraphConv(in_ch, hidden, aggr=self.aggr, bias=True)
            else:
                conv = GCNConv(in_ch, hidden)
            self.convs.append(conv.to(self.device))
            self.bns.append(nn.BatchNorm1d(hidden).to(self.device))
        out_dim = 1 if self.task_type != 'multiclass' else (num_classes or self.out_dim)
        self.head = nn.Linear(hidden, out_dim).to(self.device)  # 输出层

    def _prepare_data(self, X, y=None):
        """准备数据"""
        if isinstance(X, np.ndarray):
            X = X.tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()

        if isinstance(X, (list, )) and len(X) > 0 and isinstance(X[0], str): 
            graphs = self.graph_converter.batch_convert(X, fit_scaler=True)
            valid_indices = list(range(len(graphs)))
        else:
            graphs, valid_indices = self.graph_converter.invalid_filter(X)
            print(f"输入图列表过滤: {len(X)} -> {len(graphs)} 个有效图") 

        if len(graphs) == 0:
            raise ValueError("输入数据为空，无法构建图数据")

        if y is not None:
            if isinstance(y, (pd.Series, list)):
                y = np.array(y)
            y = np.asarray(y).ravel()

            if len(valid_indices) != len(X):
                y = y[valid_indices]  # 只保留有效图对应的标签
                print(f"同步过滤 y: {len(X)} -> {len(valid_indices)}")
                
            assert len(y) == len(graphs), f"图数量 {len(graphs)} 与标签数量 {len(y)} 不匹配！"

            for i, graph in enumerate(graphs):
                if self.task_type in ['regression', 'binary']:
                    graph.y = torch.tensor(y[i], dtype=torch.float32)
                elif self.task_type == 'multiclass':  
                    graph.y = torch.tensor(y[i], dtype=torch.long)
                else:
                    raise ValueError(f"未知任务类型: {self.task_type}")
                graph.num_graphs =1

        for i, g in enumerate(graphs):
            assert hasattr(g, 'x'), f"图 {i} 缺少 x"
            assert g.x.shape[0] > 0, f"图 {i} 节点数为 0"
            assert hasattr(g, 'edge_index'), f"图 {i} 缺少 edge_index"
            assert g.edge_index.shape[1] > 0, f"图 {i} 边数为 0"
            assert hasattr(g, 'y'), f"图 {i} 缺少标签 y"
            if not hasattr(g, 'num_nodes') or g.num_nodes is None:
                g.num_nodes = g.x.shape[0]
        
        print(f"_prepare_data 输出: {len(graphs)} 个有效图，每个图节点数范围: "
            f"[{min(g.x.shape[0] for g in graphs)}, {max(g.x.shape[0] for g in graphs)}]")
        print(f"所有 {len(graphs)} 个图验证通过")

        return graphs

    def fit(self, X, y):
        """标准接口：从SMILES训练"""
        # 1. 数据验证和转换（保留原有检查）
        X, y = self._validate_input(X, y)
        
        # 2. 转换为图（使用batch_convert获取索引）
        graphs, indices = self.graph_converter.batch_convert(X, fit_scaler=True)
        y_filtered = y[indices]
        
        print(f"训练数据过滤: {len(X)} -> {len(graphs)} 个有效图")
        if len(graphs) == 0:
            raise ValueError("没有有效的图数据")
        
        # 3. 调用核心训练
        return self.fit_graphs(graphs, y_filtered)
    
    def _validate_input(self, X, y):
        """验证和标准化输入数据（提取自原fit）"""
        # X处理
        if isinstance(X, np.ndarray):
            if X.ndim == 1 and len(X) > 0 and isinstance(X[0], str):
                X = X.tolist()
            else:
                raise ValueError(f"图模型输入 X 必须是 SMILES 列表，got shape {X.shape}")
        elif isinstance(X, pd.Series):
            X = X.tolist()
        elif not isinstance(X, list):
            raise ValueError(f"图模型输入 X 必须是 SMILES 列表，got type {type(X)}")
        
        # y处理
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.reset_index(drop=True)
        if isinstance(y, (pd.Series, list)):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            y = np.array(y)
        y = y.ravel()
        
        # 检查
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X 或 y 为空")
        if len(X) != len(y):
            raise ValueError(f"X 和 y 长度不匹配: {len(X)} vs {len(y)}")
        
        return X, y
    
    def fit_graphs(self, graphs, y):
        """核心训练：从预转换的图训练（简化版）"""
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # 网络初始化（只执行一次）
        if self.convs is None or getattr(self, '_needs_rebuild', False):
            sample = graphs[0]
            in_dim = sample.x.shape[1] if len(sample.x.shape) > 1 else 1
            self.in_dim = in_dim
            
            num_classes = len(set(y)) if self.task_type == 'multiclass' else None
            self._build_layers(in_dim, num_classes)
            print(f"第一层 conv: {type(self.convs[0]).__name__}")
            self._needs_rebuild = False
        
        # 添加标签到图
        for g, label in zip(graphs, y):
            if self.task_type in ['regression', 'binary']:
                g.y = torch.tensor(label, dtype=torch.float32)
            else:
                g.y = torch.tensor(label, dtype=torch.long)
        
        # 构建DataLoader并训练...
        from torch_geometric.loader import DataLoader
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=True,
                          pin_memory=torch.cuda.is_available())
        
       # 测试 forward
        test_batch = next(iter(loader))
        print(f"batch edge_attr: {getattr(test_batch, 'edge_attr', 'MISSING')}")

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params['lr'])

        for epoch in range(self.epochs):          # 可改成超参
            self._set_seed(self.random_state + epoch)  # 每轮不同随机种子
            total_loss = 0
            num_batches = 0
            skipped_batches = 0

            for batch_idx, batch_data in enumerate(loader):
                batch_data = batch_data.to(self.device)
                # 调试
                if not hasattr(batch_data, 'batch') or batch_data.batch is None:
                    print(f"警告：Batch {batch_idx}: batch_data.batch is None!")
                    skipped_batches +=1
                    continue
                
                num_graphs_in_batch = batch_data.num_graphs

                optimizer.zero_grad()
                
                out = self.forward(batch_data.x, batch_data.edge_index, batch_data.batch,
                                getattr(batch_data, 'edge_attr', None))
                
                if out.shape[0] != num_graphs_in_batch:
                    print(f"严重警告: Batch {batch_idx} 输出图数量 {out.shape[0]} 与输入图数量 {num_graphs_in_batch} 不匹配！")
                    # 跳过这个 batch 而不是截断
                    skipped_batches +=1
                    continue

                if self.task_type in ['regression', 'binary']:
                    out = out.view(-1).float()  # 确保输出是一维的
                    target = batch_data.y.view(-1).float()
                else:  # multiclass 
                    out = out.view(batch_data.num_graphs, -1).float()
                    target = batch_data.y.view(-1).long()
                
                if out.shape[0] != target.shape[0]:
                    print(f"错误: 最终维度不匹配 out={out.shape}, target={target.shape}，跳过此 batch")
                    skipped_batches +=1
                    continue
        
                loss = self.loss_fn(out, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """标准接口：预测SMILES或图列表"""
        if (isinstance(X, list) and len(X) > 0 and 
            hasattr(X[0], 'x') and hasattr(X[0], 'edge_index')):
            # X已经是图列表，直接预测
            graphs = X
            indices = np.arange(len(X))  # 无过滤，顺序索引
            raw_preds = self.predict_graphs(graphs)
        else: # Smiles
            if isinstance(X, np.ndarray):
                X = X.tolist()
            elif isinstance(X, pd.Series):
                X = X.tolist()
            
            graphs, indices = self.graph_converter.batch_convert(X, fit_scaler=False)
            raw_preds = self.predict_graphs(graphs)
        
        if self.task_type == 'binary':
            probs = torch.sigmoid(torch.tensor(raw_preds)).numpy()
            preds = (probs > 0.5).astype(int)
        elif self.task_type == 'multiclass':
            preds = np.argmax(raw_preds, axis=1) if raw_preds.ndim > 1 else raw_preds
        else:
            preds = raw_preds

        if len(preds) != len(X):
            fill_value = np.nan if self.task_type == 'regression' else 0
            full_preds = np.full(len(X), fill_value)
            full_preds[indices] = preds
            return full_preds
        return preds
    
    def predict_graphs(self, graphs):
        """核心预测：从预转换的图预测"""
        if not self.is_fitted_:
            raise RuntimeError("模型未训练")
        if len(graphs) == 0:
            return np.array([])
        
        from torch_geometric.loader import DataLoader
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.forward(batch.x, batch.edge_index, batch.batch,
                                 getattr(batch, 'edge_attr', None))
                preds.extend(out.view(-1).cpu().numpy())
        
        return np.array(preds)
    
    def forward(self, x, edge_index, batch_idx, edge_attr=None):
        """前向传播"""
        if self.convs is None:
            raise RuntimeError(
            "网络层未初始化 (self.convs is None)。"
            "请确保在 predict 之前调用 fit。"
        )
        
        x = x.float().to(self.device)  # 确保输入是浮点数
        edge_index = edge_index.long().to(self.device)  # 确保边索引是长整型
        batch_idx = batch_idx.long().to(self.device)  # 确保 batch 是长整型

        # 使用num_layers循环
        if edge_attr is not None:
            edge_attr = edge_attr.float().to(self.device)  # 确保边特征是浮点数
        
        # 自动构建网络
        if self.convs is None or self.head is None:
            in_dim = x.shape[1]
            self._build_layers(in_dim, None)
            self.to(self.device)

        for i, conv in enumerate(self.convs):
            try:
                if self.model_name == 'TransformerConv' and edge_attr is not None:
                    x = conv(x, edge_index, edge_attr)
                elif self.model_name == 'MPNN' :
                    if edge_attr is None:
                        edge_attr = torch.ones(edge_index.size(1), 1, device=self.device)
                    x = conv(x, edge_index, edge_attr) 
                else:
                    x = conv(x, edge_index)
            except Exception as e:
                print(f"前向传播时卷积层 {i} 失败: {e}")
                print(f"模型：{self.model_name}, 输入维度: x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape if edge_attr is not None else None}")
                raise
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.backend == 'deepchem' and self.bns is not None:
                x = self.bns[i](x)

        x = self.pool(x, batch_idx)
        x = self.head(x)
        if x.dim() ==1:
            x = x.unsqueeze(-1)
        return x
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """分类返回概率，回归返回 None"""
        if self.task_type == 'regression':
            return None
        
        self.eval()
        graphs = self._prepare_data(X, y=None)

        from torch_geometric.loader import DataLoader
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.forward(batch.x, batch.edge_index, batch.batch,
                                      edge_attr=getattr(batch, 'edge_attr', None)).cpu()
                if self.task_type == 'binary':
                    probs_pos = torch.sigmoid(out).squeeze(-1)  # 二分类概率  
                    if probs_pos.dim() == 0:
                        probs_pos = probs_pos.unsqueeze(0)
                    probs_pos = probs_pos.cpu().numpy()
                    probs_neg = 1 - probs_pos
                    probs_2d = np.column_stack([probs_neg, probs_pos])
                    all_probs.append(probs_2d)
                else:  # multiclass
                    probs = F.softmax(out, dim=1)  # 多分类概率
                    if probs.dim() == 1:
                        probs = probs.unsqueeze(0)  # 确保至少2D
                    probs = probs.cpu().numpy()
                    all_probs.append(probs)
        
        if len(all_probs) == 0:
            return np.array([]).reshape(0,2)  # 处理没有数据的情况

        if self.task_type == 'binary':
            return np.vstack(all_probs)
        else:
            return np.concatenate(all_probs)

    def get_params(self, deep=True):
        return self.params.copy()
    
    def set_params(self, **params):
        rebuild = False
        arch_params = ['hidden_dim', 'num_layers', 'num_heads', 'dropout', 'model_name', 'out_dim']
        for k, v in params.items():
            if k in self._PARAM_SPECS:
                default, typ = self._PARAM_SPECS[k]
                if isinstance(v, dict):
                    print(f"警告: 参数 {k} 的值 {v} 是 dict，使用默认值 {default}")
                    v = default
                try:
                    v = typ(v)  
                except Exception as e:
                    print(f"错误: 无法将参数 {k} 的值 {v} 转换为 {typ}: {e}, 使用默认值 {default}")
                    v = typ(default)
                if k in arch_params and getattr(self, k) != v:
                    rebuild = True  # 结构参数变化需要重建网络层
                setattr(self, k, v) 
                self.params[k] = v
    
        if rebuild:
            if self.in_dim is not None:
                num_classes = getattr(self, 'num_classes', None)
                self._build_layers(self.in_dim, num_classes)  # 重新构建网络层，保持输入维度不变
                print(f"结构参数已更新，网络层已重建 (in_dim={self.in_dim})")
            else:
                self.convs = None  # 延迟到 fit 时根据数据自动构建层
                self.head = None
                self._needs_rebuild = True
                
        return self

    def get_task_type(self):
        return self.task_type
    
    def load(self, path):
        """加载模型参数，恢复到之前训练好的状态"""
        checkpoint = torch.load(path, map_location=self.device)
       
        if 'in_dim' in checkpoint:
            self.in_dim = checkpoint['in_dim']

        for k, v in checkpoint.get('params', {}).items():
            if k in self._PARAM_SPECS:
                setattr(self, k, v)
                self.params[k] = v
        
        if 'task_type' in checkpoint:
            self.task_type = checkpoint['task_type']

        if 'num_classes' in checkpoint:
            self.num_classes = checkpoint['num_classes']

        if self.in_dim is not None:
            num_classes = getattr(self, 'num_classes', None)
            self._build_layers(self.in_dim, num_classes)

        self.load_state_dict(checkpoint['model_state'])
        print(f"模型已从 {path} 加载")
        return self
    
    def save(self, path):
        """保存模型参数，供后续加载"""
        torch.save({'model_state': self.state_dict(),
                    'params': self.params,
                    'in_dim': self.in_dim,
                    'num_classes': getattr(self, 'num_classes', None),
                    'task_type': self.task_type}, path)
        print(f"模型已保存到 {path}")
    
    def __sklearn_tags__(self):
        """兼容 sklearn 1.3+"""
        try:
            from sklearn.utils._tags import Tags
            if self.task_type == 'regression':
                estimator_type = "regressor"
            else:
                estimator_type = "classifier"
            return Tags(
                estimator_type=estimator_type,
                target_tags ={"requires_y": True},
                requires_fit=True,
            )
        except ImportError:
            # sklearn < 1.3
            return {}
