import torch, numpy as np
import torch.nn as nn
from base import BenchModel
try:
    from schnetpack import SchNet, AtomsData
except ImportError:
    raise ImportError('请先安装 schnetpack')

# 输入：3D 坐标 (x,y,z)
class SchNetWrap(BenchModel):
    """
    切换任务类型（回归 / 二分类 / 多分类）教程
    ------------------------------------------------
    仅需改 3 处，其余代码不动：

    1. 输出维度  
    regression / binary  : 1  
    multiclass (C 类)    : C  

    2. 损失函数  
    regression : nn.MSELoss()  
    binary     : nn.BCEWithLogitsLoss()  
    multiclass : nn.CrossEntropyLoss()  

    3. 预测后处理  
    regression : 直接返回 logit  
    binary     : sigmoid 后 > 0.5 → 1  
    multiclass : softmax 后 argmax → 类别索引  
    ------------------------------------------------
    """
    def __init__(self, n_atom_basis=128, n_interactions=6, cutoff=5.0, lr=1e-3,
                 task_type='regression', num_classes=None, **kw):
        self.params = dict(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            cutoff=cutoff, lr=lr, 
            task_type=task_type, 
            num_classes=num_classes,
            **kw)
        self.task_type = task_type

        if task_type == 'regression':
            self.output_dim = 1
            self.loss_fn = nn.MSELoss()
        elif task_type == 'binary':
            self.output_dim = 1
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif task_type == 'multiclass':
            if num_classes is None or num_classes < 2:
                raise ValueError("多分类必须指定 num_classes ≥ 2")
            self.output_dim = num_classes
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"未知任务类型: {task_type}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 构建 SchNet 模型
        self.model = SchNet(n_atom_basis=n_atom_basis,
                            n_interactions=n_interactions,
                            cutoff=cutoff)
        
        self.output_layer = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_atom_basis//2, self.output_dim))
        # 将模型和输出层移到设备
        self.model.to(self.device)
        self.output_layer.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + 
            list(self.output_layer.parameters()), 
            lr=lr)

    def _forward(self, batch):
        """统一前向传播"""
        # SchNet 输出原子级特征 (n_atoms, n_atom_basis)
        atomic_features = self.representation(batch)
        
        # 需要聚合到分子级（平均池化）
        # 假设 batch 中有 'batch' 字段指示原子属于哪个分子
        if 'batch' in batch:
            # 使用 PyG 的 global_mean_pool 或手动实现
            from torch_scatter import scatter_mean
            molecular_features = scatter_mean(
                atomic_features, 
                batch['batch'], 
                dim=0
            )
        else:
            # 单分子，直接平均
            molecular_features = atomic_features.mean(dim=0, keepdim=True)
        
        # 输出层
        output = self.output_layer(molecular_features)
        return output

    def fit(self, X, y):
        """
        X: 原子坐标或 Atoms 对象列表
        y: 目标值 (n_mol,) 或 (n_mol, 1)
        """
        # 确保 y 是正确的形状
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # 创建数据集
        ds = AtomsData(X, y)  # 需要确保这个类存在且正确
        loader = torch.utils.data.DataLoader(
            ds, batch_size=32, shuffle=True, collate_fn=self._collate_fn
        )
        
        self.model.train()
        self.output_layer.train()
        
        for epoch in range(self.params.get('epochs', 100)):
            total_loss = 0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                pred = self._forward(batch)
                
                # 确保维度匹配
                target = batch['y'].view(-1, self.output_dim) if self.task_type != 'multiclass' else batch['y'].long().view(-1)
                
                loss = self.loss_fn(pred, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
        
        return self

    def _collate_fn(self, batch_list):
        """自定义 collate_fn 来处理图数据"""
        from schnetpack.data import collate_atomsdata
        return collate_atomsdata(batch_list)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        self.output_layer.eval()

        with torch.no_grad():
            ds = AtomsData(X, y=None)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=128, collate_fn=self._collate_fn)
            preds = []
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                if self.task_type == 'regression':
                    out = outputs.squeeze().cpu().numpy()
                elif self.task_type == 'binary':
                    out = (torch.sigmoid(outputs) >= 0.5).int().squeeze().cpu().numpy()
                else:  # multiclass
                    out = outputs.argmax(dim=1).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0)

    def get_params(self, deep=True):
        return self.params.copy()
    
    def get_task_type(self):
        return self.task_type
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """分类返回概率，回归返回 None"""
        if self.task_type == 'regression':
            return None
        
        self.model.eval()
        self.output_layer.eval()
        
        with torch.no_grad():
            ds = AtomsData(X, y=np.zeros(len(X)))
            loader = torch.utils.data.DataLoader(
                ds, batch_size=128, collate_fn=self._collate_fn)
            probs = []
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                if self.task_type == 'binary':
                    prob = torch.sigmoid(outputs).cpu().numpy()
                else:  # multiclass
                    prob = torch.softmax(outputs, dim=1).cpu().numpy()
                probs.append(prob)
        return np.concatenate(probs, axis=0)
    
    def save(self, path: str) -> None:
        """保存模型"""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'output_layer': self.output_layer.state_dict(),
            'params': self.params
        }, os.path.join(path, 'schnet_model.pt'))
        print(f"SchNet 模型已保存到 {path}")
    
    def load(self, path: str) -> "SchNetWrap":
        """从路径加载模型"""
        checkpoint = torch.load(os.path.join(path, 'schnet_model.pt'), 
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.output_layer.load_state_dict(checkpoint['output_layer'])
        self.params = checkpoint['params']

        self.model.to(self.device)
        self.output_layer.to(self.device)
        print(f"SchNet 模型已从 {path} 加载")
        return self

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
    from model_register import register_model
    register_model(SchNetWrap, task_type='regression',
                   save_dir=r'molbench\core\hyper_parameters\test', 
                   protocol='bench', model_name='SchNet')
    
