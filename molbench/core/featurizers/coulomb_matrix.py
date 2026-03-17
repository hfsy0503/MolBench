from .base import BaseFeaturizer
import numpy as np

class CoulombFeaturizer(BaseFeaturizer):
    """Coulomb Matrix 描述符:
    Provides a representation of the electronic structure of a molecule.
    Parameters
    ----------
    flatten : bool
        True  -> 返回压平向量 (n_atoms^2,)
        False -> 返回原始方阵 (n_atoms, n_atoms)
    max_atoms : int
        统一矩阵尺寸，不足补零，超出截断
    """
    def __init__(self, max_atoms: int=50, flatten: bool= True):
        self.max_atoms= max_atoms
        self.flatten= flatten

    def transform(self, smiles_list):
        try:
            import deepchem as dc
            from rdkit.Chem import MolFromSmiles
        except ModuleNotFoundError:
            raise ImportError("CoulombFeaturizer 需要 deepchem 和 Rdkit 环境。")
        
        # 转换并过滤无效分子，用占位符替代
        mols = []
        valid_mask = []
        for smi in smiles_list:
            mol = MolFromSmiles(smi)
            if mol is not None and mol.GetNumAtoms() > 0:
                mols.append(mol)
                valid_mask.append(True)
            else:
                # 使用甲烷作为占位符（避免空分子）
                placeholder = MolFromSmiles('C')
                mols.append(placeholder)
                valid_mask.append(False)
        
        if len(mols) == 0:
            shape = (0, self.max_atoms * self.max_atoms) if self.flatten else (0, self.max_atoms, self.max_atoms)
            return np.zeros(shape, dtype=np.float32)
        
        cm = dc.feat.CoulombMatrix(max_atoms=self.max_atoms)
        try:
            fps = cm.featurize(mols)
        except Exception:
            # 逐个处理以防批量失败
            fps = [cm.featurize([m])[0] for m in mols]

        if isinstance(fps, list) or (isinstance(fps, np.ndarray) and fps.dtype == object):
            fps = self._enforce_shape(fps)
        else:
            fps = np.array(fps)
            # 确保是3维 (n_samples, max_atoms, max_atoms)
            if len(fps.shape) == 2:
                fps = fps.reshape(-1, self.max_atoms, self.max_atoms)
        
        # 压平
        if self.flatten:
            fps = fps.reshape(fps.shape[0], -1)
        
        # 将无效分子的特征置零（避免污染数据）
        fps = fps.astype(np.float32)
        for i, valid in enumerate(valid_mask):
            if not valid:
                fps[i] = 0
        
        return fps
    
    def _enforce_shape(self, features):
        """强制所有特征为统一形状 (max_atoms, max_atoms)"""
        standardized = []
        for feat in features:
            feat = np.array(feat)
            # 压平并截断/填充
            flat = feat.flatten()
            target_size = self.max_atoms * self.max_atoms
            
            if len(flat) < target_size:
                flat = np.pad(flat, (0, target_size - len(flat)), 'constant')
            else:
                flat = flat[:target_size]
            
            standardized.append(flat.reshape(self.max_atoms, self.max_atoms))
        
        return np.array(standardized)