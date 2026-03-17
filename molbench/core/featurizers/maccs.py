from .base import BaseFeaturizer
import numpy as np

class MACCSFeaturizer(BaseFeaturizer):
    def __init__(self, n_bits=167):
        self.n_bits= n_bits

    def transform(self, smiles_list):
        """
        MACCS keys 指纹
        Parameters
        ----------
        smiles_list : list[str]
            SMILES 字符串列表

        Returns
        -------
        np.ndarray
            shape = (n_molecules, 167) 的 0/1 矩阵
        """
        # 延迟导入，函数被调用时才加载 RDKit
        try:
            from rdkit import Chem
            from rdkit.Chem.AllChem import GetMACCSKeysFingerprint
        except ModuleNotFoundError:
            raise ImportError("MACCSFeaturizer 需要 RDKit 环境。")

        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(np.array(GetMACCSKeysFingerprint(mol)))
            else:                       # 无效分子用 0 向量填充
                fps.append(np.zeros(self.n_bits, dtype=np.uint8))
        return np.array(fps, dtype=np.uint8)