from .base import BaseFeaturizer
import numpy as np

class ECFPFeaturizer(BaseFeaturizer):
    """ECFP(Morgan)指纹"""
    def __init__(self, radius=2, n_bits=1024):
        self.radius = radius
        self.n_bits= n_bits

    def transform(self, smiles_list):
        try:
            from rdkit.Chem import MolFromSmiles
            from rdkit.Chem import rdFingerprintGenerator
        except ModuleNotFoundError:
            raise ImportError("ECFPFeaturizer 需要 RDKit 环境。")
        
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fps = []
        for smi in smiles_list:
            mol = MolFromSmiles(smi)
            if mol:
                fps.append(np.array(fpgen.GetFingerprint(mol)))
            else:
                fps.append(np.zeros(self.n_bits, dtype=np.uint8))  # 无效SMILES用零向量填充
        return np.array(fps,dtype=np.uint8)