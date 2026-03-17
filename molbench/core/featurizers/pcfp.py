from .base import BaseFeaturizer
import numpy as np

class PCFPFeaturizer(BaseFeaturizer):
    """PubChem Fingerprint(PCFP)指纹"""
    def __init__(self, n_bits=881):
        self.n_bits= n_bits

    def transform(self, smiles_list):
        try:
            from rdkit.Chem import MolFromSmiles,rdMolDescriptors
        except ModuleNotFoundError:
            raise ImportError("PCFPFeaturizer 需要 RDKit 环境。")
        
        fps = []
        for smi in smiles_list:
            mol = MolFromSmiles(smi)
            if mol:
                fpgen = rdMolDescriptors.GetPubChemFingerprint(mol)
                fps.append(np.array(fpgen))
            else:
                fps.append(np.zeros(self.n_bits, dtype=np.uint8))  # 无效SMILES用零向量填充
        return np.array(fps,dtype=np.uint8)