from .base import BaseFeaturizer
import numpy as np

class Mol2VecFeaturizer(BaseFeaturizer):
    """Mol2Vec 300 维预训练向量"""
    def __init__(self, n_bits=300):
        self.n_bits= n_bits

    def transform(self, smiles_list):
        try:
            from deepchem.feat import Mol2VecFingerprint
        except ModuleNotFoundError:
            raise ImportError("Mol2Vec 需要 deepchem 环境。")
        
        featurizer = Mol2VecFingerprint()
        return featurizer.featurize(smiles_list).astype(np.float32)