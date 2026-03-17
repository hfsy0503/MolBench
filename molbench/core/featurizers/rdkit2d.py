from .base import BaseFeaturizer
import numpy as np

class RDKit2DFeaturizer(BaseFeaturizer):
    def __init__(self):
        self.desc_names = [
            'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
            'NumValenceElectrons', 'NumRadicalElectrons',
            'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
            'MinAbsEStateIndex', 'NumHDonors', 'NumHAcceptors',
            'NumRotatableBonds', 'NumAliphaticRings', 'NumAromaticRings',
            'NumSaturatedRings', 'NumAliphaticHeterocycles',
            'NumAromaticHeterocycles', 'NumSaturatedHeterocycles',
            'NumAliphaticCarbocycles', 'NumAromaticCarbocycles',
            'NumSaturatedCarbocycles', 'RingCount', 'FractionCSP3',
            'HeavyAtomCount', 'NHOHCount', 'NOCount',
            'MaxPartialCharge', 'MinPartialCharge',
            'MaxAbsPartialCharge', 'MinAbsPartialCharge',
            'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
            'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v',
            'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
            'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
            'HallKierAlpha', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5',
            'Kappa1', 'Kappa2', 'Kappa3',
            'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
            'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10',
            'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
            'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
            'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10',
            'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
            'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8',
            'SlogP_VSA9', 'SlogP_VSA10', 'TPSA', 'EState_VSA1',
            'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
            'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
            'EState_VSA10', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3',
            'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
            'VSA_EState8', 'VSA_EState9', 'VSA_EState10'
        ]

    def transform(self, smiles_list):
        """RDKit 2D 描述符组"""
        try:
            from rdkit.Chem import MolFromSmiles
            from rdkit.Chem import Descriptors
        except ModuleNotFoundError:
            raise ImportError("RDKit 2D 需要 RDKit 环境。")
        
        calc = [getattr(Descriptors, name) for name in self.desc_names]
        n_feat = len(self.desc_names)
        feats = []
        for smi in smiles_list:
            mol = MolFromSmiles(smi)
            if mol is not None:
                feats.append(func(mol) for func in calc)
            else:
                feats.append([0.0]* n_feat)  # 无效SMILES填充
        return np.array(feats, dtype=np.float32)