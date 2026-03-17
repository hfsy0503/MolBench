from molbench.core.adapters import BenchModel
import numpy as np

class SklearnModel(BenchModel):
    """将 scikit-learn 模型包装为 MolBench 接口"""
    def __init__(self, base_cls, task_type: str, **params):
        self.base_cls = base_cls
        self.task_type = task_type
        self.params = params
        self.estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.estimator = self.base_cls(**self.params)
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if hasattr(self.estimator, 'predict_proba'):
            return self.estimator.predict_proba(X)
        return None

    def get_params(self, deep=True) -> dict:
        if self.estimator is not None:
            return self.estimator.get_params(deep=deep)
        return self.params.copy()
    
    def get_task_type(self) -> str:
        return self.task_type
    
    def save(self, path: str) -> None:
        import joblib
        joblib.dump({'estimator': self.estimator, 'params': self.params, 
                     'task_type': self.task_type, 'base_cls': self.base_cls}, path)
        
    def load(self, path: str) -> "SklearnModel":
        import joblib
        data = joblib.load(path)
        adapter = SklearnModel(data['base_cls'], data['task_type'], **data['params'])
        adapter.estimator = data['estimator']
        return adapter
    
