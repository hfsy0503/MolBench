from molbench.core.adapters import BenchModel   
import joblib
import numpy as np

# 编写你自己的模型适配器，步骤如下：
class CustomModel(BenchModel):          # 1. 改类名
    """用户自建模型"""
    def __init__(self, lr=0.01, n_estimators=100):   # 2. 写入超参
        self.params = dict(lr=lr, n_estimators=n_estimators)
        # ④ 把你原来的模型实例化
        from sklearn.ensemble import AdaBoostRegressor
        self.model = AdaBoostRegressor(learning_rate=lr, n_estimators=n_estimators)

    def fit(self, X, y):              # 3. 训练
        self.model.fit(X, y)
        return self

    def predict(self, X):             # 4. 预测
        return self.model.predict(X)

    def get_params(self, deep=True):  # 5. 返回超参
        return self.params.copy()

    def get_task_type(self):          # 6. 任务类型
        return 'regression'   # 或 'binary' | 'multiclass'


# 7. 一键注册
if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
    from model_register import register_model
    register_model(CustomModel, task_type='regression',
                   save_dir=r'molbench\core\hyper_parameters\test',
                   protocol='bench') # 可自行改路径