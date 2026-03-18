import warnings
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import precision_recall_curve,auc
from sklearn.calibration import CalibratedClassifierCV

def get_score(model, X, model_name=None):
    """Get a 1-D score array and a source string from a classifier model.

    Returns: (scores, score_source)
    - scores: 1-D numpy array usable by roc_curve/average_precision_score
    - score_source: one of 'probability', 'decision_function', 'internal_proba', 'hard_label'

    Priority:
    1. model.predict_proba(X) -> use column 1 for binary
    2. model.decision_function(X)
    3. model._predict_proba_lr(X)  (sklearn linear classifiers)
    4. fallback model.predict(X) (hard labels) with a UserWarning
    """
    # try predict_proba
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X)
        except Exception as e:
            print(f"[get_score] ❌ predict_proba 失败: {e}")  # 打印错误信息！
            import traceback
            traceback.print_exc()
            probs = None
        else:
            probs = np.asarray(probs)
            if probs.ndim == 1:
                return probs.ravel(), 'probability'
            if probs.shape[1] == 2:
                return probs[:, 1], 'probability'
            # multiclass: default to column 1
            return probs[:, 1], 'probability'

    # try decision_function
    if hasattr(model, 'decision_function'):
        try:
            scores = model.decision_function(X)
            scores = np.asarray(scores)
            if scores.ndim > 1 and scores.shape[1] == 2:
                return scores[:, 1], 'decision_function'
            return scores.ravel(), 'decision_function'
        except Exception:
            pass

    # try sklearn internal _predict_proba_lr
    if hasattr(model, '_predict_proba_lr'):
        try:
            probs = model._predict_proba_lr(X)
            probs = np.asarray(probs)
            if probs.ndim == 1:
                return probs.ravel(), 'internal_proba'
            if probs.shape[1] == 2:
                return probs[:, 1], 'internal_proba'
            return probs[:, 1], 'internal_proba'
        except Exception:
            pass

    # fallback to hard predict
    try:
        preds = model.predict(X)
        warnings.warn(
            f"⚠️ {model_name or model.__class__.__name__} 不支持 predict_proba/decision_function，使用硬标签 predict 作为得分替代（退化）。",
            UserWarning
        )
        return np.asarray(preds).ravel(), 'hard_label'
    except Exception:
        raise RuntimeError("无法从模型获取预测得分（既没有 predict_proba/decision_function，也无法 predict）")

def get_calibrated_models(best_models):
    calibrated_models = {}
    for name, model in best_models.items():
        if any(discrete in name for discrete in 
               ['DecisionTree', 'GaussianNB', 'KNeighbors',
                'GaussianProcessClassifier', 'QuadraticDiscriminantAnalysis']):
            print(f'对{name}进行概率校准…')
            calibrated_models[name] = CalibratedClassifierCV(model, method= 'isotonic', cv=3)
        else:
            calibrated_models[name] = model
    return calibrated_models

from sklearn.metrics import (
        mean_absolute_percentage_error, median_absolute_error,
        recall_score, f1_score, matthews_corrcoef
    )
from scipy.stats import pearsonr
EXTRA_REG_METRICS = {
    'Pearson_r': lambda y, y_p: pearsonr(y, y_p)[0],
    'MedAE': lambda y, y_p: median_absolute_error(y, y_p),
    'MAPE': lambda y, y_p: mean_absolute_percentage_error(y, y_p),
    '95%CI_Coverage': lambda y, y_p: np.mean(np.abs(y - y_p) <= np.percentile(np.abs(y - y_p), 95)),
}
EXTRA_CLF_METRICS = {
    'Recall': lambda y, y_p: recall_score(y, y_p, zero_division=0, 
                                          average='binary' if len(np.unique(y))==2 else 'macro'),
    'F1_macro': lambda y, y_p: f1_score(y, y_p, zero_division=0, average='macro'),
    'MCC': lambda y, y_p: matthews_corrcoef(y, y_p),
}
# 序号映射
IDX_TO_METRIC = {
    '1': 'Pearson_r', '2': 'MedAE', '3': 'MAPE', '4': '95%CI_Coverage',
    '5': 'Recall', '6': 'F1_macro', '7': 'MCC', 
    '0': 'ALL',                     # 一键全选
    'h': 'HELP',
}
def select_extra_metrics(task_type='regression'):
    """
    交互式点选附加指标 -> 返回用户想要的指标名列表
    输入 h 显示菜单；输入 0 全选；输入 q 退出；支持重复勾选。
    """
    pool = EXTRA_REG_METRICS if task_type == 'regression' else EXTRA_CLF_METRICS
    avail = {**pool, 'HELP': None, 'ALL': None}   # 合法键
    selected = set()

    def _print_menu():
        print("\n===== 可选附加评估指标 =====")
        for idx, name in IDX_TO_METRIC.items():
            print(f"  {idx}: {name}")
        print("  q: 完成选择并退出")

    _print_menu()
    while True:
        raw = input("请输入想加载的指标序号/名称（可多选，空格分隔；按 q 完成）: ").strip()
        if raw.lower() == 'q':
            break
        for token in raw.split():
            if token in IDX_TO_METRIC:        # 序号
                name = IDX_TO_METRIC[token]
            elif token in avail:              # 直接名称
                name = token
            else:
                print(f"不支持的指标 '{token}'，请重新选择。")
                continue
            if name == 'HELP':
                _print_menu()
            elif name == 'ALL':
                selected.update(pool.keys())
            else:
                selected.add(name)
        print(f"已选: {list(selected)}")

    return list(selected)   # 返回用户想要的指标名列表

class evaluation:
    def __init__(self,task_type='regression', extra_metrics=None):
        if task_type not in {'regression', 'binary','multiclass'}:
            raise ValueError(
                "task_type must be 'regression', 'binary' or 'multiclass'")
        self.task_type = task_type
        self.extra_metrics = extra_metrics

    def plot(self, best_model, X, y_true, dataset_type='val'):
        """
        计算评估指标
        
        参数:
        - best_model: 训练好的模型
        - X: 特征数据
        - y_true: 真实标签
        - dataset_type: 数据集类型 ('val' 或 'test')，用于键名区分
        
        返回:
        - metrics: 字典，包含评估指标
        """
        if self.task_type == 'regression':
            y_pred = best_model.predict(X)
            metrics = {
                f'{dataset_type}_r2': r2_score(y_true, y_pred),
                f'{dataset_type}_mae': mean_absolute_error(y_true, y_pred),
                f'{dataset_type}_rmse': root_mean_squared_error(y_true, y_pred)
            }
            for name in self.extra_metrics:
                if name in EXTRA_REG_METRICS:
                    func = EXTRA_REG_METRICS[name]
                    metrics[f'{dataset_type}_{name}'] = func(y_true, y_pred)
            return metrics
        else: 
            y_pred = best_model.predict(X)

            y_score, score_src = get_score(best_model, X, model_name=best_model.__class__.__name__)

            if score_src == 'decision_function':
                print(f"ℹ️  {best_model.__class__.__name__} 使用 decision_function 作为置信度，已用于 ROC-AUC / AP 计算。")

            metrics = {
                f'{dataset_type}_roc_auc': roc_auc_score(y_true, y_score),
                f'{dataset_type}_precision': precision_score(y_true, y_pred),
                f'{dataset_type}_average_precision': average_precision_score(y_true, y_score),
                'score_source': score_src
            }
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)  # 梯形积分
            metrics[f'{dataset_type}_pr_auc'] = pr_auc
            for name in self.extra_metrics:
                if name in EXTRA_CLF_METRICS:
                    func = EXTRA_CLF_METRICS[name]
                    metrics[f'{dataset_type}_{name}'] = func(y_true, y_pred)
            return metrics
        