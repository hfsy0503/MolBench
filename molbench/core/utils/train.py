import numpy as np

def predict(X_val, Model):
    predictions = Model.predict(X_val)
    # y_prob = model.predict_proba(X_val_scaled)[:, 1]
    # print("预测结果:", predictions)
    return predictions

def select_task_type():
    task_type = None
    while task_type is None:
        selection = input(
            "Select task type:\nA. regression  B. binary classification  C. multiclass classification \n")
        if selection.lower() in ['a', 'regression']:
            task_type = 'regression'
        elif selection.lower() in ['b', 'binary classification']:
            task_type = 'binary'
        elif selection.lower() in ['c', 'multiclass classification']:
            task_type = 'multiclass'
        else:
            print('Invalid input. Please select again.')
    print(f"Selected task type: {task_type}")
    return task_type

def auto_detect_task_type(y_values):
    """根据标签数据自动判断任务类型"""
    unique_vals = np.unique(y_values)
    n_unique = len(unique_vals)

    def is_sequential(vals):
        if len(vals) < 3:
            return False
        sorted_vals = np.sort(vals)
        diffs = np.diff(sorted_vals)
        return np.all(diffs == 1) and len(diffs) > 0
    
    # 字符串标签
    if y_values.dtype == object or isinstance(y_values.flat[0], (str, bytes)):
        if n_unique == 2:
            return 'binary'
        elif 3<= n_unique <= 50:
            return 'multiclass'
        else:
            raise ValueError(f"无法自动判断任务类型：字符串标签但类别数为 {n_unique}，请手动指定。")
    
    # 数值标签
    if np.issubdtype(y_values.dtype, np.floating):
        if n_unique == 2 and np.allclose(sorted(unique_vals), [0.0, 1.0]):
            return 'binary'
        else:
            return 'regression'
    
    if np.issubdtype(y_values.dtype, np.integer):
        if n_unique == 2:
            return 'binary'
        elif 3 <= n_unique <= 50:
            if is_sequential(unique_vals):
                return 'regression'  # 连续整数且类别数适中，可能是回归问题
            return 'multiclass'
        else:
            return 'regression'  # 整数但类别数很多，可能是回归问题
        
    if y_values.dtype == bool:
        return 'binary'
    
    return 'regression'  # 默认回归