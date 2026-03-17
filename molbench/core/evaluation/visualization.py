import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from molbench.core.evaluation import get_score
from typing import Dict, List, Tuple, Optional

class ModelComparisonVisualizer:
    """
    多模型对比可视化
    
    支持：
    - 雷达图：多指标综合对比
    - 柱状图：指标对比
    - 散点图矩阵：模型间相关性
    - 残差分析：误差分布对比
    - 训练曲线：loss/ metric 随 epoch 变化
    """
    
    def __init__(self, task_type: str = 'regression', save_dir: str = './figures'):
        self.task_type = task_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def radar_chart(self, results_dict: Dict[str, Dict], metrics: List[str] = None,
                    title: str = "Model Comparison", filename: str = 'radar_chart.png'):
        """
        雷达图：多维度模型对比
        
        Args:
            results_dict: {model_name: {metric: value}}
            metrics: 要显示的指标列表，None则自动选择
        """
        if metrics is None:
            # 自动选择共有的数值指标
            all_metrics = set()
            for metrics in results_dict.values():
                all_metrics.update(k for k, v in metrics.items() 
                                 if isinstance(v, (int, float)))
            metrics = sorted(all_metrics)[:6]  # 最多6个维度
        
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for model_name, model_results in results_dict.items():
            values = []
            for metric in metrics:
                val = model_results.get(metric, 0)
                # 归一化到0-1（假设指标都是越大越好）
                all_vals = [r.get(metric, 0) for r in results_dict.values()]
                min_val, max_val = min(all_vals), max(all_vals)
                if max_val > min_val:
                    norm_val = (val - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
                values.append(norm_val)
            
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 雷达图已保存: {save_path}")
        plt.close()
    
    def bar_chart_comparison(self, results_dict: Dict[str, Dict], 
                           metrics: List[str] = None,
                           title: str = "Model Performance Comparison",
                           filename: str = 'bar_comparison.png'):
        """
        柱状图对比
        
        支持：分组柱状图展示多个指标
        """
        if metrics is None:
            # 自动选择
            metrics = ['val_r2', 'test_r2', 'val_mae', 'test_mae']
            if self.task_type != 'regression':
                metrics = ['val_auc', 'test_auc', 'val_acc', 'test_acc']
        
        # 过滤存在的指标
        available_metrics = []
        for m in metrics:
            if any(m in r for r in results_dict.values()):
                available_metrics.append(m)
        
        if not available_metrics:
            print("⚠️ 没有可用的指标用于绘图")
            return
        
        # 准备数据
        models = list(results_dict.keys())
        x = np.arange(len(models))
        width = 0.8 / len(available_metrics)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(available_metrics):
            values = [results_dict[m].get(metric, 0) for m in models]
            offset = (i - len(available_metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 柱状图已保存: {save_path}")
        plt.close()
    
    def scatter_comparison_matrix(self, predictions_dict: Dict[str, np.ndarray],
                                 y_true: np.ndarray,
                                 title: str = "Model Predictions Correlation",
                                 filename: str = 'scatter_matrix.png'):
        """
        散点图矩阵：展示模型间的预测相关性
        
        对角线：每个模型 vs 真实值
        非对角线：模型 i vs 模型 j
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(n_models, n_models, figsize=(3*n_models, 3*n_models))
        if n_models == 1:
            axes = np.array([[axes]])
        
        for i, model_i in enumerate(model_names):
            for j, model_j in enumerate(model_names):
                ax = axes[i, j]
                
                if i == j:
                    # 对角线：vs 真实值
                    pred = predictions_dict[model_i]
                    ax.scatter(y_true, pred, alpha=0.5, s=20)
                    ax.plot([y_true.min(), y_true.max()], 
                           [y_true.min(), y_true.max()], 'k--', lw=1)
                    r2 = r2_score(y_true, pred)
                    ax.set_title(f'{model_i}\nR²={r2:.3f}', fontsize=10)
                    ax.set_xlabel('True')
                    ax.set_ylabel('Pred')
                else:
                    # 非对角线：模型间对比
                    pred_i = predictions_dict[model_i]
                    pred_j = predictions_dict[model_j]
                    ax.scatter(pred_j, pred_i, alpha=0.5, s=20)
                    corr = np.corrcoef(pred_i, pred_j)[0, 1]
                    ax.set_title(f'{model_i} vs {model_j}\nr={corr:.3f}', fontsize=9)
                    ax.set_xlabel(model_j)
                    ax.set_ylabel(model_i)
                
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 散点图矩阵已保存: {save_path}")
        plt.close()
    
    def residual_analysis(self, predictions_dict: Dict[str, np.ndarray],
                        y_true: np.ndarray,
                        filename: str = 'residual_analysis.png'):
        """
        残差分析：误差分布对比
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for idx, model_name in enumerate(model_names):
            y_pred = predictions_dict[model_name]
            residuals = y_true - y_pred
            
            # 上排：残差 vs 预测值
            ax1 = axes[0, idx]
            ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Residual')
            ax1.set_title(f'{model_name}\nResidual vs Fitted')
            ax1.grid(True, alpha=0.3)
            
            # 下排：残差分布直方图
            ax2 = axes[1, idx]
            ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--')
            ax2.set_xlabel('Residual')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Mean={residuals.mean():.3f}, Std={residuals.std():.3f}')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 残差分析已保存: {save_path}")
        plt.close()
    
    def training_curves(self, histories: Dict[str, List[float]],
                       metric_name: str = 'loss',
                       filename: str = 'training_curves.png'):
        """
        训练曲线对比
        
        Args:
            histories: {model_name: [metric_epoch_1, metric_epoch_2, ...]}
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, history in histories.items():
            epochs = range(1, len(history) + 1)
            ax.plot(epochs, history, marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'Training {metric_name.capitalize()} Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练曲线已保存: {save_path}")
        plt.close()

class visualizer(ModelComparisonVisualizer):
    def __init__(self, task_type='regression', save_dir = './figures', 
                 selector=None, data_map = None):
        super().__init__(task_type=task_type, save_dir=save_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.selector = selector
        self.data_map = data_map

        if task_type not in {'regression', 'binary','multiclass'}:
            raise ValueError(
                "task_type must be 'regression', 'binary' or 'multiclass'")
        self.task_type = task_type
        self._discrete_models = ['DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier', 
                      'GaussianProcessClassifier', 'QuadraticDiscriminantAnalysis']

    # 使用共享 scoring helper: get_score(model, X) -> (scores, score_source)

    def _plot_binary_val(self, best_models, X_val, y_val, file_name='', 
                         col_idx=0, data_map= None): 
        plt.figure(figsize=(10,8))
        best_auc, best_name = -1, None
        data_map = self.data_map if data_map is None else data_map

        for name, model in best_models.items():
            if data_map and name in data_map:
                model_data = data_map[name]
                X_input= model_data['X']
                y_input = model_data.get('y_val', y_val)  # 优先使用缓存的标签数据
                sel = model_data.get('selector')
                is_graph = model_data.get('is_graph', False)
            else:
                X_input = X_val
                y_input = y_val
                sel = self.selector
                is_graph = False
            
            if y_input is None:
                raise ValueError(f"{name}: y_val 为 None")

            if (not is_graph and sel is not None and hasattr(X_val, 'ndim') and 
                X_input.ndim == 2 and X_input.shape[1] > 200):  # 1024 > 200，需要 transform
                X_input = sel.transform(X_input)
           
            # 长度检查
            if len(X_input) != len(y_input):
                raise ValueError(f"{name}: 长度不匹配 X={len(X_input)}, y={len(y_input)}")
            # 类别检查
            if len(set(y_input)) < 2:
                raise ValueError(f"{name}: y_val 只有 {len(set(y_input))} 个类别")
            
            y_score, score_src = get_score(model, X_input, model_name=name)
            fpr, tpr, _ = roc_curve(y_input, y_score)
            roc_auc = auc(fpr, tpr)
            
            if any(discrete in name for discrete in self._discrete_models) and len(fpr)>2:
                from scipy import interpolate
                f_smooth = interpolate.interp1d(fpr, tpr, kind='linear')
                fpr = np.linspace(0, 1, 100)
                tpr = f_smooth(fpr)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})') 
            if roc_auc > best_auc:
                best_auc, best_name = roc_auc, name

        # ROC曲线
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Validation-set ROC Curves")
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.05))
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.save_dir / f'roc_val_{file_name}_{col_idx}.png'
        plt.savefig(save_path)
        plt.show()
        return best_name, best_auc

    def _plot_binary_test(self, model, X_test, y_test, model_name, file_name='', col_idx=0, data_map=None):
        data_map = self.data_map if data_map is None else data_map

        if data_map and model_name in data_map:
            # 优先使用缓存的测试集数据
            model_data = data_map[model_name]
            X_input = model_data.get('X_test', X_test)  
            y_input = model_data.get('y_test', y_test)
            is_graph = model_data.get('is_graph', False)
        else:
            X_input = X_test
            y_input = y_test
            is_graph = False
        
        if y_input is None:
            raise ValueError(f"{model_name}: y_test 为 None")

        if (not is_graph and self.selector is not None and hasattr(X_input, 'ndim') and 
            X_input.ndim == 2 and X_input.shape[1] > 200):  # 1024 > 200，需要 transform
                X_input = self.selector.transform(X_input)
        
        if len(X_input) != len(y_input):
            raise ValueError(f"{model_name}: 测试集长度不匹配 X={len(X_input)}, y={len(y_input)}")  
        
        y_score, score_src = get_score(model, X_input, model_name=model_name)
        fpr, tpr, _ = roc_curve(y_input, y_score)
        roc_auc = auc(fpr, tpr)
        print(f"roc_auc: {roc_auc:.4f}")
        
        # 平滑处理
        if any(discrete in model_name for discrete in self._discrete_models) and len(fpr) > 2:
            from scipy import interpolate
            f_smooth = interpolate.interp1d(fpr, tpr, kind='linear')
            fpr = np.linspace(0, 1, 100)
            tpr = f_smooth(fpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Test-set ROC - best model")
        plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.save_dir / f'roc_test_{file_name}_{col_idx}.png'
        plt.savefig(save_path)
        plt.show()
        return roc_auc

    def _plot_binaryclass(self, best_models, y_test, X): 
        plt.figure(figsize=(12,10))
        roc_data ={}
        # 处理离散型概率模型: 使用插值让曲线更平滑（仅用于可视化）
        discrete_models = ['DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier', 
                      'GaussianProcessClassifier', 'QuadraticDiscriminantAnalysis']

        for name, model in best_models.items():
            y_score, score_src = get_score(model, X, model_name=name)
            is_discrete = any(discrete in name for discrete in discrete_models)
            if is_discrete:
                fpr, tpr, thresholds = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                from scipy import interpolate
                if len(fpr) > 2:
                    f_smooth = interpolate.interp1d(fpr, tpr, kind='linear')
                    fpr_smooth = np.linspace(0, 1, 100)
                    tpr_smooth = f_smooth(fpr_smooth)
                    roc_data[name] = {'fpr': fpr_smooth, 'tpr': tpr_smooth, 'auc': roc_auc}
                else:
                    roc_data[name] = {'fpr':fpr, 'tpr': tpr, 'auc':roc_auc}
            else:  
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                roc_data[name] = {'fpr': fpr,'tpr': tpr,'auc': roc_auc}

            if len(best_models)==1:
                print(f"ROC AUC: {roc_auc:.4f}")
        
        for name, data in roc_data.items():
            plt.plot(data['fpr'],data['tpr'],label=f'{name} (AUC = {data["auc"]:.3f})') 
            # plt.fill_between(data['fpr'],data['tpr']-0.05,data['tpr']+0.05,alpha=0.1)
        # precision, recall, _ = precision_recall_curve(y_test, y_prob)
        # avg_precision = average_precision_score(y_test, y_prob)

        # ROC曲线
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.05))
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.save_dir / f'roc_figure.png'
        plt.savefig(save_path)
        plt.show()

    def _plot_multiclass(self, class_num, y_true, y_pred, y_prob):
        # 混淆矩阵(以三分类为例)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        save_path = self.save_dir / f'roc_figure.png'
        plt.savefig(save_path) # 待修改：class_num>3时的处理方式、多元分类的其他评估指标
        plt.show()
    
    def _plot_regression(self, y_true, y_pred, file_name='', col_idx=0):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print(f"r2: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        # 数据范围
        y_min = min(np.min(y_true), np.min(y_pred))
        y_max = max(np.max(y_true), np.max(y_pred))
        margin = (y_max - y_min) * 0.1

        fig, ax = plt.subplots() #绘制
        ax.scatter(y_true,y_pred) #绘制预测-真值对散点
        ax.plot([y_min-margin, y_max+margin],[y_min-margin, y_max+margin],"--k") 
        ax.set(xlabel="True Target",
                ylabel="Target predicted",
                title="Test model", 
                xlim=(y_min-margin, y_max+margin), ylim=(y_min-margin, y_max+margin))
        #计算决定系数R2,MAE以及RMSE作为图注
        ax.text(
            y_min, 
            y_max, 
            r"$R^2$=%.2f, MAE=%.2f, RMSE=%.2f"
            %(r2, mae, mse),
            fontsize=10,
            va='top',  # 垂直对齐方式
            ha='left'  # 水平对齐方式
        ) 
        fig.tight_layout()
        save_path = self.save_dir / f'scatter_figure_{file_name}_{col_idx}.png'
        plt.savefig(save_path)
        plt.show()

    def plot(self, mode='auto', *args, **kwargs):
        """
        根据 task_type 自动调用对应绘图函数
        用法：
        - 分类: vis.plot(fpr, tpr, roc_auc, precision, recall, avg_precision)
        - 回归: vis.plot(y_true, y_pred)
        """
        data_map = kwargs.pop('data_map', self.data_map)
        if self.task_type == 'binary':
            if mode == 'val':
                return self._plot_binary_val(*args, data_map=data_map,**kwargs)
            elif mode == 'test':
                return self._plot_binary_test(*args, **kwargs)
            else:
                return self._plot_binaryclass(*args, **kwargs)
        elif self.task_type == 'multiclass':
            self._plot_multiclass(*args, **kwargs)
        elif self.task_type == 'regression':
            self._plot_regression(*args, **kwargs)
    
    def compare_models(self, results_dict: Dict, predictions_dict: Dict,
                      y_true: np.ndarray, y_val: np.ndarray = None,
                      prefix: str = ''):
        """一键生成所有对比图表"""

        print("\n生成模型对比图表...")
        
        # 1. 雷达图
        try:
            self.radar_chart(results_dict, filename=f'{prefix}radar.png')
        except Exception as e:
            print(f"  雷达图生成失败: {e}")
        
        # 2. 柱状图
        try:
            self.bar_chart_comparison(results_dict, filename=f'{prefix}bar.png')
        except Exception as e:
            print(f"  柱状图生成失败: {e}")
        
        # 3. 散点图矩阵
        try:
            self.scatter_comparison_matrix(predictions_dict, y_true, 
                                        filename=f'{prefix}scatter_matrix.png')
        except Exception as e:
            print(f"  散点图矩阵生成失败: {e}")
        
        # 4. 残差分析（仅回归）
        if self.task_type == 'regression':
            try:
                self.residual_analysis(predictions_dict, y_true,
                                    filename=f'{prefix}residuals.png')
            except Exception as e:
                print(f"  残差分析生成失败: {e}")
        
        print(f"✓ 所有图表已保存到: {self.save_dir}")
