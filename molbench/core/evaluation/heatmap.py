import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# # 1. 准备数据（R2）
# data = {
#     'ESOL': [0.892, 0.888, 0.876, 0.732],
#     'FreeSolv': [0.804, 0.901, 0.899, 0.761],
#     'Lipophilicity': [0.705, 0.603, 0.662, 0.533],
#     'QM7': [0.773, 0.609, 0.553, 0.703],
#     'QM8': [0.743, 0.731, 0.777, 0.653],
#     'QM9': [0.886, 0.836, 0.884, 0.794]
# }
# df = pd.DataFrame(data, index=['ChemBERTa-2', 'GAT', 'GIN', 'XGBoost'])

# # 2. 绘制热力图
# plt.figure(figsize=(12, 6))
# sns.heatmap(df, 
#             annot=True,           # 显示数值
#             fmt='.3f',             # 数值格式（保留3位小数）
#             cmap='YlGnBu',       # 颜色映射
#             linewidths=0.5,        # 单元格间隔线
#             linecolor='white',
#             vmin=0.5, vmax=1.0,
#             cbar_kws={'label': 'R² (↑ better)'})  # 颜色条标签

# plt.title('Regression Task Performance Heatmap (R²)', fontsize=14)
# plt.xlabel('Dataset', fontsize=12)
# plt.ylabel('Model', fontsize=12)
# plt.tight_layout()
# plt.savefig('regression_heatmap.png', dpi=300, bbox_inches='tight')
# plt.show()

# 分类数据（ROC-AUC，越大越好）
# 不平衡数据集数据（PRC-AUC）
data_imbalanced = {
    'HIV': [0.295, 0.349, 0.395, 0.342, 0.384],
    'MUV': [0.029, 0.022, 0.076, 0.060, 0.042],
    'ClinTox': [0.768, 0.747, 0.657, 0.62, 0.937],
    'Tox21': [0.425, 0.389, 0.402, 0.414, 0.483],
    'ToxCast': [0.565, 0.552, 0.616, 0.585, 0.639],
    'SIDER': [0.602, 0.601, 0.633, 0.626, 0.594]
}
df_imbalanced = pd.DataFrame(data_imbalanced,
                            index=['GAT', 'GCN', 'RF', 'XGBoost','ChemBERTa-2'])

plt.figure(figsize=(12, 6))
sns.heatmap(df_imbalanced, 
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',        # 同样用黄绿蓝，越大越好
            linewidths=0.5,
            linecolor='white',
            vmin=0, vmax=1.0,      # PRC-AUC范围0-1
            cbar_kws={'label': 'PRC-AUC (↑ better)'})

plt.title('Imbalanced Datasets: PRC-AUC Performance', fontsize=14)
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig('imbalanced_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# 平衡数据集数据（ROC-AUC）
# data_balanced = {
#     'BBBP': [0.900, 0.906, 0.890, 0.906, 0.930],
#     'BACE': [0.865, 0.830, 0.941, 0.963, 0.938],
#     'HIV': [0.823, 0.813, 0.784, 0.760, 0.807],
#     'ClinTox': [0.885, 0.876, 0.733, 0.776, 0.957],
#     'Tox21': [0.821, 0.812, 0.779, 0.797, 0.822],
#     'ToxCast': [0.862, 0.881, 0.845, 0.841, 0.887],
#     'SIDER': [0.645, 0.635, 0.669, 0.652, 0.589]
# }
# df_balanced = pd.DataFrame(data_balanced, 
#                           index=['GAT', 'GCN', 'RF', 'XGBoost', 'ChemBERTa-2'])

# plt.figure(figsize=(12, 6))
# sns.heatmap(df_balanced, 
#             annot=True,
#             fmt='.3f',
#             cmap='YlGnBu',        # 越大越好，深色代表好
#             linewidths=0.5,
#             linecolor='white',
#             vmin=0.5, vmax=1.0,    # ROC-AUC范围
#             cbar_kws={'label': 'ROC-AUC (↑ better)'})

# plt.title('Balanced Datasets: ROC-AUC Performance', fontsize=14)
# plt.xlabel('Dataset', fontsize=12)
# plt.ylabel('Model', fontsize=12)
# plt.tight_layout()
# plt.savefig('balanced_heatmap.png', dpi=300, bbox_inches='tight')
# plt.show()

# 准备数据（用你的实际数据替换）
# datasets = ['ESOL', 'FreeSolv', 'Lipophilicity', 'QM7', 'QM8', 'QM9',
#             'BBBP', 'BACE', 'Tox21', 'HIV', 'MUV', 'ClinTox', 'ToxCast', 'SIDER']
# time_traditional = [21.7, 13.6, 0.5, 724.0, 404.5, 1859.4, 
#                     11.7, 43.4, 26.0, 177.5, 147.4, 1.7, 66.0, 16.4]
# time_gnn = [294.7, 81.7, 595.3, 1177.8, 3464.5, 6363.8, 
#             317.3, 271.7, 1036.8, 6267.5, 2791.0, 271.3, 661.1, 217.2]
# time_transformer = [173.9, 67.7, 1166.1, 66.2, 2618.9, 790.9,
#                     602.3, 24.7, 60.3, 377.1, 103.4, 414.4, 79.5, 399.6]

# # 创建DataFrame（模型类别为行，数据集为列）
# df = pd.DataFrame([time_traditional, time_gnn, time_transformer],
#                   columns=datasets,
#                   index=['Traditional', 'GNN', 'Transformer'])
# df = df.T
# plt.figure(figsize=(6, 10))
# sns.heatmap(df,
#             annot=True,
#             fmt='.1f',
#             cmap='OrRd',
#             linewidths=0.5,
#             cbar_kws={'label': 'Training Time (seconds)'})
# plt.xticks(rotation=0)
# plt.title('Training Time of Best Models by Category', fontsize=14)
# plt.xlabel('Model Category')
# plt.ylabel('Dataset')
# plt.tight_layout()
# plt.savefig('efficiency_heatmap_final.png', dpi=300, bbox_inches='tight')
# plt.show()