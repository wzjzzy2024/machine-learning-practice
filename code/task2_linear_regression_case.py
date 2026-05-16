# -*- coding: utf-8 -*-
# 任务2：线性回归案例分析 —— 加州房价预测
import matplotlib
matplotlib.use('Agg')
import os, tarfile, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据获取
# ============================================================
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# 尝试下载数据
try:
    fetch_housing_data()
    housing = load_housing_data()
    print("数据下载成功！")
except Exception as e:
    print(f"下载失败({e})，使用内置加州房价数据...")
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    housing = pd.DataFrame(data.data, columns=data.feature_names)
    housing['median_house_value'] = data.target
    housing['ocean_proximity'] = '<1H OCEAN'  # 补充分类特征

print(f"数据集形状: {housing.shape}")
print(housing.head())
print("\n数据信息:")
print(housing.info())
print("\n描述性统计:")
print(housing.describe())

# ============================================================
# 2. 数据可视化探索
# ============================================================
# 各数值特征的直方图
housing.hist(bins=50, figsize=(14, 10))
plt.tight_layout()
plt.savefig('task2_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("直方图已保存")

# ============================================================
# 3. 训练集/测试集划分
# ============================================================
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"\n训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}")

# ============================================================
# 4. 数据预处理：分离特征与标签
# ============================================================
housing_labels = train_set["median_house_value"].copy()
housing_train = train_set.drop("median_house_value", axis=1)

# 处理缺失值（如果有的话）
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# 分离数值特征
housing_num = housing_train.select_dtypes(include=[np.number])
imputer.fit(housing_num)
X_imputed = imputer.transform(housing_num)

# ============================================================
# 5. 特征工程：添加组合特征
# ============================================================
housing_num_df = pd.DataFrame(X_imputed, columns=housing_num.columns, index=housing_num.index)

# 如果有 rooms_per_household 等特征则添加组合特征
num_cols = list(housing_num_df.columns)
print(f"数值特征列: {num_cols}")

# 添加组合特征（基于可用列）
if 'AveRooms' in num_cols and 'AveBedrms' in num_cols:
    housing_num_df["rooms_per_household"] = housing_num_df["AveRooms"] / housing_num_df["AveBedrms"].replace(0, 1)
if 'Population' in num_cols and 'AveRooms' in num_cols:
    housing_num_df["population_per_household"] = housing_num_df["Population"] / housing_num_df["AveRooms"].replace(0, 1)

# ============================================================
# 6. 处理分类特征
# ============================================================
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 获取分类特征列
housing_cat = housing_train.select_dtypes(include=['object'])

# 构建预处理Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# 如果存在分类特征
cat_cols = list(housing_cat.columns)
num_cols_orig = list(housing_num.columns)

if cat_cols:
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols_orig),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])
else:
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols_orig),
    ])

housing_prepared = full_pipeline.fit_transform(housing_train)
print(f"预处理后特征维度: {housing_prepared.shape}")

# ============================================================
# 7. 线性回归模型训练
# ============================================================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 训练集预测与评估
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_r2 = r2_score(housing_labels, housing_predictions)

print(f"\n=== 线性回归模型评估 ===")
print(f"训练集 RMSE: ${lin_rmse:.2f}")
print(f"训练集 R2: {lin_r2:.4f}")

# ============================================================
# 8. 交叉验证
# ============================================================
from sklearn.model_selection import cross_val_score

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                              scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(f"\n=== 10折交叉验证 ===")
print(f"RMSE 均值: ${lin_rmse_scores.mean():.2f}")
print(f"RMSE 标准差: ${lin_rmse_scores.std():.2f}")

# ============================================================
# 9. 尝试其他模型并对比
# ============================================================
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 决策树
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
tree_predictions = tree_reg.predict(housing_prepared)
tree_rmse = np.sqrt(mean_squared_error(housing_labels, tree_predictions))
tree_r2 = r2_score(housing_labels, tree_predictions)

# 随机森林
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
forest_predictions = forest_reg.predict(housing_prepared)
forest_rmse = np.sqrt(mean_squared_error(housing_labels, forest_predictions))
forest_r2 = r2_score(housing_labels, forest_predictions)

print(f"\n=== 模型对比（训练集表现）===")
print(f"线性回归:     RMSE=${lin_rmse:,.2f},  R2={lin_r2:.4f}")
print(f"决策树:       RMSE=${tree_rmse:,.2f},  R2={tree_r2:.4f}")
print(f"随机森林:     RMSE=${forest_rmse:,.2f},  R2={forest_r2:.4f}")

# ============================================================
# 10. 交叉验证对比
# ============================================================
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print(f"\n=== 交叉验证对比 ===")
print(f"线性回归  CV RMSE: ${lin_rmse_scores.mean():,.2f} ± ${lin_rmse_scores.std():,.2f}")
print(f"随机森林  CV RMSE: ${forest_rmse_scores.mean():,.2f} ± ${forest_rmse_scores.std():,.2f}")

# ============================================================
# 11. 可视化：预测值 vs 真实值
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, pred, title in zip(axes,
    [housing_predictions, tree_predictions, forest_predictions],
    ['线性回归', '决策树', '随机森林']):
    ax.scatter(housing_labels, pred, alpha=0.3, s=2)
    ax.plot([housing_labels.min(), housing_labels.max()],
            [housing_labels.min(), housing_labels.max()], 'r--', lw=2)
    ax.set_xlabel('真实值'); ax.set_ylabel('预测值')
    ax.set_title(f'{title}\nRMSE=${np.sqrt(mean_squared_error(housing_labels, pred)):,.0f}')

plt.tight_layout()
plt.savefig('task2_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("预测对比图已保存为 task2_predictions.png")

# ============================================================
# 12. 特征重要性（随机森林）
# ============================================================
feature_names = list(housing_num_df.columns)
if hasattr(forest_reg, 'feature_importances_'):
    importances = forest_reg.feature_importances_
    # 只取数值特征的重要性（随机森林使用了所有特征）
    indices = np.argsort(importances[:len(feature_names)])[::-1][:10]

    plt.figure(figsize=(10, 6))
    plt.title("特征重要性 Top 10 (随机森林)")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('task2_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("特征重要性图已保存为 task2_feature_importance.png")

# ============================================================
# 13. 在测试集上最终评估
# ============================================================
test_labels = test_set["median_house_value"].copy()
test_data = test_set.drop("median_house_value", axis=1)
test_prepared = full_pipeline.transform(test_data)

final_predictions = forest_reg.predict(test_prepared)
final_rmse = np.sqrt(mean_squared_error(test_labels, final_predictions))
final_r2 = r2_score(test_labels, final_predictions)

print(f"\n=== 最终测试集评估（随机森林）===")
print(f"测试集 RMSE: ${final_rmse:,.2f}")
print(f"测试集 R2: {final_r2:.4f}")

print("\n===== 任务2完成 =====")
