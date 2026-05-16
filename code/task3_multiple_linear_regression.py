# -*- coding: utf-8 -*-
# 任务3：多元线性回归 —— 餐厅利润预测
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import copy, math, os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据源路径
DATA_DIR = r'E:\各科作业\机器学习\个人作业\发给学生的编程作业\发给学生的编程作业\第一大部分：Supervised Machine Learning Regression and Classification\week2\3.编程作业Week 2 practice lab Linear regression\data'

# ============================================================
# 1. 加载数据
# ============================================================
def load_data():
    data = np.loadtxt(os.path.join(DATA_DIR, 'ex1data1.txt'), delimiter=',')
    X = data[:, 0]  # 城市人口 (x 10,000)
    y = data[:, 1]  # 餐厅利润 (x $10,000)
    return X, y

x_train, y_train = load_data()
print(f"x_train前5个: {x_train[:5]}")
print(f"y_train前5个: {y_train[:5]}")
print(f"数据维度: x_train={x_train.shape}, y_train={y_train.shape}")
print(f"训练样本数 m = {len(x_train)}")

# ============================================================
# 2. 数据可视化
# ============================================================
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, marker='x', c='r', s=50)
plt.title("餐厅利润 vs 城市人口")
plt.xlabel("城市人口 (x 10,000)")
plt.ylabel("利润 (x $10,000)")
plt.grid(True, alpha=0.3)
plt.savefig('task3_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 3. 实现代价函数 J(w,b)
# ============================================================
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

# 测试
initial_w, initial_b = 2, 1
cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(f"\n初始参数 w=2, b=1 时的代价: {cost:.3f} (期望: 75.203)")

# ============================================================
# 4. 实现梯度计算
# ============================================================
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw, dj_db = 0, 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# 测试
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, 0, 0)
print(f"初始梯度 (w=0,b=0): dj_dw={tmp_dj_dw:.4f}, dj_db={tmp_dj_db:.4f}")
print(f"期望: dj_dw=-65.3288, dj_db=-5.8391")

# ============================================================
# 5. 梯度下降算法
# ============================================================
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(compute_cost(x, y, w, b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"  迭代 {i:4d}: 代价 = {J_history[-1]:.2f}")
    return w, b, J_history

print("\n梯度下降运行中...")
w_final, b_final, J_history = gradient_descent(
    x_train, y_train, 0.0, 0.0, alpha=0.01, num_iters=1500)
print(f"\n最终参数: w = {w_final:.4f}, b = {b_final:.4f}")
print(f"期望: w=1.1664, b=-3.6303")

# ============================================================
# 6. 可视化：代价下降曲线
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(range(len(J_history)), J_history)
axes[0].set_title("代价函数下降曲线")
axes[0].set_xlabel("迭代次数"); axes[0].set_ylabel("代价 J(w,b)")
axes[0].grid(True, alpha=0.3)

# 线性拟合图
axes[1].scatter(x_train, y_train, marker='x', c='r', s=30, label='训练数据')
predicted = w_final * x_train + b_final
axes[1].plot(x_train, predicted, c='b', linewidth=2, label='线性拟合')
axes[1].set_title(f"线性拟合: f(x)={w_final:.2f}x+{b_final:.2f}")
axes[1].set_xlabel("城市人口 (x10,000)"); axes[1].set_ylabel("利润 (x$10,000)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 预测vs真实值
axes[2].scatter(y_train, predicted, alpha=0.5, s=30)
axes[2].plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', lw=2)
axes[2].set_xlabel("真实利润"); axes[2].set_ylabel("预测利润")
axes[2].set_title("预测 vs 真实值")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task3_results.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. 预测新数据
# ============================================================
print("\n=== 预测结果 ===")
for pop in [3.5, 7.0]:
    profit = pop * w_final + b_final
    print(f"人口 = {pop*10000:.0f}人, 预测利润 = ${profit*10000:.2f}")

# ============================================================
# 8. 多元线性回归（扩展）
# ============================================================
print("\n=== 多元线性回归扩展 ===")
data2 = np.loadtxt(os.path.join(DATA_DIR, 'ex1data2.txt'), delimiter=',')
X_multi = data2[:, :2]
y_multi = data2[:, 2]
print(f"多变量数据: X={X_multi.shape}, y={y_multi.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_multi)

# sklearn线性回归
model = LinearRegression()
model.fit(X_scaled, y_multi)
y_pred = model.predict(X_scaled)

rmse = np.sqrt(mean_squared_error(y_multi, y_pred))
r2 = r2_score(y_multi, y_pred)
cv_scores = cross_val_score(model, X_scaled, y_multi, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print(f"sklearn线性回归 RMSE: {rmse:.4f}")
print(f"sklearn线性回归 R2: {r2:.4f}")
print(f"5折CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
print(f"系数: {model.coef_}, 截距: {model.intercept_:.4f}")

# 可视化多元回归
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X_multi[:, 0], y_multi, alpha=0.5)
axes[0].set_xlabel("特征1 (房屋面积)"); axes[0].set_ylabel("价格")
axes[0].set_title("特征1 vs 目标")
axes[1].scatter(X_multi[:, 1], y_multi, alpha=0.5, c='g')
axes[1].set_xlabel("特征2 (卧室数量)"); axes[1].set_ylabel("价格")
axes[1].set_title("特征2 vs 目标")
plt.tight_layout()
plt.savefig('task3_multi_features.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n===== 任务3完成 =====")
