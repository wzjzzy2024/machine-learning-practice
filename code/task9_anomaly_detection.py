# -*- coding: utf-8 -*-
# 任务9：异常检测 —— 服务器异常行为检测
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys, os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SRC = r'E:\各科作业\机器学习\个人作业\发给学生的编程作业\发给学生的编程作业\第三部分：Unsupervised learning recommenders reinforcement learning\week1\2 Practice Lab2'

# ============================================================
# 1. 加载数据
# ============================================================
X_train = np.load(os.path.join(SRC, 'data', 'X_part1.npy'))
X_val = np.load(os.path.join(SRC, 'data', 'X_val_part1.npy'))
y_val = np.load(os.path.join(SRC, 'data', 'y_val_part1.npy'))

print(f"2D数据: X_train={X_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")
print(f"X_train前5行:\n{X_train[:5]}")
print(f"y_val分布: 正常{(y_val==0).sum()}, 异常{(y_val==1).sum()}")

# ============================================================
# 2. 实现高斯参数估计
# ============================================================
def estimate_gaussian(X):
    m, n = X.shape
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)  # 注意：这里用总体方差(除以m)，不是样本方差(除以m-1)
    return mu, var

mu, var = estimate_gaussian(X_train)
print(f"\n特征均值: {mu}")
print(f"特征方差: {var}")
print(f"期望: mu=[14.11, 14.99], var=[1.83, 1.71]")

# ============================================================
# 3. 计算多元高斯概率密度
# ============================================================
def multivariate_gaussian(X, mu, var):
    k = len(mu)
    if var.ndim == 1:
        var = np.diag(var)
    X_centered = X - mu
    p = (2 * np.pi) ** (-k / 2) * np.linalg.det(var) ** (-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X_centered, np.linalg.pinv(var)) * X_centered, axis=1))
    return p

p_train = multivariate_gaussian(X_train, mu, var)
p_val = multivariate_gaussian(X_val, mu, var)
print(f"训练集概率范围: [{p_train.min():.2e}, {p_train.max():.2e}]")

# ============================================================
# 4. 选择最佳阈值 (基于F1分数)
# ============================================================
def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_F1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = (p_val < epsilon)
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        F1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1

epsilon, F1 = select_threshold(y_val, p_val)
print(f"\n最佳epsilon: {epsilon:.6e}")
print(f"最佳F1: {F1:.4f}")
print(f"期望: epsilon=8.99e-05, F1=0.875")

# 找出异常点
outliers = p_train < epsilon
print(f"训练集中检测到的异常点: {outliers.sum()}个")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 散点图
axes[0].scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b', alpha=0.5)
axes[0].set_xlabel('Latency (ms)'); axes[0].set_ylabel('Throughput (mb/s)')
axes[0].set_title('训练数据分布')
axes[0].set_xlim(0, 30); axes[0].set_ylim(0, 30)

# 高斯轮廓 + 异常点标记
X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
Z = Z.reshape(X1.shape)
axes[1].scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b', alpha=0.5)
if np.sum(np.isinf(Z)) == 0:
    axes[1].contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
axes[1].scatter(X_train[outliers, 0], X_train[outliers, 1], s=100,
                facecolors='none', edgecolors='r', linewidths=2, label=f'异常点({outliers.sum()}个)')
axes[1].set_xlabel('Latency (ms)'); axes[1].set_ylabel('Throughput (mb/s)')
axes[1].set_title(f'高斯分布拟合 (epsilon={epsilon:.2e})')
axes[1].legend()

# 概率直方图
axes[2].hist(p_train, bins=50, alpha=0.7, label='训练集')
axes[2].hist(p_val[y_val == 1], bins=20, alpha=0.7, color='r', label='验证集异常点')
axes[2].axvline(x=epsilon, color='g', linestyle='--', label=f'阈值={epsilon:.2e}')
axes[2].set_xlabel('概率 p(x)'); axes[2].set_ylabel('频数')
axes[2].set_title('概率分布与阈值')
axes[2].legend()

plt.tight_layout()
plt.savefig('task9_anomaly_2d.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. 高维数据集 (11个特征)
# ============================================================
X_train_h = np.load(os.path.join(SRC, 'data', 'X_part2.npy'))
X_val_h = np.load(os.path.join(SRC, 'data', 'X_val_part2.npy'))
y_val_h = np.load(os.path.join(SRC, 'data', 'y_val_part2.npy'))
print(f"\n高维数据: X_train={X_train_h.shape}, X_val={X_val_h.shape}")

mu_h, var_h = estimate_gaussian(X_train_h)
p_train_h = multivariate_gaussian(X_train_h, mu_h, var_h)
p_val_h = multivariate_gaussian(X_val_h, mu_h, var_h)
epsilon_h, F1_h = select_threshold(y_val_h, p_val_h)

print(f"高维最佳epsilon: {epsilon_h:.6e}")
print(f"高维最佳F1: {F1_h:.4f}")
print(f"检测到的异常点: {(p_train_h < epsilon_h).sum()}个")
print(f"期望: epsilon=1.38e-18, F1=0.615, anomalies=117")

print("\n===== 任务9完成 =====")
