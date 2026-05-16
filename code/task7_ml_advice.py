# -*- coding: utf-8 -*-
# 任务7：机器学习实践建议 —— 偏差/方差分析、正则化调优
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Part A: 多项式回归 —— 偏差/方差分析
# ============================================================
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成数据
np.random.seed(42)
n_samples = 40
X_all = np.random.rand(n_samples, 1) * 6 - 0.5
y_all = np.sin(X_all).ravel() + np.random.randn(n_samples) * 0.3

# 三组划分
X_train, X_tmp, y_train, y_tmp = train_test_split(X_all, y_all, test_size=0.40, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=1)

print(f"Part A: 多项式回归偏差/方差分析")
print(f"训练集: {len(X_train)}, 交叉验证: {len(X_cv)}, 测试集: {len(X_test)}")

# 测试不同多项式阶数
max_degree = 10
err_train = np.zeros(max_degree)
err_cv = np.zeros(max_degree)

for degree in range(1, max_degree + 1):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_cv_pred = model.predict(X_cv)
    err_train[degree-1] = mean_squared_error(y_train, y_train_pred) / 2
    err_cv[degree-1] = mean_squared_error(y_cv, y_cv_pred) / 2

optimal_degree = np.argmin(err_cv) + 1

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 偏差/方差 vs 多项式阶数
axes[0, 0].plot(range(1, max_degree+1), err_train, 'b-o', label='训练误差')
axes[0, 0].plot(range(1, max_degree+1), err_cv, 'r-o', label='CV误差')
axes[0, 0].axvline(x=optimal_degree, color='g', linestyle='--', label=f'最优阶数={optimal_degree}')
axes[0, 0].set_xlabel("多项式阶数"); axes[0, 0].set_ylabel("MSE/2")
axes[0, 0].set_title("偏差/方差权衡")
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# 最佳模型拟合
best_model = Pipeline([
    ('poly', PolynomialFeatures(degree=optimal_degree)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])
best_model.fit(X_train, y_train)
X_plot = np.linspace(-0.5, 5.5, 200).reshape(-1, 1)
y_plot = best_model.predict(X_plot)

axes[0, 1].scatter(X_train, y_train, c='r', alpha=0.5, label='训练集')
axes[0, 1].scatter(X_cv, y_cv, c='orange', alpha=0.5, label='CV集')
axes[0, 1].plot(X_plot, y_plot, 'b-', lw=2, label=f'阶数={optimal_degree}拟合')
axes[0, 1].plot(X_plot, np.sin(X_plot).ravel(), 'g--', lw=1, label='真值 sin(x)')
axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
axes[0, 1].set_title(f"最优模型拟合 (degree={optimal_degree})")
axes[0, 1].legend()

# ============================================================
# Part B: 正则化调优
# ============================================================
degree_high = 10
lambdas = [0.0, 1e-5, 1e-3, 1e-1, 1, 10, 100]
err_train_reg = []; err_cv_reg = []

for lambda_ in lambdas:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree_high)),
        ('scaler', StandardScaler()),
        ('ridge', __import__('sklearn.linear_model').linear_model.Ridge(alpha=lambda_))
    ])
    model.fit(X_train, y_train)
    err_train_reg.append(mean_squared_error(y_train, model.predict(X_train)) / 2)
    err_cv_reg.append(mean_squared_error(y_cv, model.predict(X_cv)) / 2)

optimal_lambda = lambdas[np.argmin(err_cv_reg)]

axes[1, 0].semilogx(lambdas, err_train_reg, 'b-o', label='训练误差')
axes[1, 0].semilogx(lambdas, err_cv_reg, 'r-o', label='CV误差')
axes[1, 0].axvline(x=optimal_lambda, color='g', linestyle='--', label=f'最优lambda={optimal_lambda}')
axes[1, 0].set_xlabel("lambda (log)"); axes[1, 0].set_ylabel("MSE/2")
axes[1, 0].set_title(f"正则化调优 (degree={degree_high})")
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

# 不同lambda的拟合曲线
X_plot_2 = np.linspace(-0.5, 5.5, 200).reshape(-1, 1)
axes[1, 1].scatter(X_train, y_train, c='gray', alpha=0.3, s=10)
for lam, ls in [(0, 'r-'), (0.01, 'b-'), (10, 'g-')]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree_high)),
        ('scaler', StandardScaler()),
        ('ridge', __import__('sklearn.linear_model').linear_model.Ridge(alpha=lam))
    ])
    model.fit(X_train, y_train)
    axes[1, 1].plot(X_plot_2, model.predict(X_plot_2), ls, lw=1.5, label=f'lambda={lam}')
axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
axes[1, 1].set_title("不同正则化强度的拟合效果")
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task7_bias_variance.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"最优多项式阶数: {optimal_degree}")
print(f"最优正则化参数: {optimal_lambda}")
print(f"训练误差 (最优): {err_train[optimal_degree-1]:.4f}")
print(f"CV误差 (最优): {err_cv[optimal_degree-1]:.4f}")

# ============================================================
# Part C: 神经网络复杂度分析
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs

print(f"\nPart C: 神经网络复杂度分析")

# 生成6类数据
X_blob, y_blob = make_blobs(n_samples=500, centers=6, cluster_std=0.6, random_state=1)
Xb_train, Xb_tmp, yb_train, yb_tmp = train_test_split(X_blob, y_blob, test_size=0.50, random_state=1)
Xb_cv, Xb_test, yb_cv, yb_test = train_test_split(Xb_tmp, yb_tmp, test_size=0.20, random_state=1)
classes = 6

# 复杂模型
tf.random.set_seed(1234)
model_complex = Sequential([
    Dense(120, activation='relu', name='L1'),
    Dense(40, activation='relu', name='L2'),
    Dense(classes, activation='linear', name='L3')
], name="Complex")
model_complex.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01)
)
model_complex.fit(Xb_train, yb_train, epochs=500, verbose=0)

# 简单模型
tf.random.set_seed(1234)
model_simple = Sequential([
    Dense(6, activation='relu', name='L1'),
    Dense(classes, activation='linear', name='L2')
], name="Simple")
model_simple.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01)
)
model_simple.fit(Xb_train, yb_train, epochs=500, verbose=0)

# 正则化模型
tf.random.set_seed(1234)
model_reg = Sequential([
    Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name='L1'),
    Dense(40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name='L2'),
    Dense(classes, activation='linear', name='L3')
], name="Regularized")
model_reg.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01)
)
model_reg.fit(Xb_train, yb_train, epochs=500, verbose=0)

# 评估
def calc_accuracy(model, X, y):
    logits = model.predict(X, verbose=0)
    probs = tf.nn.softmax(logits).numpy()
    return np.mean(np.argmax(probs, axis=1) == y) * 100

models_data = [
    ("复杂模型\n(120+40神经元)", model_complex),
    ("简单模型\n(6神经元)", model_simple),
    ("正则化模型\n(L2=0.1)", model_reg)
]

print("\n模型对比:")
print(f"{'模型':<20} {'训练准确率':>10} {'CV准确率':>10}")
for name, m in models_data:
    train_acc = calc_accuracy(m, Xb_train, yb_train)
    cv_acc = calc_accuracy(m, Xb_cv, yb_cv)
    print(f"{name:<20} {train_acc:>9.2f}% {cv_acc:>9.2f}%")

# 可视化决策边界
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
xx, yy = np.meshgrid(np.linspace(X_blob[:,0].min()-1, X_blob[:,0].max()+1, 100),
                      np.linspace(X_blob[:,1].min()-1, X_blob[:,1].max()+1, 100))

for ax, (name, m) in zip(axes, models_data):
    grid = np.c_[xx.ravel(), yy.ravel()]
    logits = m.predict(grid, verbose=0)
    Z = np.argmax(tf.nn.softmax(logits).numpy(), axis=1).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
    ax.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap='tab10', s=10, alpha=0.6)
    train_acc = calc_accuracy(m, Xb_train, yb_train)
    cv_acc = calc_accuracy(m, Xb_cv, yb_cv)
    ax.set_title(f"{name}\n训练={train_acc:.1f}% CV={cv_acc:.1f}%", fontsize=10)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

plt.tight_layout()
plt.savefig('task7_nn_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 最终测试集评估
test_acc = calc_accuracy(model_reg, Xb_test, yb_test)
print(f"\n最优模型(正则化)测试集准确率: {test_acc:.2f}%")

print("\n===== 任务7完成 =====")
