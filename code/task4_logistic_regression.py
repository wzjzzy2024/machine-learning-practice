# -*- coding: utf-8 -*-
# 任务4：逻辑回归 —— 大学录取预测 & 微芯片质检
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import copy, math, os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r'E:\各科作业\机器学习\个人作业\发给学生的编程作业\发给学生的编程作业\第一大部分：Supervised Machine Learning Regression and Classification\week3\5.编程作业Week 3 practice lab logistic regression\data'

# ============================================================
# Part A: 逻辑回归基础 —— 大学录取预测
# ============================================================
def load_data(filename):
    data = np.loadtxt(os.path.join(DATA_DIR, filename), delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    return X, y

X_train, y_train = load_data("ex2data1.txt")
print(f"Part A: 录取预测数据 X={X_train.shape}, y={y_train.shape}")
print(f"前5个特征:\n{X_train[:5]}")
print(f"前5个标签: {y_train[:5]}")

# 1. Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(f"\nSigmoid测试: sigmoid(0)={sigmoid(0):.4f} (期望0.5)")

# 2. 代价函数
def compute_cost(X, y, w, b):
    m, n = X.shape
    total_cost = 0
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += w[j] * X[i, j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        loss = -y[i] * np.log(f_wb + 1e-10) - (1 - y[i]) * np.log(1 - f_wb + 1e-10)
        total_cost += loss
    return total_cost / m

initial_w = np.zeros(2)
initial_b = 0.0
print(f"初始代价: {compute_cost(X_train, y_train, initial_w, initial_b):.3f} (期望: 0.693)")

# 3. 梯度计算
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += w[j] * X[i, j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        err = f_wb - y[i]
        dj_db += err
        for j in range(n):
            dj_dw[j] += err * X[i, j]
    return dj_db / m, dj_dw / m

# 4. 梯度下降
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(compute_cost(X, y, w, b))
    return w, b, J_history

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

print("\n梯度下降运行中...")
w_final, b_final, J_history = gradient_descent(
    X_train, y_train, initial_w, initial_b, alpha=0.001, num_iters=10000)
print(f"最终参数: w={w_final}, b={b_final:.4f}")
print(f"最终代价: {J_history[-1]:.4f}")

# 5. 预测函数
def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += w[j] * X[i, j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb >= 0.5 else 0
    return p

p = predict(X_train, w_final, b_final)
accuracy = np.mean(p == y_train) * 100
print(f"训练集准确率: {accuracy:.2f}%")

# 6. 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 代价下降
axes[0].plot(J_history)
axes[0].set_title("代价函数下降曲线")
axes[0].set_xlabel("迭代次数"); axes[0].set_ylabel("代价")

# 决策边界
pos = y_train == 1; neg = y_train == 0
axes[1].scatter(X_train[pos, 0], X_train[pos, 1], c='b', marker='o', label='录取')
axes[1].scatter(X_train[neg, 0], X_train[neg, 1], c='r', marker='x', label='未录取')
x1_vals = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
x2_vals = -(w_final[0] * x1_vals + b_final) / w_final[1]
axes[1].plot(x1_vals, x2_vals, 'g-', lw=2, label='决策边界')
axes[1].set_xlabel("考试1成绩"); axes[1].set_ylabel("考试2成绩")
axes[1].set_title(f"逻辑回归决策边界\n准确率={accuracy:.1f}%")
axes[1].legend()

# Sigmoid可视化
z = np.linspace(-10, 10, 100)
axes[2].plot(z, sigmoid(z), 'b-', lw=2)
axes[2].axhline(y=0.5, color='r', linestyle='--')
axes[2].axvline(x=0, color='gray', linestyle='--')
axes[2].set_xlabel("z"); axes[2].set_ylabel("g(z)")
axes[2].set_title("Sigmoid函数")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task4_logistic_basic.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Part B: 正则化逻辑回归 —— 微芯片质检
# ============================================================
X_train2, y_train2 = load_data("ex2data2.txt")
print(f"\nPart B: 微芯片数据 X={X_train2.shape}, y={y_train2.shape}")

# 特征映射 (多项式特征到6次方)
def map_feature(X1, X2, degree=6):
    out = [np.ones(len(X1))]
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))
    return np.column_stack(out)

X_mapped = map_feature(X_train2[:, 0], X_train2[:, 1])
print(f"特征映射后: {X_mapped.shape[1]}维")

# 正则化逻辑回归
def compute_cost_reg(X, y, w, b, lambda_):
    m, n = X.shape
    cost = 0
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        cost += -y[i] * np.log(f_wb + 1e-10) - (1 - y[i]) * np.log(1 - f_wb + 1e-10)
    cost = cost / m + (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost

def compute_gradient_reg(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        err = f_wb - y[i]
        dj_db += err
        dj_dw += err * X[i]
    dj_dw = dj_dw / m + (lambda_ / m) * w
    dj_db = dj_db / m
    return dj_db, dj_dw

# 测试不同正则化参数
print("\n不同lambda正则化效果:")
for lambda_val in [0, 0.01, 1, 100]:
    np.random.seed(1)
    w_init = np.random.rand(X_mapped.shape[1]) - 0.5
    w = copy.deepcopy(w_init); b = 1.0
    for i in range(10000):
        dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train2, w, b, lambda_val)
        w = w - 0.01 * dj_dw
        b = b - 0.01 * dj_db

    p2 = predict(X_mapped, w, b)
    acc = np.mean(p2 == y_train2) * 100
    print(f"  lambda={lambda_val:6}: 准确率={acc:.2f}%")

# 最终可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, lambda_val in enumerate([0, 0.01, 1]):
    np.random.seed(1)
    w_init = np.random.rand(X_mapped.shape[1]) - 0.5
    w = copy.deepcopy(w_init); b = 1.0
    for i in range(10000):
        dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train2, w, b, lambda_val)
        w = w - 0.01 * dj_dw; b = b - 0.01 * dj_db

    pos = y_train2 == 1; neg = y_train2 == 0
    axes[idx].scatter(X_train2[pos, 0], X_train2[pos, 1], c='b', marker='o', s=20, label='通过')
    axes[idx].scatter(X_train2[neg, 0], X_train2[neg, 1], c='r', marker='x', s=20, label='未通过')
    axes[idx].set_xlabel("测试1"); axes[idx].set_ylabel("测试2")

    # 绘制决策边界网格
    u = np.linspace(-1, 1.5, 100)
    v = np.linspace(-1, 1.5, 100)
    zz = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            mapped = map_feature(np.array([u[i]]), np.array([v[j]]))
            z_val = np.dot(mapped[0], w) + b
            zz[i, j] = sigmoid(z_val)
    axes[idx].contour(u, v, zz.T, levels=[0.5], colors='g', linewidths=2)

    p_tmp = predict(X_mapped, w, b)
    acc = np.mean(p_tmp == y_train2) * 100
    axes[idx].set_title(f"lambda={lambda_val}\n准确率={acc:.1f}%")

plt.tight_layout()
plt.savefig('task4_regularized.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n===== 任务4完成 =====")
