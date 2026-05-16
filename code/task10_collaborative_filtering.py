# -*- coding: utf-8 -*-
# 任务10：协同过滤推荐系统 —— 电影推荐
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys, os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SRC = r'E:\各科作业\机器学习\个人作业\发给学生的编程作业\发给学生的编程作业\第三部分：Unsupervised learning recommenders reinforcement learning\week2\Practice Lab 1'

# 加载数据（直接读CSV，因为recsys_utils.py里用的是相对路径）
def load_data():
    X = np.loadtxt(os.path.join(SRC, 'data', 'small_movies_X.csv'), delimiter=',')
    W = np.loadtxt(os.path.join(SRC, 'data', 'small_movies_W.csv'), delimiter=',')
    b = np.loadtxt(os.path.join(SRC, 'data', 'small_movies_b.csv'), delimiter=',').reshape(1, -1)
    num_movies, num_features = X.shape
    num_users = W.shape[0]
    return X, W, b, num_movies, num_features, num_users

def load_ratings():
    Y = np.loadtxt(os.path.join(SRC, 'data', 'small_movies_Y.csv'), delimiter=',')
    R = np.loadtxt(os.path.join(SRC, 'data', 'small_movies_R.csv'), delimiter=',')
    return Y, R

X, W_pre, b_pre, num_movies, num_features, num_users = load_data()
Y, R = load_ratings()
print(f"电影评分矩阵: Y={Y.shape}, R={R.shape}")
print(f"电影数: {num_movies}, 用户数: {num_users}, 特征数: {num_features}")
print(f"平均评分: {np.mean(Y[R==1]):.2f} / 5")
print(f"评分数量: {np.sum(R)}")

# ============================================================
# 1. 协同过滤代价函数（for-loop版本）
# ============================================================
def cofi_cost_func(X, W, b, Y, R, lambda_):
    nm, nu = Y.shape
    J = 0
    for j in range(nu):
        w = W[j, :]
        b_j = b[0, j]
        for i in range(nm):
            x = X[i, :]
            y = Y[i, j]
            r = R[i, j]
            J += r * np.square(np.dot(w, x) + b_j - y)
    J = J / 2
    # 正则化
    J += (lambda_ / 2) * (np.sum(np.square(W)) + np.sum(np.square(X)))
    return J

# 测试代价函数
X_r = X[:5, :3]
W_r = W_pre[:4, :3]
b_r = b_pre[0, :4].reshape(1, -1)
Y_r = Y[:5, :4]
R_r = R[:5, :4]
print(f"\n代价(λ=0): {cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0):.2f} (期望: 13.67)")
print(f"代价(λ=1.5): {cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5):.2f} (期望: 28.09)")

# ============================================================
# 2. TensorFlow向量化版本 + 自定义训练
# ============================================================
import tensorflow as tf
from tensorflow import keras

# 向量化代价函数
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

# 归一化
def normalizeRatings(Y, R):
    Ymean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
    Ynorm = Y - np.multiply(Ymean, R)
    return Ynorm, Ymean

# 添加虚拟用户评分
my_ratings = np.zeros(num_movies)
# 模拟评分（从课程预设中取几个）
for movie_id, rating in [(2700, 5), (2609, 2), (929, 5), (246, 5), (2716, 3),
                           (1150, 5), (382, 2), (366, 5), (793, 5)]:
    if movie_id < num_movies:
        my_ratings[movie_id] = rating

Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0).astype(int), R]
Ynorm, Ymean = normalizeRatings(Y, R)
num_movies, num_users = Y.shape
num_features = 100

print(f"\n添加新用户后: Y={Y.shape}, 评分数={np.sum(R)}")

# 初始化参数
tf.random.set_seed(1234)
W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
X_tf = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
b_tf = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

# 自定义训练循环
print("\n训练协同过滤模型...")
iterations = 200
lambda_ = 1
for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X_tf, W, b_tf, Ynorm, R, lambda_)
    grads = tape.gradient(cost_value, [X_tf, W, b_tf])
    optimizer.apply_gradients(zip(grads, [X_tf, W, b_tf]))
    if iter % 40 == 0:
        print(f"  迭代{iter:3d}: loss={cost_value:.1f}")

# ============================================================
# 3. 生成推荐
# ============================================================
p = np.matmul(X_tf.numpy(), np.transpose(W.numpy())) + b_tf.numpy()
pm = p + Ymean
my_predictions = pm[:, 0]
ix = tf.argsort(my_predictions, direction='DESCENDING').numpy()

# 加载电影列表
import pandas as pd
movie_df = pd.read_csv(os.path.join(SRC, 'data', 'small_movie_list.csv'), header=0, index_col=0)
movie_list = movie_df["title"].to_list()

my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

print("\n=== 对新用户的Top-10推荐 ===")
count = 0
for i in range(len(ix)):
    j = ix[i]
    if j not in my_rated and count < 10:
        print(f"  预测评分 {my_predictions[j]:.2f}: {movie_list[j]}")
        count += 1

print("\n=== 原始评分 vs 预测评分 ===")
for i in my_rated[:8]:
    print(f"  {movie_list[i][:40]} | 原始={my_ratings[i]:.0f}, 预测={my_predictions[i]:.2f}")

print("\n===== 任务10完成 =====")
