# -*- coding: utf-8 -*-
# 任务8：K-Means聚类 —— 图像压缩应用
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SRC_DIR = r'E:\各科作业\机器学习\个人作业\发给学生的编程作业\发给学生的编程作业\第三部分：Unsupervised learning recommenders reinforcement learning\week1\1 Practice Lab1'

# ============================================================
# 1. 实现K-Means核心函数
# ============================================================
def find_closest_centroids(X, centroids):
    """将每个样本分配到最近的质心"""
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distances = [np.linalg.norm(X[i] - centroids[j]) for j in range(K)]
        idx[i] = np.argmin(distances)
    return idx

def compute_centroids(X, idx, K):
    """根据分配结果重新计算质心"""
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx == k]
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
    return centroids

def kMeans_init_centroids(X, K):
    """随机选择K个初始质心"""
    randidx = np.random.permutation(X.shape[0])
    return X[randidx[:K]]

def run_kMeans(X, initial_centroids, max_iters=10):
    """运行K-Means算法"""
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

# ============================================================
# 2. 在示例数据集上测试 (加载源目录数据)
# ============================================================
X = np.load(os.path.join(SRC_DIR, 'data', 'ex7_X.npy'))
print(f"示例数据集: X.shape = {X.shape}")

# 运行K-Means
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
centroids, idx = run_kMeans(X, initial_centroids, max_iters=10)
print(f"最终质心:\n{centroids}")

# 可视化聚类结果
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 原始数据
axes[0].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6)
axes[0].set_title("原始数据"); axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")

# 聚类结果
colors = ['r', 'g', 'b']
for k in range(K):
    axes[1].scatter(X[idx == k, 0], X[idx == k, 1], c=colors[k], alpha=0.6, label=f'簇{k}')
axes[1].scatter(centroids[:, 0], centroids[:, 1], marker='X', c='black', s=200, label='质心')
axes[1].set_title(f"K-Means聚类 (K={K})"); axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2")
axes[1].legend()

plt.tight_layout()
plt.savefig('task8_kmeans_demo.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 3. 图像压缩应用
# ============================================================
# 加载bird_small.png
bird_img = plt.imread(os.path.join(SRC_DIR, 'bird_small.png'))
print(f"\n原始图像: shape = {bird_img.shape}, dtype = {bird_img.dtype}")

# 归一化并重塑
bird_img = bird_img / 255.0
X_img = bird_img.reshape(-1, 3)
print(f"像素数据: X_img.shape = {X_img.shape}")

# 运行K-Means压缩到16种颜色
K_img = 16
np.random.seed(42)
init_centroids = kMeans_init_centroids(X_img, K_img)
centroids_img, idx_img = run_kMeans(X_img, init_centroids, max_iters=10)
print(f"压缩完成，{K_img}种颜色代表{len(X_img)}个像素")

# 重建压缩图像
X_recovered = centroids_img[idx_img]
X_recovered = X_recovered.reshape(bird_img.shape)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(bird_img)
axes[0].set_title("原始图像"); axes[0].axis('off')
axes[1].imshow(X_recovered)
axes[1].set_title(f"压缩后 ({K_img}种颜色)"); axes[1].axis('off')
plt.tight_layout()
plt.savefig('task8_image_compression.png', dpi=150, bbox_inches='tight')
plt.close()

# 压缩比计算
original_bits = 128 * 128 * 24
compressed_bits = K_img * 24 + 128 * 128 * 4
print(f"\n压缩前: {original_bits:,} bits")
print(f"压缩后: {compressed_bits:,} bits")
print(f"压缩比: {original_bits / compressed_bits:.1f}x")

# ============================================================
# 4. 不同K值对比
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for idx_k, k_val in enumerate([2, 4, 8, 16, 32, 64]):
    ax = axes[idx_k // 3, idx_k % 3]
    np.random.seed(42)
    init_c = kMeans_init_centroids(X_img, k_val)
    c, idxx = run_kMeans(X_img, init_c, max_iters=5)
    recovered = c[idxx].reshape(bird_img.shape)
    ax.imshow(recovered)
    ax.set_title(f"K={k_val}"); ax.axis('off')

plt.suptitle("不同K值的压缩效果", fontsize=14)
plt.tight_layout()
plt.savefig('task8_k_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n===== 任务8完成 =====")
