# -*- coding: utf-8 -*-
# 任务11：基于内容的神经网络推荐系统
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys, os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SRC = r'E:\各科作业\机器学习\个人作业\发给学生的编程作业\发给学生的编程作业\第三部分：Unsupervised learning recommenders reinforcement learning\week2\Practice Lab 2'

# ============================================================
# 1. 加载数据
# ============================================================
from numpy import genfromtxt
from collections import defaultdict
import csv, pickle

item_train = genfromtxt(os.path.join(SRC, 'data', 'content_item_train.csv'), delimiter=',')
user_train = genfromtxt(os.path.join(SRC, 'data', 'content_user_train.csv'), delimiter=',')
y_train = genfromtxt(os.path.join(SRC, 'data', 'content_y_train.csv'), delimiter=',')
item_vecs = genfromtxt(os.path.join(SRC, 'data', 'content_item_vecs.csv'), delimiter=',')

with open(os.path.join(SRC, 'data', 'content_item_train_header.txt')) as f:
    item_features = list(csv.reader(f))[0]
with open(os.path.join(SRC, 'data', 'content_user_train_header.txt')) as f:
    user_features = list(csv.reader(f))[0]

movie_dict = defaultdict(dict)
with open(os.path.join(SRC, 'data', 'content_movie_list.csv')) as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)  # skip header
    for line in reader:
        movie_dict[int(line[0])] = {"title": line[1], "genres": line[2]}

with open(os.path.join(SRC, 'data', 'content_user_to_genre.pickle'), 'rb') as f:
    user_to_genre = pickle.load(f)

num_user_features = user_train.shape[1] - 3
num_item_features = item_train.shape[1] - 1
print(f"训练样本数: {len(item_train)}")
print(f"用户特征数: {num_user_features}, 电影特征数: {num_item_features}")
print(f"电影数: {len(item_vecs)}, 电影字典条目: {len(movie_dict)}")

# ============================================================
# 2. 数据预处理
# ============================================================
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# 标准化
scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train_s = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train_s = scalerUser.transform(user_train)

# 划分训练/测试集
item_train_s, item_test = train_test_split(item_train_s, train_size=0.80, shuffle=True, random_state=1)
user_train_s, user_test = train_test_split(user_train_s, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test = train_test_split(y_train, train_size=0.80, shuffle=True, random_state=1)

# 目标值缩放到[-1, 1]
scaler = MinMaxScaler((-1, 1))
scaler.fit(y_train.reshape(-1, 1))
ynorm_train = scaler.transform(y_train.reshape(-1, 1))
ynorm_test = scaler.transform(y_test.reshape(-1, 1))
print(f"训练集: {item_train_s.shape[0]}条, 测试集: {item_test.shape[0]}条")

# ============================================================
# 3. 构建双塔神经网络模型
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dot, Lambda

u_s = 3  # user特征起始列
i_s = 1  # item特征起始列
num_outputs = 32

tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_outputs)
])

item_NN = tf.keras.models.Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_outputs)
])

# 用户和物品输入
input_user = Input(shape=(num_user_features,))
vu = user_NN(input_user)
vu = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)

input_item = Input(shape=(num_item_features,))
vm = item_NN(input_item)
vm = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)

# 点积输出
output = Dot(axes=1)([vu, vm])
model = Model([input_user, input_item], output)

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.01)
)
model.summary()

# ============================================================
# 4. 训练模型
# ============================================================
print("\n训练中...")
tf.random.set_seed(1)
history = model.fit(
    [user_train_s[:, u_s:], item_train_s[:, i_s:]],
    ynorm_train,
    epochs=30,
    verbose=0
)

test_loss = model.evaluate(
    [user_test[:, u_s:], item_test[:, i_s:]],
    ynorm_test,
    verbose=0
)
print(f"训练损失: {history.history['loss'][-1]:.4f}")
print(f"测试损失: {test_loss:.4f}")

# ============================================================
# 5. 实现平方距离函数 (找相似电影)
# ============================================================
def sq_dist(a, b):
    return np.sum(np.square(a - b))

# 测试
a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
print(f"\n平方距离测试: d(a1,b1)={sq_dist(a1,b1):.4f}, d(a2,b2)={sq_dist(a2,b2):.4f}")

# ============================================================
# 6. 生成电影特征向量并找相似电影
# ============================================================
input_item_m = Input(shape=(num_item_features,))
vm_m = item_NN(input_item_m)
vm_m = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm_m)
model_m = Model(input_item_m, vm_m)

scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:, i_s:], verbose=0)
print(f"电影特征向量: {vms.shape} (694部电影 × 32维)")

# 计算距离矩阵（取前50个电影做示例）
n_sample = min(50, len(vms))
dist = np.zeros((n_sample, n_sample))
for i in range(n_sample):
    for j in range(n_sample):
        dist[i, j] = sq_dist(vms[i, :], vms[j, :])

# 找每部电影最相似的电影（跳过自身）
print("\n=== 相似电影示例（前5个）===")
for i in range(5):
    dist_i = dist[i].copy()
    dist_i[i] = np.inf  # 排除自身
    min_idx = np.argmin(dist_i)
    movie1_id = int(item_vecs[i, 0])
    movie2_id = int(item_vecs[min_idx, 0])
    genre1 = movie_dict[movie1_id]['genres'] if movie1_id in movie_dict else '?'
    genre2 = movie_dict[movie2_id]['genres'] if movie2_id in movie_dict else '?'
    name1 = movie_dict[movie1_id]['title'] if movie1_id in movie_dict else f'ID{movie1_id}'
    name2 = movie_dict[movie2_id]['title'] if movie2_id in movie_dict else f'ID{movie2_id}'
    print(f"  {name1[:30]} ({genre1[:25]})")
    print(f"    → {name2[:30]} ({genre2[:25]}) [距离={dist_i[min_idx]:.4f}]")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
axes[0].plot(history.history['loss'], 'b-')
axes[0].set_title("训练损失"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
axes[0].grid(True, alpha=0.3)

# 相似度矩阵（前20个电影的距离热力图）
import matplotlib
im = axes[1].imshow(dist[:20, :20], cmap='hot', aspect='auto')
axes[1].set_title("电影间平方距离矩阵 (前20部)")
axes[1].set_xlabel("电影j"); axes[1].set_ylabel("电影i")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('task11_recsys_nn.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n===== 任务11完成 =====")
