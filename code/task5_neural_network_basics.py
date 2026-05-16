# -*- coding: utf-8 -*-
# 任务5：神经网络基础 —— 手写数字识别 (0 vs 1)
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys, os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SRC = r'E:\各科作业\机器学习\个人作业\发给学生的编程作业\发给学生的编程作业\第二部分：Advanced Learning Algorithms\week1\6.编程作业Practice Lab Neural networks'
sys.path.insert(0, SRC)

# ============================================================
# 1. 加载数据
# ============================================================
X = np.load(os.path.join(SRC, 'data', 'X.npy'))[:1000]
y = np.load(os.path.join(SRC, 'data', 'y.npy'))[:1000]
print(f"数据集: X.shape={X.shape}, y.shape={y.shape}")
print(f"标签分布: 0有{np.sum(y==0)}个, 1有{np.sum(y==1)}个")

# 可视化部分样本
fig, axes = plt.subplots(5, 8, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(len(X))
    ax.imshow(X[idx].reshape(20, 20).T, cmap='gray')
    ax.set_title(f"y={int(y[idx])}", fontsize=8)
    ax.axis('off')
plt.suptitle("手写数字样本 (0 vs 1)", fontsize=14)
plt.tight_layout()
plt.savefig('task5_samples.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 2. 用TensorFlow构建神经网络模型
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    tf.keras.Input(shape=(400,)),
    Dense(25, activation='sigmoid', name='L1'),
    Dense(15, activation='sigmoid', name='L2'),
    Dense(1, activation='sigmoid', name='L3')
], name="DigitRecognizer")

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.summary()

# ============================================================
# 3. 训练模型
# ============================================================
print("\n训练中...")
history = model.fit(X, y, epochs=20, verbose=0)
print(f"最终损失: {history.history['loss'][-1]:.4f}")

# 损失曲线
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], 'b-')
plt.title("训练损失曲线"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.savefig('task5_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 4. 模型预测与评估
# ============================================================
# 测试几个样本
print("\n预测示例:")
for i in [0, 500]:
    pred = model.predict(X[i].reshape(1, 400), verbose=0)[0][0]
    label = "1" if pred >= 0.5 else "0"
    print(f"  样本{i}: 真实={int(y[i])}, 预测={label}, 概率={pred:.4f}")

# 全部预测准确率
predictions = model.predict(X, verbose=0)
y_pred = (predictions >= 0.5).astype(int).flatten()
accuracy = np.mean(y_pred == y.flatten()) * 100
print(f"\n训练集准确率: {accuracy:.2f}%")

# ============================================================
# 5. NumPy手动前向传播验证
# ============================================================
def sigmoid_np(z):
    return 1 / (1 + np.exp(-z))

def my_dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        z = np.dot(W[:, j], a_in) + b[j]
        a_out[j] = sigmoid_np(z)
    return a_out

# 获取训练好的权重
W1, b1 = model.layers[0].get_weights()
W2, b2 = model.layers[1].get_weights()
W3, b3 = model.layers[2].get_weights()

# NumPy前向传播
a1 = my_dense(X[0], W1, b1)
a2 = my_dense(a1, W2, b2)
a3 = my_dense(a2, W3, b3)
print(f"\nNumPy手动预测结果: {a3[0]:.4f} (阈值判定: {'1' if a3[0]>=0.5 else '0'})")
print(f"TensorFlow预测结果: {predictions[0][0]:.4f}")

# 可视化预测vs真实
fig, axes = plt.subplots(5, 8, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(len(X))
    ax.imshow(X[idx].reshape(20, 20).T, cmap='gray')
    true_label = int(y[idx])
    pred_label = int(y_pred[idx])
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f"T:{true_label},P:{pred_label}", color=color, fontsize=8)
    ax.axis('off')
plt.suptitle("预测结果 (T=真实, P=预测, 红色=错误)", fontsize=14)
plt.tight_layout()
plt.savefig('task5_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n===== 任务5完成 =====")
