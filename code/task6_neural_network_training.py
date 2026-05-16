# -*- coding: utf-8 -*-
# 任务6：神经网络训练 —— 手写数字多分类 (0-9)
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os, sys

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SRC = r'E:\各科作业\机器学习\个人作业\发给学生的编程作业\发给学生的编程作业\第二部分：Advanced Learning Algorithms\week2\5.编程作业Practice Lab Neural network training'

# ============================================================
# 1. 加载数据与Softmax实现
# ============================================================
X = np.load(os.path.join(SRC, 'data', 'X.npy'))
y = np.load(os.path.join(SRC, 'data', 'y.npy'))
print(f"数据集: X.shape={X.shape}, y.shape={y.shape}")
print(f"标签分布: {np.bincount(y.flatten())}")

# Softmax函数
def my_softmax(z):
    ez = np.exp(z - np.max(z))  # 减去最大值防止数值溢出
    return ez / np.sum(ez)

# 测试
z_test = np.array([1., 2., 3., 4.])
print(f"Softmax测试: {my_softmax(z_test)}")

# 可视化样本
fig, axes = plt.subplots(5, 8, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(len(X))
    ax.imshow(X[idx].reshape(20, 20).T, cmap='gray')
    ax.set_title(f"{int(y[idx])}", fontsize=8)
    ax.axis('off')
plt.suptitle("手写数字样本 (0-9)", fontsize=14)
plt.tight_layout()
plt.savefig('task6_samples.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 2. TensorFlow神经网络模型
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(1234)
model = Sequential([
    tf.keras.Input(shape=(400,)),
    Dense(25, activation='relu', name='L1'),
    Dense(15, activation='relu', name='L2'),
    Dense(10, activation='linear', name='L3')
], name="DigitClassifier")

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
model.summary()

# ============================================================
# 3. 训练模型
# ============================================================
print("\n训练中...")
history = model.fit(X, y, epochs=40, verbose=0)
print(f"最终损失: {history.history['loss'][-1]:.4f}")

# ============================================================
# 4. 评估
# ============================================================
logits = model.predict(X, verbose=0)
probs = tf.nn.softmax(logits).numpy()
y_pred = np.argmax(probs, axis=1)
y_true = y.flatten()
accuracy = np.mean(y_pred == y_true) * 100
errors = np.sum(y_pred != y_true)

print(f"训练集准确率: {accuracy:.2f}%")
print(f"错误样本数: {errors}/{len(X)}")

# 预测示例
for sample_idx in [0, 1015, 2000]:
    pred = model.predict(X[sample_idx].reshape(1, 400), verbose=0)
    prob = tf.nn.softmax(pred).numpy()[0]
    predicted_digit = np.argmax(prob)
    print(f"  样本{sample_idx}: 真实={int(y[sample_idx])}, 预测={predicted_digit}, "
          f"概率={prob[predicted_digit]:.4f}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 损失曲线
axes[0, 0].plot(history.history['loss'])
axes[0, 0].set_title("训练损失曲线")
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(True, alpha=0.3)

# 预测结果展示
np.random.seed(42)
rand_idx = np.random.choice(len(X), 40, replace=False)
for i, (idx, ax_i) in enumerate(zip(rand_idx, range(40))):
    row, col = divmod(ax_i, 8)
    # 简单展示前40个随机样本
    if i < 40:
        pass

# 更多可视化 - 混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
axes[0, 1].imshow(cm, cmap='Blues')
for i in range(10):
    for j in range(10):
        axes[0, 1].text(j, i, cm[i, j], ha='center', va='center', fontsize=7)
axes[0, 1].set_xlabel("预测"); axes[0, 1].set_ylabel("真实")
axes[0, 1].set_title("混淆矩阵")
axes[0, 1].set_xticks(range(10)); axes[0, 1].set_yticks(range(10))

# 各类准确率
class_acc = np.diag(cm) / np.sum(cm, axis=1)
axes[1, 0].bar(range(10), class_acc * 100)
axes[1, 0].set_xlabel("数字"); axes[1, 0].set_ylabel("准确率 (%)")
axes[1, 0].set_title("各类别准确率")
axes[1, 0].set_ylim(80, 105)
for i, acc in enumerate(class_acc):
    axes[1, 0].text(i, acc * 100 + 0.5, f"{acc*100:.1f}%", ha='center', fontsize=8)

# 预测示例（含错误）
sample_idx_list = list(range(1000, 1064))
display_pred = []
for idx in sample_idx_list:
    logit = model.predict(X[idx].reshape(1, 400), verbose=0)
    prob = tf.nn.softmax(logit).numpy()[0]
    p_class = np.argmax(prob)
    display_pred.append((idx, int(y[idx]), p_class, prob[p_class]))

for i, (row, col) in enumerate([(i//8, i%8) for i in range(64)]):
    if i < len(display_pred):
        idx, true_l, pred_l, prob = display_pred[i]
        color = 'green' if true_l == pred_l else 'red'
        axes[1, 1].text(col/8+0.06, 1-(row/8+0.05), f"{true_l}->{pred_l}",
                        color=color, fontsize=7, transform=axes[1,1].transAxes)
axes[1, 1].set_title("预测示例 (真实->预测)\n绿色正确,红色错误", fontsize=12)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('task6_results.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n===== 任务6完成 =====")
