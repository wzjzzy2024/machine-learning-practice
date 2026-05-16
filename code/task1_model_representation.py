# -*- coding: utf-8 -*-
# 任务1：模型表示 (Model Representation) —— 一元线性回归模型
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互模式，不弹窗
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 准备训练数据：房价预测（面积单位：1000平方英尺，价格单位：1000美元）
# ============================================================
x_train = np.array([1.0, 2.0])           # 房屋面积
y_train = np.array([300.0, 500.0])       # 房屋价格
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# ============================================================
# 2. 获取训练样本数量 m
# ============================================================
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"训练样本数量 m = {m}")

# 也可用 len() 获取
m = len(x_train)
print(f"使用 len(x_train) 获取 m = {m}")

# 访问单个样本
i = 0
print(f"第{i}个样本: x^({i}) = {x_train[i]}, y^({i}) = {y_train[i]}")

# ============================================================
# 3. 数据可视化：绘制散点图
# ============================================================
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, marker='x', c='r', s=100, label='训练数据')
plt.title("房价预测训练数据")
plt.xlabel("房屋面积 (1000 sqft)")
plt.ylabel("价格 (1000 dollars)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('task1_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("散点图已保存为 task1_scatter.png")

# ============================================================
# 4. 定义线性模型：f_{w,b}(x) = w * x + b
# ============================================================
def compute_model_output(x, w, b):
    """
    计算线性模型的预测输出
    Args:
        x: 输入特征 (ndarray, shape (m,))
        w: 权重参数
        b: 偏置参数
    Returns:
        f_wb: 模型预测值 (ndarray, shape (m,))
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

# ============================================================
# 5. 设定模型参数并绘制预测直线
# ============================================================
# 手动选择参数 w=200, b=100，使得直线穿过两个训练点
w = 200
b = 100

# 计算预测值
tmp_f_wb = compute_model_output(x_train, w, b)

# 绘制模型预测直线与训练数据
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, marker='x', c='r', s=100, label='训练数据')
plt.plot(x_train, tmp_f_wb, c='b', linewidth=2, label=f'模型预测: f(x)={w}x+{b}')
plt.title("线性回归模型拟合")
plt.xlabel("房屋面积 (1000 sqft)")
plt.ylabel("价格 (1000 dollars)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('task1_model_fit.png', dpi=150, bbox_inches='tight')
plt.close()
print("模型拟合图已保存为 task1_model_fit.png")

# ============================================================
# 6. 使用模型进行预测
# ============================================================
# 预测1200平方英尺（即x=1.2）的房价
x_new = 1.2
y_pred = w * x_new + b
print(f"\n预测1200 sqft房屋的价格: ${y_pred:.2f}K (即${y_pred*1000:,.0f})")

# 预测多个新数据点
x_test = np.array([1.2, 1.5, 2.5, 3.0])
y_test = compute_model_output(x_test, w, b)
print("\n多组预测结果:")
for i in range(len(x_test)):
    print(f"  面积={x_test[i]*1000:.0f} sqft → 预测价格=${y_test[i]:.2f}K")

# ============================================================
# 7. 模型参数的影响分析
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 不同 w 值（斜率）
for ax, (w_val, b_val, title) in zip(axes, [
    (100, 100, 'w=100 (较小平坦)'),
    (200, 100, 'w=200 (适中)'),
    (400, 100, 'w=400 (较大陡峭)')
]):
    f_wb = compute_model_output(x_train, w_val, b_val)
    ax.scatter(x_train, y_train, marker='x', c='r', s=100)
    ax.plot(x_train, f_wb, c='b', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("面积"); ax.set_ylabel("价格")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_param_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("参数影响图已保存为 task1_param_effect.png")

print("\n===== 任务1完成 =====")
