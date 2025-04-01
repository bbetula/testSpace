import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,confusion_matrix, classification_report
from make_moons import make_moons_3d  # 假设将数据生成函数保存在make_moons.py中

# 生成训练数据 (1000 samples)
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)  # 默认每类500，共1000

# 生成测试数据 (500 samples，250 per class)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)  # 每类250，共500

# 创建决策树分类器（设置随机种子保证可重复性）
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 评估性能
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

#预测准确率
print(f"Accuracy: {acc:.2f}")
# 评估（二分类专用指标）
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
print("\nConfusion Matrix:")
print(cm)

# 可视化预测结果（3D）
fig = plt.figure(figsize=(12, 5))

save_path="Kernel Methods\Assignment 2\picture\Decision_trees"
# 真实标签
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=y_test, cmap='viridis', s=20)
ax1.set_title("True Labels")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 预测结果
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=y_pred, cmap='viridis', s=20)
ax2.set_title("Decision Tree Predictions")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.savefig(f"{save_path}/Decision_tree_predictions.png")

# plt.tight_layout() 
# plt.show()


