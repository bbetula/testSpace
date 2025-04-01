import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from make_moons import make_moons_3d
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

# 生成数据
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)  # 1000 points (500 per class)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)    # 500 points (250 per class)

# 初始化AdaBoost+决策树桩（确保二分类）
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),  # 弱分类器为单层决策树
    n_estimators=50, # 50个弱分类器
    learning_rate=0.8, # 学习率
    algorithm='SAMME',  # 专为二分类设计的算法
    random_state=42 # 随机种子，确保可重复性
)

# 训练
adaboost.fit(X_train, y_train)

# 预测
y_pred = adaboost.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
# 评估（二分类专用指标）
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
save_path="Kernel Methods\Assignment 2\picture\AdaBoost"
# 可视化决策边界（3D）
def plot_decision_boundary():
    fig = plt.figure(figsize=(12, 5))
    
    # 真实分布
    ax1 = fig.add_subplot(121, projection='3d')
    sc1=ax1.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='coolwarm', s=20)
    ax1.set_title("True Labels")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 预测结果
    ax2 = fig.add_subplot(122, projection='3d')
    sc2=ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='coolwarm', s=20)
    ax2.set_title("AdaBoost Predictions")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.savefig(f"{save_path}/AdaBoost_predictions.png")
    # plt.tight_layout()
    # plt.show()

plot_decision_boundary()

# 特征重要性（二分类适用）
plt.figure()
plt.barh(['X', 'Y', 'Z'], adaboost.feature_importances_) 
plt.title("特征重要性")
plt.savefig(f"{save_path}/feature_importance.png")
# plt.show()