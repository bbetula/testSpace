import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from make_moons import make_moons_3d  # 导入数据生成函数
save_path="Kernel Methods\Assignment 2\picture\SVM"
# 生成数据
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)  # 训练集 (1000点)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)    # 测试集 (500点)

# 定义三种核函数的SVM
kernels = {
    'Linear': {'kernel': 'linear', 'C': 1.0}, # 线性核
    'RBF': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}, # RBF核
    'Polynomial': {'kernel': 'poly', 'C': 1.0, 'degree': 3, 'gamma': 'scale'} # 多项式核
}

# 训练并评估每种核函数
results = {}
for name, params in kernels.items():
    # 初始化模型
    svm = SVC(**params, random_state=42)
    
    # 训练
    svm.fit(X_train, y_train)
    
    # 预测
    y_pred = svm.predict(X_test)
    
    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    cm = confusion_matrix(y_test, y_pred)
    
    # 存储结果
    results[name] = {
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm,
        'model': svm
    }

# 打印结果
for name, res in results.items():
    print(f"\n=== {name} Kernel SVM ===")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print("\nClassification Report:")
    print(res['report'])
    print("\nConfusion Matrix:")
    print(res['confusion_matrix'])

# 可视化混淆矩阵
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    ConfusionMatrixDisplay(res['confusion_matrix'], display_labels=['Class 0', 'Class 1']).plot(ax=ax, cmap='Blues')
    ax.set_title(f'{name} Kernel')
plt.savefig(f"{save_path}/SVM_confusion_matrices.png")

# 可视化决策边界（3D切片）
def plot_decision_boundary_3d(model, X, y, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    z_slice = 0  # 固定Z轴值观察XY平面
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    grid_points = np.c_[xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), z_slice)]
    
    # 预测并绘制
    Z = model.predict(grid_points).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='coolwarm', s=20)
    ax.set_title(f"{title} (Z={z_slice:.1f} Slice)") #Slice是z轴的值
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f"{save_path}/{title}_decision_boundary.png")

# 绘制RBF核的决策边界（其他核类似）
plot_decision_boundary_3d(results['RBF']['model'], X_test, y_test, "RBF Kernel SVM")
