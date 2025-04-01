#线形模型拟合的不是很理想，尝试用非线性模型拟合
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import leastsq
input_path=r"Learning Models\Assignment 1\train\data.txt"
test_path=r"Learning Models\Assignment 1\predict\new_data.txt"
#读取txt文件
def read_data(path):
    data=pd.read_csv(path,sep="\t")
    return data
data=read_data(input_path)
test_data=read_data(test_path)
#数据预处理 x和y可以看作一个一维数组
def data_preprocess(data):
    x=data.iloc[:,0].values
    y=data.iloc[:,1].values 
    x=np.array(x,dtype=float)
    y=np.array(y,dtype=float)
    return x,y
x,y=data_preprocess(data) 
x_test,y_test=data_preprocess(test_data)
#非线性模型拟合an
#Report:e大概5.0为上限
# 定义非线性模型（三次多项式 + 正弦项）
def func(p, x):
    a, b, c, d, e, f, g = p
    return a * x**3 + b * x**2 + c * x + d + e * np.sin(f * x + g)

# 定义残差函数
def residuals(p, x, y):
    return y - func(p, x)

# 初始参数估计（通过多项式拟合初步估计趋势部分
p_poly = np.polyfit(x, y, 3)  # 三次多项式系数
# 初始参数：[a, b, c, d, e, f, g] a,b,c,d为三次多项式系数，e为正弦项系数，f为正弦项频率，g为正弦项相位
p0 = [p_poly[0], p_poly[1], p_poly[2], p_poly[3], 5.0, 2.0, 0.0] 

# 使用最小二乘法拟合
plsq = leastsq(residuals, p0, args=(x, y))

# 获取拟合参数
params = plsq[0].round(6)
# 打印拟合曲线表达式
print("拟合曲线为：y=",params[0],"x^3+",params[1],"x^2+",params[2],"x+",params[3],"+",params[4],"sin(",params[5],"x+",params[6],")")

# 生成拟合曲线
x_fit = np.linspace(x.min(), x.max(), 500)
y_fit = func(params, x_fit)

# 计算拟合曲线的均方误差
train_MSE = np.sum((y - func(params, x))**2) / len(y)
print("Nonlinear_train_MSE：",train_MSE)

# 绘制原始数据和拟合曲线
plt.scatter(x, y, label='Original Data', color='blue')
plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig("Learning Models/Assignment 1/picture/Nonlinear Regression.png")