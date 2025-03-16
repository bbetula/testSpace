#给定一组二维数据，data.txt为训练数据，new_data.txt为测试数据，数据格式如下：
#x	y_complex	第一行为列名，第二行开始为数据，x为自变量，y_complex为因变量
#1）分别使用最小二乘法，梯度下降法(GD)和牛顿法来对数据进行线性拟合，观察其训练误差与测试误差。				
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

#线性模型拟合
def print_ret(x,y,y_fit,y_test,y_test_fit,k,b,method):
    print("拟合方法为：",method)
    print("拟合曲线为：y=",k,"x+",b)
    print("训练误差为：",np.sum((y_fit-y)**2)/len(y))
    print("测试误差为：",np.sum((y_test_fit-y_test)**2)/len(y_test))
    save_fit_curve(x,y,y_fit,method)
#存取拟合的曲线
def save_fit_curve(x,y,y_fit,method):
    plt.title(method)
    plt.xlabel("x")
    plt.ylabel("y_complex")
    plt.scatter(x,y)
    #曲线颜色一律为黑色
    plt.plot(x,y_fit,color="black")
    plt.savefig("Learning Models/picture/"+method+".png")
#最小二乘法
def LS(x,y,x_test,y_test):
    def func(p,x):
        k,b=p
        return k*x+b
    def error(p,x,y):
        return func(p,x)-y
    p0=[1,1]
    Para=leastsq(error,p0,args=(x,y))
    k,b=Para[0]
    y_fit=k*x+b
    y_test_fit=k*x_test+b
    print_ret(x,y,y_fit,y_test,y_test_fit,k,b,"LS")
#梯度下降法
def GD(x,y,x_test,y_test):
    def func(p,x):
        k,b=p
        return k*x+b
    def error(p,x,y):
        return func(p,x)-y
    def gradient(p,x,y):
        k,b=p
        grad_k=np.sum(2*(k*x+b-y)*x)/len(y)
        grad_b=np.sum(2*(k*x+b-y))/len(y)
        return [grad_k,grad_b]
    def GD(p,x,y,lr=0.01):
        grad=gradient(p,x,y)
        p[0]=p[0]-lr*grad[0]
        p[1]=p[1]-lr*grad[1]
        return p
    p0=[1,1]
    for i in range(1000):
        p0=GD(p0,x,y)
    k,b=p0
    y_fit=k*x+b
    y_test_fit=k*x_test+b
    print_ret(x,y,y_fit,y_test,y_test_fit,k,b,"GD")
#牛顿法
def Newton(x,y,x_test,y_test):
    def func(p,x):
        k,b=p
        return k*x+b
    def error(p,x,y):
        return func(p,x)-y
    def gradient(p,x,y):
        k,b=p
        grad_k=np.sum(2*(k*x+b-y)*x)/len(y)
        grad_b=np.sum(2*(k*x+b-y))/len(y)
        return [grad_k,grad_b]
    def Hessian(p,x,y):
        k,b=p
        H_kk=np.sum(2*x*x)/len(y)
        H_kb=np.sum(2*x)/len(y)
        H_bb=len(y)
        return np.array([[H_kk,H_kb],[H_kb,H_bb]])
    def Newton(p,x,y):
        grad=gradient(p,x,y)
        H=Hessian(p,x,y)
        p=np.array(p)
        p=p-np.dot(np.linalg.inv(H),grad)
        return p
    p0=[1,1]
    for i in range(1000):
        p0=Newton(p0,x,y)
    k,b=p0
    y_fit=k*x+b
    y_test_fit=k*x_test+b
    print_ret(x,y,y_fit,y_test,y_test_fit,k,b,"Newton")


if __name__=="__main__":
    LS(x,y,x_test,y_test)
    GD(x,y,x_test,y_test)
    Newton(x,y,x_test,y_test)
