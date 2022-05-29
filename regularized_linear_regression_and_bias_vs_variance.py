# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:44:50 2021

@author: taylo
"""
import os
import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy import random

os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\5- bias_vs_variance')
data = scio.loadmat('ex5data1.mat') #读取出来的data是字典格式
#training set
X_train = data['X']
y_train = data['y']

#cross validation set
X_val = data['Xval']
y_val = data['yval']
#test set
X_test = data['Xtest']
y_test = data['ytest']

#insert bias unit
x_train = np.insert(X_train,0,1,axis = 1)
x_val = np.insert(X_val,0,1,axis = 1)
x_test = np.insert(X_test,0,1,axis = 1)

theta = np.ones(x_train.shape[1])
inner = np.dot(x_train, theta) 
#test = inner - y_train     #!!!注意inner列表形式跟y_train不一样！！！ (一个是只有一对大括号，另一个是每个元素都有大括号，所以y要flatten)减出来不是期望的减法 （12*12）
Y = y_train.flatten()

def plot_data():
    fig,ax = plt.subplots()
    ax.scatter(x_train[:,1],y_train)
    ax.set(xlabel='Change in water level (x)',ylabel='Water flowing out of the dam (y)')

#plot = plot_data()

def reg_costfunction(theta,X,Y,lamda):
    inner = np.dot(X, theta)    #(12,1)
    term = np.sum(np.power((inner-Y.flatten()),2))
    reg = lamda*np.sum((np.power(theta[1:],2)))
    return (term+reg)/(2*len(X))

theta = np.ones(x_train.shape[1]) #(2,1)
lamda = 0
regcostfunc = reg_costfunction(theta,x_train, y_train, lamda)
#print(regcostfunc) 对的

def reg_gradientdescent(theta,X,Y,lamda):
#    theta[0] = 0  <下面的x*theta还用到theta所以不能直接这样设>
#    theta_af = (np.dot(X.T,(np.dot(X, theta)-Y.flatten())) + (lamda*theta))/len(X)
    term_1 = np.dot(X.T,(np.dot(X, theta)-Y.flatten()))
    reg = lamda*theta
    reg[0] = 0
    return(term_1+reg)/len(X)
#    return theta_af
reg_gradientdescent(theta,x_train, y_train, lamda)


def training(X, Y, lamda):
    theta_init = np.ones(X.shape[1]) #(2,1)#在后面这个是(9,1)
    res = minimize(fun = reg_costfunction, x0= theta_init, args = (X, Y, lamda), method = 'TNC', jac = reg_gradientdescent)
    return res.x     #返回最优参数
theta_final = training(x_train, y_train, lamda = 0)
print(theta_final)

#拟合的直线可视化(对的)
#plot_data()
#plt.plot(x_train[:,1],np.dot(x_train,theta_final), c = 'r')
#plt.show()

#bias variance
#training set error,cross validation error
def plot_learning_curve(x_train,y_train,x_val,y_val,lamda):
    X = range(1,len(x_train)+1) #从1到12
    error_train=[]
    error_val = []
    #训练集取样本个数i个
    for i in X:
        res = training(x_train[:i,:], y_train[:i,:], lamda)
        training_set_error = reg_costfunction(res,x_train[:i,:], y_train[:i,:], 0)
        error_train.append(training_set_error)
        cross_validation_error = reg_costfunction(res,x_val, y_val, 0)
        error_val.append(cross_validation_error)
#lamda等于0是因为题目上要求In particular, note that the training error does not include the regularization term.
#同一个表里有俩线
    plt.plot(X,error_train, label = 'Train')
    plt.plot(X,error_val, label = 'Cross Validation')
    plt.legend()
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.show()

#plot_learning_curve(x_train,y_train,x_val,y_val,lamda)
#you can observe that both the train error and cross validation
#error are high when the number of training examples is increased
#This reflects a high bias problem in the model,the linear regression model is
#too simple and is unable to fit our dataset well. In the next section, you will
#implement polynomial regression to fit a better model for this dataset.
#address the bias problem by adding more features

# map features into various powers of the original value (waterLevel)
def mapping(X,p):
    for i in range(2,p+1):   #[2,p]
        X = np.insert(X, X.shape[1], np.power(X[:,1],i), axis = 1)
    return X
#除了第一列的1，第二列保持原状，全部按照第二列x方掉然后插入最后一列
p = 6
x_train_poly = mapping(x_train,p)
x_val_poly = mapping(x_val,p)
x_test_poly = mapping(x_test,p)

#feature normalization
def get_mean_std(X):
    mean = np.mean(X, axis = 0) #按行取均值和方差(就是每一列的均值方差)
    std = np.std(X, axis = 0)
    return mean, std
def feature_normalization(X,mean,std):
    X[:,1:] = (X[:,1:]-mean[1:])/std[1:]    #第一列全为1，不用归一化
    return X
train_mean, train_std = get_mean_std(x_train_poly)
#不懂为啥全部都用training set的mean和std
x_train_norm = feature_normalization(x_train_poly,train_mean, train_std)
x_val_norm = feature_normalization(x_val_poly,train_mean, train_std)
x_test_norm = feature_normalization(x_test_poly,train_mean, train_std)

#注意这里y_train不用做改变
theta_poly = training(x_train_norm, y_train, lamda = 1)

#plot fitting curve
#plt.plot(x_train_poly[:,1],np.dot(x_train_norm,theta_poly))错的
def plot_poly_fit():
    plot_data()
    x = np.linspace(-60,60,100)
    xx = x.reshape(100,1)
    xx = np.insert(xx,0,1,axis=1)
    xx = mapping(xx,p)
    xx = feature_normalization(xx,train_mean,train_std)
    plt.plot(x,np.dot(xx,theta_poly),'r--')
#plot_poly_fit()

#However, the
#polynomial fit is very complex and even drops off at the extremes. This is
#an indicator that the polynomial regression model is overfitting the training
#data and will not generalize well.

#changing lamda,plot error/number of training example(learning curve)
#λ=0，训练集过拟合（高方差），表现为验证集误差较大，此时可以适当增加 λ 
#plot_learning_curve(x_train_norm,y_train,x_val_norm,y_val,0)
#λ=1，过拟合基本消除，最后 Jtrain(θ)和Jcv(θ)趋于相等，模型泛化效果好；it achieves a good trade-off between
#bias and variance.
#plot_learning_curve(x_train_norm,y_train,x_val_norm,y_val,1)
#λ = 100 \lambda=100λ=100，正则化项过大，模型偏差大，欠拟合
#plot_learning_curve(x_train_norm,y_train,x_val_norm,y_val,100)

#implement an automated method to select the λ parameter
def choose_lamda(x_train_norm,y_train,x_val_norm,y_val,x_test_norm,y_test):
    lamda = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    training_cost = []
    cross_validation_error = []
    for i in lamda:
        theta_cholam = training(x_train_norm, y_train, i)
        training_cost.append(reg_costfunction(theta_cholam,x_train_norm, y_train, lamda = 0))
        cross_validation_error.append(reg_costfunction(theta_cholam,x_val_norm,y_val,lamda = 0))
    lamda_result_index = np.argmin(cross_validation_error)   #默认将列表展平，显示最小值索引
    theta_result = training(x_train_norm, y_train, lamda[lamda_result_index])
    test_set_err_result = reg_costfunction(theta_result,x_test_norm,y_test,lamda = 0)
    return test_set_err_result,lamda[lamda_result_index]
#    plt.plot(lamda,training_cost,label = 'Train')
#    plt.plot(lamda,cross_validation_error,label = 'Cross Validation')
#    plt.legend()
#    plt.xlabel('Lamda')
#    plt.ylabel('Error')
#    plt.show()
#Due to randomness in the training and validation splits of the dataset, the cross validation error
#can sometimes be lower than the training error.
best_lamda_cost_result,best_lamda_index = choose_lamda(x_train_norm,y_train,x_val_norm,y_val,x_test_norm,y_test)
print(best_lamda_cost_result)
#将best_lamda放在测试集上，绘制训练集，验证集，测试集学习曲线
def plot_learning_curve_best_lamda(x_train_norm,y_train,x_val_norm,y_val,x_test_norm,y_test,lamda):
    X = range(1,len(x_train_norm)+1) #从1到12
    error_train=[]
    error_val = []
    error_test = []
    #训练集取样本个数i个
    for i in X:
        res = training(x_train_norm[:i,:], y_train[:i,:], lamda)
        training_set_error = reg_costfunction(res,x_train_norm[:i,:], y_train[:i,:], 0)
        error_train.append(training_set_error)
        cross_validation_error = reg_costfunction(res,x_val_norm, y_val, 0)
        error_val.append(cross_validation_error)
        test_error = reg_costfunction(res,x_test_norm, y_test, 0)
        error_test.append(test_error)
        
#lamda等于0是因为题目上要求In particular, note that the training error does not include the regularization term.
#同一个表里有俩线
    plt.plot(X,error_train, label = 'Train')
    plt.plot(X,error_val, label = 'Cross Validation')
    plt.plot(X,error_test, label = 'Test')
    plt.legend()
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.show()
plot_learning_curve_best_lamda(x_train_norm,y_train,x_val_norm,y_val,x_test_norm,y_test,best_lamda_index)
