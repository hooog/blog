---
title: 一个自编译线性回归的案例
date: 2018-04-2 17:05:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn 
---
<!--more-->


在线性回归中，我们想要建立一个模型，来拟合一个因变量 y 与一个或多个独立自变量(预测变量) x 之间的关系。

给定：

数据集
$$\left\{ \left( x^{\left( 1 \right)},y^{\left( 1 \right)} \right),\; ...,\; \left(x^{\left( m \right)},y^{\left( m \right)} \right) \right\}$$

$x_{i}$是d-维向量$X^{i}\; =\; \left( x_{1}^{\left( i \right)},\; ...,\; x_{d}^{\left( i \right)} \right)$

$y^{(i)}$是一个目标变量，它是一个标量

线性回归模型可以理解为一个非常简单的神经网络：

它有一个实值加权向量$w\; =\; \left( w^{\left( i \right)},\; ...,\; w^{\left( d \right)} \right)$
它有一个实值偏置量 b
它使用恒等函数作为其激活函数

线性回归模型可以使用以下方法进行训练

a) **梯度下降法**

b) **正态方程(封闭形式解)**： $w\; =\; \left( X^{T}X \right)^{-1}X^{T}y$

其中 X 是一个矩阵，其形式为$\left( m,\; n_{featu\mbox{re}s} \right)$，包含所有训练样本的维度信息。

而正态方程需要计算$\left( X^{T}X \right)$的转置。这个操作的计算复杂度介于$O\left( n_{featu\mbox{re}s}^{2.4} \right)$和$O\left( n_{featu\mbox{re}s}^{3} \right)$之间，而这取决于所选择的实现方法。因此，如果训练集中数据的特征数量很大，那么使用正态方程训练的过程将变得非常缓慢。
 
线性回归模型的训练过程有不同的步骤。首先(在步骤 0 中)，模型的参数将被初始化。在达到指定训练次数或参数收敛前，重复以下其他步骤。


### 第 0 步：

用0 (或小的随机值)来初始化权重向量和偏置量，或者直接使用正态方程计算模型参数

### 第 1 步(只有在使用梯度下降法训练时需要)：

计算输入的特征与权重值的线性组合，这可以通过矢量化和矢量传播来对所有训练样本进行处理：
$\dot{y}\; =\; X\; \cdot \; w\; +b$

其中 X 是所有训练样本的维度矩阵，其形式为$\left( m,\; n_{featu\mbox{re}s} \right)$；这里我用· 表示$\wedge$ 。

### 第 2 步(只有在使用梯度下降法训练时需要)：

用均方误差计算训练集上的损失：$J\left( w,b \right)\; =\; \frac{1}{m}\sum_{i=1}^{m}{\left( \dot{y}^{\left( i \right)}\; -\; y^{\left( i \right)} \right)^{2}}$

### 第 3 步(只有在使用梯度下降法训练时需要):

对每个参数，计算其对损失函数的偏导数：

$\frac{\partial J}{\partial w_{j}}\; =\; \frac{2}{m}\sum_{i=1}^{m}{\left( \dot{y}^{\left( i \right)}\; -\; y^{\left( i \right)} \right)}x_{j}^{\left( i \right)}$

$\frac{\partial J}{\partial b}\; =\; \frac{2}{m}\sum_{i=1}^{m}{\left( \dot{y}^{\left( i \right)}\; -\; y^{\left( i \right)} \right)}$

所有偏导数的梯度计算如下：

$\Delta _{w}J\; =\; \frac{2}{m}X^{T}\; \left( \dot{y}\; -\; y \right)$

$\Delta _{b}J\; =\; \frac{2}{m}\left( \dot{y}\; -\; y \right)$

### 第 4 步(只有在使用梯度下降法训练时需要）:

更新权重向量和偏置量：

$w\; =\; w\; -\; \eta \Delta _{w}J$

$\Delta _{b}J\; =\; \frac{2}{m}\left( \dot{y}\; -\; y \right)$

其中η表示学习率



### 代码实现

#### 数据集


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(123)

X = 2 * np.random.rand(500, 1)
y = 5 + 3 * X + np.random.randn(500, 1)
fig = plt.figure(figsize=(8,6))
plt.scatter(X, y)
plt.title("Dataset")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
```

![png](/images/sklearn/7.png)


```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')
```

    Shape X_train: (375, 1)
    Shape y_train: (375, 1)
    Shape X_test: (125, 1)
    Shape y_test: (125, 1)


#### 线性回归分类 源码编译


```python
 class LinearRegression:

    def __init__(self):
        pass

    def train_gradient_descent(self, X, y, learning_rate=0.01, n_iters=100):
        """
        Trains a linear regression model using gradient descent
        """
        # Step 0: Initialize the parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(shape=(n_features,1))
        self.bias = 0
        costs = []

        for i in range(n_iters):
            # Step 1: Compute a linear combination of the input features and weights
            y_predict = np.dot(X, self.weights) + self.bias

            # Step 2: Compute cost over training set
            cost = (1 / n_samples) * np.sum((y_predict - y)**2)
            costs.append(cost)

            if i % 100 == 0:
                print(f"Cost at iteration {i}: {cost}")

            # Step 3: Compute the gradients
            dJ_dw = (2 / n_samples) * np.dot(X.T, (y_predict - y))
            dJ_db = (2 / n_samples) * np.sum((y_predict - y)) 

            # Step 4: Update the parameters
            self.weights = self.weights - learning_rate * dJ_dw
            self.bias = self.bias - learning_rate * dJ_db

        return self.weights, self.bias, costs

    def train_normal_equation(self, X, y):
        """
        Trains a linear regression model using the normal equation
        """
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        self.bias = 0

        return self.weights, self.bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

#### 使用梯度下降进行训练


```python
regressor = LinearRegression()
w_trained, b_trained, costs = regressor.train_gradient_descent(X_train, y_train, learning_rate=0.005, n_iters=600)
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(600), costs)
plt.title("Development of cost during training")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()
```

    Cost at iteration 0: 66.45256981003433
    Cost at iteration 100: 2.208434614609594
    Cost at iteration 200: 1.2797812854182806
    Cost at iteration 300: 1.2042189195356685
    Cost at iteration 400: 1.1564867816573
    Cost at iteration 500: 1.121391041394467





    Text(0,0.5,'Cost')



![png](/images/sklearn/8.png)

#### 测试（梯度下降模型）


```python
n_samples, _ = X_train.shape
n_samples_test, _ = X_test.shape

y_p_train = regressor.predict(X_train)
y_p_test = regressor.predict(X_test)

error_train =  (1 / n_samples) * np.sum((y_p_train - y_train) ** 2)
error_test =  (1 / n_samples_test) * np.sum((y_p_test - y_test) ** 2)

print(f"Error on training set: {np.round(error_train, 4)}")
print(f"Error on test set: {np.round(error_test)}")
```

    Error on training set: 1.0955
    Error on test set: 1.0


#### 使用正规方程（normal equation）训练


```python
X_b_train = np.c_[np.ones((n_samples)), X_train]
X_b_test = np.c_[np.ones((n_samples_test)), X_test]

reg_normal = LinearRegression()
w_trained = reg_normal.train_normal_equation(X_b_train, y_train)
```

#### 测试（正规方程模型）


```python
y_p_train = reg_normal.predict(X_b_train)
y_p_test = reg_normal.predict(X_b_test)

error_train =  (1 / n_samples) * np.sum((y_p_train - y_train) ** 2)
error_test =  (1 / n_samples_test) * np.sum((y_p_test - y_test) ** 2)

print(f"Error on training set: {np.round(error_train, 4)}")
print(f"Error on test set: {np.round(error_test, 4)}")
```

    Error on training set: 1.0228
    Error on test set: 1.0432


#### 可视化测试预测


```python
fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_p_test)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
```




    Text(0,0.5,'Second feature')



![png](/images/sklearn/9.png)
