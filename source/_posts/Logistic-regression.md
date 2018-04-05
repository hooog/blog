---
title: scikit-learn逻辑回归
date: 2018-04-5 21:05:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn
---

## 算法原理
### 逻辑回归算法的预测函数
参考：[逻辑回归梯度下降算法推导](http://blog.kamidox.com/logistic-regression.html)

这是一个非黑即白的世界，我们需要找出一个函数模型使其输出值在[0,1]之间，然后选择0.5位基准值大于0.5就认为预测值为1反之为0。

该函数就是
`Sigmoid函数，也叫做Logistic函数：`
$g\left( x \right)\; =\; \frac{1}{1+e^{-z}}$

它结合线性回归函数$h_{\theta }\left( x \right)\; =\; \theta ^{T}x$，生成逻辑回归算法的预测函数：
$h_{\theta }\left( x \right)\; =\; g\left( z \right)=g\left( \theta ^{T}x \right)=\frac{1}{1+e^{-\theta ^{^{T}x}}}$

### 成本函数
我们不能用线性回归模型的成本函数来推导逻辑回归的成本函数那样态复杂。为了方便求出成本函数我们分别分成y=1和y=0两种情况分别考虑其预测值与真实值的误差：
$$\;\mbox{C}ost\left(h_{\theta }\left(x\right),y\right)\; =\;\left[\begin{array}{c}-\log\left(h_{\theta}\left(x\right)\right),\;\;\;\;\; y\;=1 \\ -\log\left(1-h_{\theta }\left( x \right) \right),\; \; \; y\; =0 \end{array} \right]$$

成本函数统一写法：
$$\; \mbox{C}ost\left( h_{\theta }\left( x \right),y \right)\; =\; -y\log \left( h_{\theta }\left( x \right) \right)\; -\; \left( 1-y \right)\log \left( 1-h_{\theta }\left( x \right) \right)$$

由于y是{0,1}区间内的离散值，当y=1时， 1-y=0，上半式的后半部分为0；反之亦然。因此上市与分表达的成本计算公式是等价的。至此根据一个样本的计算公式，很容易卸除所有样本的成本平均值，可以得出：
$$J\left( \theta  \right)\; =\; -\frac{1}{m}\left[ \sum_{i=1}^{m}{y^{\left( i \right)}\log \left( h_{\theta }\left( x^{\left( i \right)} \right) \right)\; +\; \left( 1-y^{\left( i \right)} \right)\log \left( 1-h_{\theta }\left( x^{\left( i \right)} \right) \right)} \right]$$

### 梯度下降算法
根据梯度下降算法的定义可以得出：



$$\theta_{j\; =\; }\theta_{j\; }-\; \alpha \frac{\partial }{\partial \theta_{j}}J\left( \theta\right)$$
这里的关键是求解成本函数的偏导数，最终高推导出来的梯度下降算法公式为：

$$\theta_{j\;=\;}\theta_{j\;}-\;\alpha\frac{1}{m}\sum_{i=1}^{m}{\left(h_{\theta }\left(x^{\left(i\right)\;}\right)\;-\;y^{\left(i\right)}\right)x_{j}^{\left(i\right)}}$$

## 多元分类

逻辑回归解决多元分类问题的思路是：y={0,1,2,...n}，总共有n+1个类别，首先把问题转化为二元分类问题，分别把y=0作为一个类别同时y={1,2,3...n}作为另一个类别；接着把y=1作为一个类别，以此类推，在计算他们的概率。这里总共需要n+1个预测函数：
$$$$
预测出来的概率最高的类别，就是样本所属的类别。（因为概率越接近1成本函数值越低，越接近真实值）

## 正则化
过拟合是指墨芯很好的拟合了驯良样本，但对新数据预测的准确性很差，这是因为模型太复杂了。解决办法是减少输入特征的个数，或者获取更多的驯良样本。这里介绍正则化也是用来解决模型过拟合问题的一个方法。
- 保留所有特征，减小特征的权重$\theta_j$的值，确保所有的特征对预测值都有很少量大贡献。
- 当每个特征x对预测值都有少量的贡献时，模型就可以很好的工作，也就是正则化的目的，可以用来解决特征过多时的过拟合问题

### 线性回归模型的正则化

$$J\left(\theta\right)\;=\;\frac{1}{2m}\left[\sum_{i=1}^{m}{\left(h_{\theta}\left(x^{\left(i\right)}\right)\;-\;y^{\left(i\right)}\right)^{2}}\right]\;+\;\lambda\sum_{j=1}^{n}{\theta_{j}^{2}}$$

公式的前半部分局势线性回归的成本函数，后半部分加入**正则项**。其中$λ$的值有两个目的，既要维持对训练样本的拟合，又要避免对训练样本的过拟合。如果λ太大，则能确保不出现过拟合，但可能对现有训练样本出现欠拟合。
从数学的角度来看，成本函数增加了一个正则项 $\lambda\sum_{j=1}^{n}{\theta_{j}^{2}}$后，成本函数不再唯一的由预测值与真实值的误差所决定，还和参数$\theta$的大小有关。有了这个限制后要实现成本函数最小的目的，$\theta$就不能随便取值。比如某个较大的$\theta$值可能会让预测值与真实值的误差$\left( h_{\theta }\left( x^{\left( i \right)} \right)-y^{\left( i \right)} \right)^{2}$值很小，但是会导致$\theta_{j}^{2}$很大，最终的结果是成本函数太大。这样通过调节参数λ就可以控制正则项的权重，从而避免线性回归算法过拟合。
利用正则化的成本函数，可以推导出正则化后的参数迭代函数：


$$\theta_{j}\;=\;\theta_{j}\; -\; \alpha \frac{1}{m}\sum_{i=1}^{m}{\left[ \left( \left( h\left( x^{\left( i \right)} \right)\; -\; y^{\left( i \right)} \right)x_{j}^{\left( i \right)} \right)\; +\; \frac{\lambda }{m}\theta_{j} \right]}\; $$


$$\theta_{j}\;=\;\theta_{j}\; \left( 1\; -\; \alpha \frac{\lambda }{m} \right)\; -\; \alpha \frac{1}{m}\sum_{i=1}^{m}{}\left( \left( h\left( x^{\left( i \right)} \right)\; -\; y^{\left( i \right)} \right)x_{j}^{\left( i \right)} \right)\;$$

$\left( 1\; -\; \alpha \frac{\lambda }{m} \right)$因子在每次迭代时都将把$θ_{j}$ 收缩一点点。因为α和λ是正数，而 $m$ 是训练阳历的个数，是个比较大的正整数。为什么要对 $θ_{j }$ 进行收缩呢？因为加入正则项的成本函数和 $θ_{j}^{2}$ 成正比，所以迭代是需要不断试图减小 $θ_{j }$ 的值。

### 逻辑回归模型正则化

使用相同的思路，也可以对逻辑回归模型的成本函数进行正则化，其方法也是在原来的成本函数上加上正则项：
$$J\left( \theta  \right)\; =\; -\frac{1}{m}\left[ \sum_{i=1}^{m}{y^{\left( i \right)}\log \left( h_{\theta }\left( x^{\left( i \right)} \right) \right)\; +\; \left( 1-y^{\left( i \right)} \right)\log \left( 1-h_{\theta }\left( x^{\left( i \right)} \right) \right)} \right]\; +\; \frac{\lambda }{2m}\sum_{j=1}^{n}{\theta_{j}^{2}\;}$$
相应的，正则化后参数迭代公式为：

$$\theta_{j}\; \; =\; \theta_{j}\; -\; \alpha \frac{\partial }{\partial \theta_{j}}J\left( \theta  \right)$$
$$\theta_{j}\; =\; \theta_{j}\; -\; \alpha \left[ \frac{1}{m}\sum_{i=1}^{m}{}\left( h_{\theta }\left( x^{\left( i \right)} \right)\; -\; y^{\left( i \right)} \right)x_{j}^{\left( i \right)}\; +\; \frac{\lambda }{m}\theta_{j} \right]\;$$
$$\theta_{j}\; =\; \theta_{j}\; \left( 1\; -\; \alpha \frac{\lambda }{m} \right)\; -\; \alpha \frac{1}{m}\sum_{i=1}^{m}{}\left( \left( h\left( x^{\left( i \right)} \right)\; -\; y^{\left( i \right)} \right)x_{j}^{\left( i \right)} \right)\;$$

需要注意的是，上式中$j>=1$，因此$\theta_{0}$没有参与正则化。另外需要注意逻辑回归和线性回归的参数迭代算法看起来形式是一样的，但其实他们的算法不一样，因为两个式子的预测函数$h_{θ}({x})$不一样。针对线性回归：$h_{θ}({x}) = θ^{T}x$，而针对逻辑回归的是：$h_{\theta }\left( x \right)\; =\; g\left( z \right)=g\left( \theta ^{T}x \right)=\frac{1}{1+e^{-\theta ^{^{T}x}}}$



## 算法参数

### 正则项权重
上面介绍的正则项权重 λ ，在`LogisiticRegression`里有个参数 C 与之对应但成反比：
- C 越大正则项权重越小，模型容易出现过拟合
- C 越小正则项权重越大，模型容易出现欠拟合

### L1\L2范数

创建逻辑回归模型时，有个参数 penalty ，其取值有‘l1’或‘l2’，实际上就是指定前面介绍的正则项的形式。 在成本函数里添加的正则项 $\sum_{j=1}^{n}{\theta_{j}^{2}}$， 这时实际上就是个L2正则项。
L1:

$\left|\left|\theta\right|\right|_ {1}\; =\; \left|\theta_{1}\right|\;+\; \left| \theta_{2} \right|$
L2:


$\left| \left| \theta  \right| \right|_ {2}\; =\; \sqrt{\theta_{1}^{2}\; +\; \theta_{2}^{2}}$



L1 L1范数（L1 norm）是指向量中各个元素绝对值之和，也有个美称叫“稀疏规则算子”（Lasso regularization）。范数作为正则项，会让模型参数$\theta$稀疏化， 既让模型参数向量里为0的元素尽量多。在支持向量机（support vector machine）学习过程中，实际是一种对于成本函数(cost function)求解最优，得出稀疏解。
L2 范数作为正则项式让模型参数尽量小，但不会为0，尽量让每个特征对预测值都有一些小的贡献，得出稠密解。
在梯度下降算法的迭代过程中，实际上是在成本函数的等高线上跳跃，并最终收敛在误差最小的点上（此处为未加正则项之前的成本误差）。而正则项的本质就是**惩罚**。 模型在训练的过程中，如果没有遵守正则项所表达的规则，那么成本会变大，即受到了惩罚，从而往正则项所表达的规则处收敛。 成本函数哎这两项规则的综合作用下，正则化后的模型参数应该收敛在误差等值线与正则项等值线相切的点上。
作为推论， L1 范数作为正则项由以下几个用途：
- 特征选择： 它会让模型参数向量里的元素为0的点尽量多。 因此可以排除掉那些对预测值没有什么影响的特征，从而简化问题。所以 L1 范数解决过拟合措施实际上是减少特征数量。
- 可解释性： 模型参数向量稀疏化后，只会留下那些对预测值有重要影响的特征。 这样我们就容易解释模型的因果关系。 比如针对某个癌症的筛查，如果有100个特征，那么我们无从解释到底哪些特征对阳性成关键作用。 稀疏化后，只留下几个关键特征，就更容易看到因果关系

由此可见， L1 范数作为正则项，更多的是一个分析工具，而适合用来对模型求解。因为它会把不重要的特征直接去除。 大部分情况下，我们解决过拟合问题，还是选择 L2 单数作为正则项， 这也是 sklearn 里的默认值。

## 实例：乳腺癌检测

使用逻辑回归算法解决乳腺癌检测问题。 我们需要先采集肿瘤病灶造影图片， 然后对图片进行分析， 从图片中提取特征， 在根据特征来训练模型。 最终使用模型来检测新采集到的肿瘤病灶造影， 判断是良性还是恶性。 这个是典型的二元分类问题。



```python
# 加载数据
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(X.shape, y.shape,'\n', X[0], '\n', y[0])
```

    (569, 30) (569,) 
     [1.799e+01 1.038e+01 1.228e+02 1.001e+03 1.184e-01 2.776e-01 3.001e-01
     1.471e-01 2.419e-01 7.871e-02 1.095e+00 9.053e-01 8.589e+00 1.534e+02
     6.399e-03 4.904e-02 5.373e-02 1.587e-02 3.003e-02 6.193e-03 2.538e+01
     1.733e+01 1.846e+02 2.019e+03 1.622e-01 6.656e-01 7.119e-01 2.654e-01
     4.601e-01 1.189e-01] 
     0


实际上它只关注了 10 个特征，然后又构造出来每个特征的标准差及最大值，这样每个特征又衍生出了两个特征，所以共有30个特征。

这里该案例使用了特征提取手段，这一方法在实际工程应用中是很常用的。

举个例子：
我们需要监控数据中心每个物理主机的运行情况，其中 CPU 占用率、内存占用率、网络吞吐量是几个重要的指标。 问：有台主机 CPU 占用率80%， 这个主机状态是否正常？ 要不要发警告？ 这个要看情况！仅从 CPU 占用率来看还不能判断主机是否正常，还要看内存占用情况和网络吞吐量情况。 如果此时内存占用也成比例上升， 且网络吞吐量也在合理的水平，那么造成这一状态的可能是用户访问量过大， 导致主机负责增加， 不需要警告。 但如果内存占用、 网络吞吐量和 CPU 占用不在同一量级那么主机就可以处于不正常的状态。 所以这里需要构建一个复合特征， 如 CPU 占用率和内存占用率的比值， 以及 CPU 占用率和网络吞吐量的值， 这样构造出来的特征更真实地体现了现实问题中的内在规则。 

所以： **提取特征时，不妨从事物内在逻辑关系入手，分析已有特征之间的关系， 从二构造出新的特征**

疑问： 该方式是否直接会导致多重共线性的出现？


### 代码实现


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```


```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
print("train_score:", train_score)
print("test_score:", test_score)
```

    train_score: 0.9626373626373627
    test_score: 0.9473684210526315



```python
from sklearn.metrics import precision_score, recall_score, f1_score
print("查准率：", precision_score(y_test, y_pred))
print("召回率：", recall_score(y_test, y_pred))
print("F1Score：", f1_score(y_test, y_pred))

print(np.equal(y_pred, y_test).shape[0], y_test.shape[0]) # 输出预测匹配成功数量和测试样本的数量
```

    查准率： 0.9358974358974359
    召回率： 0.9733333333333334
    F1Score： 0.954248366013072
    114 114


这里数量上显示全部都预测正确，而test_score却不是1，是因为sklearn不是使用这个数据来计算得分，因为这个数据不能完全反映误差情况，而是使用预测概率来计算模型得分。

*那么查准率和召回率是否同理？*

### 查看预测自信度
二元分类模型会针对每个样本输出的两个概率，即0和1的概率，哪个概率高就预测器哪个类别。我们可以找出针对测试数据集，模型预测的“自信度”低于90%的样本。我们先计算出测试数据集里每个样本的预测概率数据，针对每个样本会有两个数据：一个预测为0，一个预测为1。结合找出预测为阴性和阳性的概率大于0.1的样本。我们可以看下概率数据：


```python
# 计算每个测试样本的预测概率：
y_pred_proba = model.predict_proba(X_test)
print("自信度示例：",y_pred_proba[0])
```

    自信度示例： [0.00452578 0.99547422]



```python
y_pred_proba_0 = y_pred_proba[:, 0] > 0.1
result = y_pred_proba[y_pred_proba_0]

y_pred_proba_1 = result[:, 1] > 0.1
print(result[y_pred_proba_1])
```

    [[0.11338788 0.88661212]
     [0.18245824 0.81754176]
     [0.13110396 0.86889604]
     [0.35245276 0.64754724]
     [0.30664405 0.69335595]
     [0.24931118 0.75068882]
     [0.8350464  0.1649536 ]
     [0.44807883 0.55192117]
     [0.74071324 0.25928676]
     [0.43085792 0.56914208]
     [0.13388416 0.86611584]
     [0.33507985 0.66492015]
     [0.53672412 0.46327588]
     [0.11422612 0.88577388]
     [0.42946531 0.57053469]
     [0.69759146 0.30240854]
     [0.25982004 0.74017996]
     [0.12179042 0.87820958]
     [0.88546887 0.11453113]]


### 模型优化

这里使用Pipeline来增加多项式特征


```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def poly_model(degree=2, penalty=penalty):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    log_regr = LogisticRegression(penalty=penalty) # 注意这里是L1而不是11，指的是使用L1范式作为其正则项
    pipeline = Pipeline([("poly_features",poly_features),("log_regr",log_regr)])
    return pipeline
```


```python
# 接着增加二阶多项式特征，创建并训练模型
import time

model = poly_model(degree=2, penalty='l1')

start = time.clock()
model.fit(X_train, y_train)

print("train_score:",model.score(X_train, y_train))
print("test_score:",model.score(X_test, y_test))
```

    train_score: 0.9934065934065934
    test_score: 0.9649122807017544


这里要注意的是使用L1范式作为其正则项，参数为`penalty=l1`。L1范数作为其正则项，可以实现参数的稀疏化，即自动帮我买选择出哪些对模型有关联的特征。我买可以观察下有多少个特征没有被丢弃即对应的模型参数$\theta_j$非0：


```python
log_regr = model.named_steps['log_regr']
print("特征总量：",log_regr.coef_.shape[1])
print("特征保留量：", np.count_nonzero(log_regr.coef_))
```

    特征总量： 495
    特征保留量： 114


逻辑回归模型的`coef_ `属性里保存的就是模型参数。 从输出结果看，增加二阶多项式特征后，输入特征由原来的30个增加到了595个，在L1范数的“惩罚”下最终只保留了92个有效特征
### 实验：利用决策树画出原始数据对预测相关性非0对特征
```python
from sklearn.tree import DecisionTreeRegressor
dtmodel = DecisionTreeRegressor(max_depth=5)
dtmodel.fit(X_train, y_train)
print("train_score", dtmodel.score(X_train, y_train))
print("test_score", dtmodel.score(X_test, y_test))
from pyecharts import Bar
index = np.nonzero(dtmodel.feature_importances_)
bar = Bar()
bar.add("", cancer.feature_names[index],dtmodel.feature_importances_[index])
bar
```
    train_score 0.9910875596851206
    test_score 0.6296416546416548

![png](/images/sklearn/10.png)

### 评估模型：画出学习曲线
首先画出L1范数作为正则项所对应的一阶和二阶多项式的学习曲线：
```python
%matplotlib inline
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learn_curve(estimator, title, X, y, ylim = None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1., 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("train exs")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_score_mean = np.mean(train_scores, axis=1)
    train_score_std = np.std(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    test_score_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_score_mean - train_score_std, 
                     train_score_mean + train_score_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_score_mean - test_score_std, 
                     test_score_mean + test_score_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_score_mean, 'o-', color='r', label='train score训练得分')
    plt.plot(train_sizes, test_score_mean, 'o-', color='g', label='cross-validation score交叉验证得分')
    
    plt.legend(loc='best')
    return plt
```


```python
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
titles = ["degree:1 penalty=L1","degree:2 penalty=L1"]
degrees = [1,2]
penalty = 'l1'

start = time.clock()
plt.figure(figsize=(12,4), dpi=120)
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i + 1)
    plot_learn_curve(poly_model(degree=degrees[i], penalty=penalty), titles[i],
                    X, y, ylim = (0.8, 1.01), cv = cv)

print('耗时：', time.clock() - start)
plt.show()
```

![png](/images/sklearn/11.png)

L2范数作为正则项画出对应一阶和二阶多项式学习曲线


```python
import time
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
titles = ["degree:1 penalty=L2","degree:2 penalty=L2"]
degrees = [1,2]
penalty = 'l2'

start = time.clock()
plt.figure(figsize=(12,4), dpi=120)
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i + 1)
    plot_learn_curve(poly_model(degree=degrees[i],penalty=penalty), titles[i],
                    X, y, ylim = (0.8, 1.01), cv = cv)

print('耗时：', time.clock() - start)
plt.show()
```

![png](/images/sklearn/12.png)

从上面两个图可以看出，使用二阶多项式并使用L1范数作为正则项的模型最优，训练样本评分最高，交叉验证样本评分最高。
训练样本评分和交叉验证样本评分之间的间隙还比较大，这说明可以通过采集更多数据来训练模型，以便进一步优化模型.

通过时间消耗对比上可以看出利用L1范式作为正则项需要花费的时间更多，是因为`sklearn`的`learning_curve()`函数在画学习曲线的过程中要对模型进行多次训练，并计算交叉验证样本评分。同时为了让曲线更平滑，针对每个点还会进行多次计算球平均值。这个就是`ShufferSplit`类的作用。在这个实例里只有569个样本是很小的数据集。如果数据集增加100倍，拿出来画学习曲线将是场灾难。

问题是针对大数据集，怎么画学习曲线？

思路一：可以考虑从大数据集选取一小部分数据来画学习曲线，待选择好最优的模型之后，在使用全部的数据来训练模型。这时需要警惕的是，尽量保证选择出来的这部分数据的**标签分布与大数据集的标签分布相同**，如针对二元分类，**阳性和阴性比例要一致！**
