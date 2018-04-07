---
title: DecisionTree决策树大全
date: 2018-04-7 1:05:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn 
---
<!--more-->

##  利用信息墒判定先对那个特征进行分裂
信息墒是衡量信息不确定性的指标，信息墒公式：
$$H\left( X \right)=-\sum_{x\in X}^{}{P\left( x \right)\log_{2}P\left( x \right)}$$
其中$P(x)$表示事件$x$出现的概率。回到决策树的构建问题上，我们遍历所有特征，分别计算，使用这个公式划分数据集前后信息墒的变化值，然后选择信息墒变化幅度最大的那个特征来作为数据集划分依据。即：选择**信息增益**最大的特征作为分裂节点。

这里以概率$P(X)$为横坐标，以信息墒$Entropy$为纵坐标把信息墒和概率的函数关系$\mbox{E}ntropy=-P\left( x \right)\log_{2}P\left( x \right)$在二维坐标上画出来，可以看出，当概率$P(X)$越接近0或越接近1时，信息墒的值越小。当概率值为1时信息墒为0，此时数据是最“纯净”的。我们在选择特征时，选择信息增益最大的特征，物理上即让数据尽量往更纯净的方向上变换。因此**信息增益**是用来衡量数据变得更有序更纯净的程度的指标。

*延伸一下：如写博客或者看书的过程，是减熵的过程。通过阅读和写作减少了不确定的信息，从而实现减熵。人生价值的实现，在于消费资源（增熵过程）来获取能量，经过自己的劳动付出（减熵过程），让世界变得更加纯净有序，`信息增益 = 减熵量 - 增熵量`即是衡量人生价值的尺度*。

## 决策树的创建
决策树的构建过程，就是从训练集中归纳出一组分类规则，使它与熟练数据矛盾较小的同时具有较强的泛化能力。基本分为以下几步：

- 计算数据集划分钱的信息墒
- 遍历所有为作为划分条件的特征，分别计算根据每个特征划分数据集后的信息墒。
- 选择信息增益最大的特征，并使用这个特镇作为数据划分节点来划分数据。
- 递归地处理被划分后的所子数据集，从未被选择的特征里继续重复以上步骤，选择出最优数据划分特征来划分子数据集...

这里递归结束的条件一般有两个：**一是所有特征都用完了，二是划分后的信息墒增益足够小了**。针对这个停止条件，需要实现选择信息增益的门限值来作为递归结束条件。

使用信息增益作为特征选择指标的决策树构建算法，称为**ID3算法**

1、离散化

当特征数据是连续值时，就需要先对数据进行离散化处理。（如考试得分0～100，1～59为不及格，60～80为达标，80～100为优秀）这样离散处理后的数据就可以用来构建决策树了。

2、**正则项（重要）**

最大化信息增益来选择特征，在决策树的构建过程中，容易造成优先选择类别最多的特征进行分裂，因为这样划分后的子数据集最“纯净”，其信息增益最大。但这不是我们想看到的结果。解决办法如下：

- **计算划分后的子数据集的信息墒时，加上一个与类别个数成正比的正则项来作为最后的信息墒**：这样当算法选择的某个类别较多的特征，使信息墒较小时，由于受到正则项的“惩罚”，导致最终的信息墒也较大。这样通过合适的参数可以使算法训练得到某种程度的平衡。

- **使用信息增益比**来作为特征选择的标准。

3、基尼不纯度

信息墒是横量信息不确定性的指标，实际上也是衡量信息“纯度”的指标。

基尼不纯度`(Gini impurity)`也是衡量信息不纯度的指标，公式如下：

$$Gini\left( D \right)=\sum_{x\in X}^{_ {_ {\;}}}{P\left( x \right)\left( 1-P\left( x \right) \right)=1-}\sum_{x\in X}^{_ {_ {\; }}}{P\left( x \right)^{2}}$$
同样，这里以概率$P(X)$为横坐标，以信息墒$Gini(x)$为纵坐标把信息墒和概率的函数关系在二维坐标上画出来，可以看出其形状几乎和信息墒的形状一样。`CART`算法使用基尼不纯度来作为特征选择标准，`GART`也是一种决策树构建算法。

## 剪枝算法
使用决策树模型拟合数据时，容易产生过拟合。解决办法是对决策树进行剪枝处理。决策树剪枝有两种思路：

1、**前剪枝**`（Pre-Pruning）`

在构造决策树的同时进行剪枝。在决策树构建中，如果无法进一步降低信息墒的情况下就会停止创建分支。为了避免过拟合，可以设定一个阀值，信息墒见效的数量小于这个阀值，即是还可以继续降低熵也停止继续创建分支。这种方法就是前剪枝。

2、**后剪枝**`（Post-Pruning）`

后剪枝是指决策树构造完成后进行剪枝。剪枝的过程是对拥有同样符节点的一组节点进行检查，判断如果将其合并，信息墒的增加量是否小于某一阀值。如果小于阀值即可合并分支。

后剪枝是目前比较普遍的做法。后剪枝的过程就是删除一些子树，然后用子树的根节点代替作为新的叶子节点。这个新叶子所标示的类别通过大多数原则来确定。即把这个叶子节点里样本最多的类别，作为这个叶子节点的类别。

后剪枝的算法有很多种，其中常见的一种称为**减低错误率剪枝法（Reduced-Errorpruning）**。其思路是自底向上，从已经构建好的完全决策树中找出一个子树，然后用子树的根节点代替这颗子树，作为新的叶子节点。叶子节点所表示的类别通过大多数原则确定，这样就构建出一个简化版决策树。然后使用交叉验证数据集来测试简化版本的决策树，看看其错误率是不是降低了。如果错误率降低了，则可以用这个简化版的决策树来代替完全决策树，否则还采用原来的决策树。通过遍历所有的子树，直到针对交叉验证数据集无法进一步降低错误率为止。


## sklearn种决策树的算法参数

### 1、模型参数
sklern中使用`sklearn.tree.DecisionTreeClassifier`类来实现决策树分类算法。其实几个典型的参数解释如下：

| 名称|功能|   描述   |
|:--|:--------:| :------ |
|criterion|特征选择标准| ‘gini’ or ‘entropy’ (default=”gini”)，前者是基尼系数，后者是信息熵。两种算法差异不大对准确率无影响，信息墒云孙效率低一点，因为它有对数运算.一般说使用默认的基尼系数”gini”就可以了，即CART算法。除非你更喜欢类似ID3, C4.5的最优特征选择方法。|
| splitter |特征划分标准| ‘best’ or ‘random’ (default=”best”) 前者在特征的所有划分点中找出最优的划分点。后者是随机的在部分划分点中找局部最优的划分点。 默认的”best”适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐”random” 。|
|max_depth|决策树最大深度|int or None, optional (default=None) 一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。常用来解决过拟合|
|min_impurity_decrease|节点划分最小不纯度|float, optional (default=0.) 这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点。 sklearn 0.19.1版本之前叫 min_impurity_split|
|min_samples_split|内部节点再划分所需最小样本数|int, float, optional (default=2) 如果是 int，则取传入值本身作为最小样本数； 如果是 float，则去 ceil(min_samples_split * 样本数量) 的值作为最小样本数，即向上取整。 |
|min_samples_leaf|叶子节点最少样本数|如果是 int，则取传入值本身作为最小样本数； 如果是 float，则去 ceil(min_samples_leaf * 样本数量) 的值作为最小样本数，即向上取整。 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。|
|max_leaf_nodes|最大叶子节点数| int or None, optional (default=None) 通过限制最大叶子节点数，可以防止过拟合，默认是”None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。|
|min_impurity_split|信息增益的阀值|决策树在创建分支时，信息增益必须大于这个阀值，否则不分裂|
|min_weight_fraction_leaf|叶子节点最小的样本权重和|float, optional (default=0.) 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。|
|class_weight|类别权重|dict, list of dicts, “balanced” or None, default=None 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重，或者用“balanced”，如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个参数，选择默认的”None” 不适用于回归树 sklearn.tree.DecisionTreeRegressor|

### 模型调参注意事项：
- 当样本少数量但是样本特征非常多的时候，决策树很容易过拟合，一般来说，样本数比特征数多一些会比较容易建立健壮的模型
- 如果样本数量少但是样本特征非常多，在拟合决策树模型前，推荐先做维度规约，比如主成分分析（PCA），特征选择（Losso）或者独立成分分析（ICA）。这样特征的维度会大大减小。再来拟合决策树模型效果会好。
- 推荐多用决策树的可视化，同时先限制决策树的深度（比如最多3层），这样可以先观察下生成的决策树里数据的初步拟合情况，然后再决定是否要增加深度。
- 在训练模型先，注意观察样本的类别情况（主要指分类树），如果类别分布非常不均匀，就要考虑用class_weight来限制模型过于偏向样本多的类别。
- 决策树的数组使用的是numpy的float32类型，如果训练数据不是这样的格式，算法会先做copy再运行。
- 如果输入的样本矩阵是稀疏的，推荐在拟合前调用csc_matrix稀疏化，在预测前调用csr_matrix稀疏化。

## 实例：预测泰坦尼克号幸存者
数据预处理前期工作：

- 筛选特征值，丢掉不需要的特征数据
- 对性别进行二值化处理（转换为0和1）
- 港口转换成数值型数据
- 处理缺失值（如年龄，有很多缺失值）

1、首先读取数据


```python
import pandas as pd
import numpy as np

def read_dataset(fname):
#     指定第一列作为行索引
    data = pd.read_csv(fname, index_col=0)
#     丢弃无用数据
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
#     处理性别数据
    lables = data['Sex'].unique().tolist()
    data['Sex'] = [*map(lambda x: lables.index(x) , data['Sex'])]
#     处理登船港口数据
    lables = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: lables.index(n))
#     处理缺失数据填充0
    data = data.fillna(0)
    return data
train = read_dataset('code/datasets/titanic/train.csv')
```

2、拆分数据集

把`Survived`列提取出来作为标签，然后在元数据集中将其丢弃。同时拆分数据集和交叉验证数据集


```python
from sklearn.model_selection import train_test_split

y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train_shape:", X_train.shape, " y_train_shape:", y_train.shape)
print("X_test_shape:", X_test.shape,"  y_test_shape:", y_test.shape)
```

    X_train_shape: (712, 7)  y_train_shape: (712,)
    X_test_shape: (179, 7)   y_test_shape: (179,)


3、拟合数据集


```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("train score:", clf.score(X_train, y_train))
print("test score:", clf.score(X_test, y_test))
```

    train score: 0.9845505617977528
    test score: 0.7597765363128491


## 优化模型参数
### 1、通过`max_depth`参数来优化模型

从以上输出数据可以看出，针对训练样本评分很高，但针对测试数据集评分较低。很明显这是过拟合的特征。解决决策树过拟合的方法是剪枝，包括前剪枝和后剪枝。但是`sklearn`不支持后剪枝，这里通过`max_depth`参数限定决策树深度，在一定程度上避免过拟合。

这里先创建一个函数使用不同的模型深度训练模型，并计算评分数据。


```python
def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    return(clf.score(X_train, y_train), clf.score(X_test, y_test))
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
depths = np.arange(1,10)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]

# 找出交叉验证数据集评分最高的索引
tr_best_index = np.argmax(tr_scores)
te_best_index = np.argmax(te_scores)

print("bestdepth:", te_best_index+1, " bestdepth_score:", te_scores[te_best_index], '\n')
```

    bestdepth: 5  bestdepth_score: 0.8603351955307262 
    


**这里由于以上`train_test_split`方法对数据切分是随机打散的，造成每次用不同的数据集训练模型总得到不同的最佳深度。**这里写个循环反复测试，最终验证这里看到最佳的分支深度为5出现的频率最高，初步确定5为深度模型最佳。

把模型参数和对应的评分画出来：


```python
%matplotlib inline
from matplotlib import pyplot as plt
depths = np.arange(1,10)
plt.figure(figsize=(6,4), dpi=120)
plt.grid()
plt.xlabel('max depth of decison tree')
plt.ylabel('Scores')
plt.plot(depths, te_scores, label='test_scores')
plt.plot(depths, tr_scores, label='train_scores')
plt.legend()
```

![](https://ws4.sinaimg.cn/large/006tNc79ly1fq3frlhc9vj30ii0c6jsa.jpg)

### 2、通过`min_impurity_decrease`来优化模型

这个参数用来指定信息墒或者基尼不纯度的阀值，当决策树分裂后，其信息增益低于这个阀值时则不再分裂。


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def minsplit_score(val):
    clf = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=val)
    clf.fit(X_train, y_train)
    return (clf.score(X_train, y_train), clf.score(X_test, y_test), )

# 指定参数范围，分别训练模型并计算得分

vals = np.linspace(0, 0.2, 100)
scores = [minsplit_score(v) for v in vals]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]

bestmin_index = np.argmax(te_scores)
bestscore = te_scores[bestmin_index]
print("bestmin:", vals[bestmin_index])
print("bestscore:", bestscore)

plt.figure(figsize=(6,4), dpi=120)
plt.grid()
plt.xlabel("min_impurity_decrease")
plt.ylabel("Scores")
plt.plot(vals, te_scores, label='test_scores')
plt.plot(vals, tr_scores, label='train_scores')

plt.legend()
```

    bestmin: 0.00202020202020202
    bestscore: 0.7988826815642458


![](https://ws4.sinaimg.cn/large/006tNc79ly1fq3frwqn2fj30hx0c7aac.jpg)

**问题：每次使用不同随机切割的数据集得出最佳参数为0.002很接近0，该怎么解读？**

值此为我们找到了两个参数,最佳深度depth=5 和最佳min_impurity_decrease=0.002，下面我来用两个参数简历模型进行测试：


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn import metrics 

model = DecisionTreeClassifier(max_depth=5, min_impurity_decrease=0.002)
model.fit(X_train, y_train)

print("tees_score:", model.score(X_test, y_test))

y_pred = model.predict(X_test)

print("查准率:",metrics.precision_score(y_test, y_pred))
print("召回率:",metrics.recall_score(y_test, y_pred))
print("F1_score:",metrics.f1_score(y_test, y_pred))
```

    tees_score: 0.7821229050279329
    查准率: 0.8461538461538461
    召回率: 0.5866666666666667
    F1_score: 0.6929133858267718


## 模型参数选择工具包
至此发现以上两种模型优化方法有两问题：
- 1、数据不稳定：--> 每次重新分配训练集测试集，原参数就不是最优了。 解决办法是多次计算求平均值。

- 2、不能一次选择多个参数：--> 想考察max_depth和min_impurity_decrease两者结合起来的最优参数就没法实现。

所幸`scikit-learn`在`sklearn.model_selection`包提供了大量的模型选择和评估的工具供我们使用。针对该问题可以使用`GridSearchCV`类来解决。

### 利用`GridSearchCV`求最优参数


```python
from sklearn.model_selection import GridSearchCV

thresholds = np.linspace(0, 0.2, 50)
param_grid = {'min_impurity_decrease':thresholds}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(X,y)

print("best_parms:{0}\nbest_score:{1}".format(clf.best_params_, clf.best_score_))
```

    best_parms:{'min_impurity_decrease': 0.00816326530612245}
    best_score:0.8114478114478114


模型解读：
1、关键字参数`param_grid`是一个字典，字典的关键字对应的值是一个列表。`GridSearchCV`会枚举列表里所有值来构建模型多次计算训练模型，并计算模型评分，最终得出指定参数值的平均评分及标准差。

2、关键参数`sv`，用来指定交叉验证数据集的生成规则。这里sv=5表示每次计算都把数据集分成5份，拿其中一份作为交叉验证数据集，其他作为训练集。最终得出最优参数及最优评分保存在`clf.best_params_`和`clf.best_score_`里。

3、此外`clf.cv_results_`里保存了计算过程的所有中间结果。

### 画出学习曲线：


```python
def plot_curve(train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.figure(figsize=(6, 4), dpi=120)
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, 
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, 
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '.--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '.-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
```


```python
from sklearn.model_selection import GridSearchCV

thresholds = np.linspace(0, 0.2, 50)
# Set the parameters by cross-validation
param_grid = {'min_impurity_decrease': thresholds}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(X, y)
print("best param: {0}\nbest score: {1}".format(clf.best_params_, 
                                                clf.best_score_))

# plot_curve(thresholds, clf.cv_results_, xlabel='gini thresholds')
```

    best param: {'min_impurity_decrease': 0.00816326530612245}
    best score: 0.8114478114478114


![](https://ws1.sinaimg.cn/large/006tNc79ly1fq3fs68mvsj30i70cr3zb.jpg)

### 多组参数之间选择最优参数：


```python
from sklearn.model_selection import GridSearchCV

entropy_thresholds = np.linspace(0, 1, 100)
gini_thresholds = np.linspace(0, 0.2, 100)
#设置参数矩阵：
param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
              {'max_depth': np.arange(2,10)},
              {'min_samples_split': np.arange(2,30,2)}]
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(X, y)
print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))
```

    best param:{'min_impurity_decrease': 0.00816326530612245}
    best score:0.8114478114478114


`结果1、{'criterion': 'gini', 'min_impurity_decrease': 0.00816326530612245} ->6`

`结果2、{'min_samples_split': 22} ->10`

`结果3、{'min_samples_split': 20} ->4`

**结果波动很大，这里做了20次测试，对应结果1出现6次，结果2出现10次，结果3出现4次。**

**代码解读**：
关键部分还是`param_grid`参数，他是一个列表。很对列表的第一个字典，选择信息墒`（entropy）`作为判断标准，取值0～1范围50等分；

第二个字典选择基尼系数，`min_impurity_decrease`取值0～0.2范围50等分。

`GridSearchCV`会针对列表中的每个字典进行迭代，最终比较列表中每个字典所对应的参数组合，选择出最优的参数。

### 生成决策树图形
下面代码可以生成.dot文件，需要电脑上安装`graphviz`才能把文件转换成图片格式。

`Mac`上可以使用`brew install graphviz`命令来安装，它会同时安装8个依赖包。这里一定注意`Mac`环境下的权限问题：由于`Homebrew`默认是安装在`/usr/local`下，而`Mac`有强制保护不支持`sudo chown -R uname local`对`local`文件夹进行权限修改。 

这里的解决方式是把`local`下`bin`,`lib`,`Cellar`等所需单个文件夹下进行赋权，即可成功安装。

1. 在电脑上安装 graphviz
2. 运行 `dot -Tpng tree.dot -o filename.png`
3. 在当前目录查看生成的决策树 filename.png

```python
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

clf = DecisionTreeClassifier(min_samples_split=22)
clf = clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))

# 导出 titanic.dot 文件
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
```

    train score: 0.8834269662921348; test score: 0.8268156424581006


![](https://ws1.sinaimg.cn/large/006tNc79ly1fq3fe05mm1j31kw0zfgqd.jpg)
