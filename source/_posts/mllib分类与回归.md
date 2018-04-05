---
title: MLlib分类与回归
date: 2018-03-13 11:59:21
update: 
comments: true
categories:
  - Machine learning
tags:
  - MLlib
---

<!--more-->
本文参考- Machine learning with Scala by CDA 吴昊天

## 一、分类算法概述

​	分类是一种重要的机器学习和数据挖掘技术。分类的目的是根据数据集的特点构造一个分类函数或分类模型(也常常称作分类器)，该模型能把未知类别的样本映射到给定类别中的一种技术。

分类的具体规则可描述如下：给定一组训练数据的集合T(Training set)，T的每一条记录包含若干条属性（Features）组成一个特征向量，用矢量

$x=(x_1,x_2,x_3·····x_n)$

表示。

$x_i$

可以有不同的值域，当一属性的值域为连续域时，该属性为连续属性(Numerical Attribute)，否则为离散属性(Discrete Attribute)。用

$C = （c_1,c_2,c_3·····c_k）$

表示类别属性，即数据集有k个不同的类别。那么，T就隐含了一个从矢量X到类别属性C的映射函数：

$f(X)->$

。分类的目的就是分析输入数据，通过在训练集中的数据表现出来的特性，为每一个类找到一种准确的描述或者模型，采用该种方法(模型)将隐含函数表示出来。

​	构造分类模型的过程一般分为训练和测试两个阶段。在构造模型之前，将数据集随机地分为训练数据集和测试数据集。先使用训练数据集来构造分类模型，然后使用测试数据集来评估模型的分类准确率。如果认为模型的准确率可以接受，就可以用该模型对其它数据元组进分类。一般来说，测试阶段的代价远低于训练阶段。

## 二、spark.mllib分类算法

​	分类算法基于不同的思想，算法也不尽相同，例如支持向量机SVM、决策树算法、贝叶斯算法、KNN算法等。`spark.mllib`包支持各种分类方法，主要包含 [二分类](http://en.wikipedia.org/wiki/Binary_classification)， [多分类](http://en.wikipedia.org/wiki/Multiclass_classification)和 [回归分析](http://en.wikipedia.org/wiki/Regression_analysis)。下表列出了每种类型的问题支持的算法。

![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/12/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7-2016-12-12-%E4%B8%8B%E5%8D%888.13.43.png)

## 三、逻辑斯蒂回归的分类器

### 1.方法简介

 逻辑斯蒂回归（logistic regression）是统计学习中的经典分类方法，属于对数线性模型。logistic回归的因变量可以是二分类的，也可以是多分类的。

### 2.基本原理

#### 2.1 logistic分布

 设X是连续随机变量，X服从logistic分布是指X具有下列分布函数和密度函数：

 

​                                                      ![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9f354d64015b505f718cd9ec4f98890e_l3.svg)

 

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-5e9ba7a182194723ea58568d27bb2301_l3.svg)

 其中，

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-75c83eedd5141ba143a525ba7949c2ae_l3.svg)

为位置参数，

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9df94356151401c13b3565eab38aa533_l3.svg)

为形状参数。

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-f711bdb612e16efa1e06ed33178a159d_l3.svg)

与

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-52866ddba463128c071bba7cfdc18c46_l3.svg)

图像如下，其中分布函数是以

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-4a760e6ab4e5a799ec3cb12f96c044b3_l3.svg)

为中心对阵，

​                                                                                ![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9df94356151401c13b3565eab38aa533_l3.svg)

越小曲线变化越快。

![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/12/6d96cc41gw1etfkt9bbhwj20c603xmx4.jpg)

#### 2.2 二项logistic回归模型：

 二项logistic回归模型如下：

​                                                        ![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-0fbac46d9cb25d3da0c5e0ce2491aa6e_l3.svg)

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-44fa7f0601e3607fabfda93ae964be51_l3.svg)

 其中，

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-477e1cd490f7330c49621f5e744e5038_l3.svg)

是输入，

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-fcb9be84a0978a9b5f986ea8e23c1de4_l3.svg)

是输出，w称为权值向量，b称为偏置，

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-f645be27f6922dcf5e963e9d3d41d4fb_l3.svg)

为w和x的内积。

#### 2.3 参数估计

 假设： 

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-a8f70fd9541936adb9a5dbefb48231e4_l3.svg)

 则采用“极大似然法”来估计w和b。似然函数为:

​                                                         ![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-f5d63ea80a9fd78a9f6a3841b41b955c_l3.svg)

 为方便求解，对其“对数似然”进行估计：

​                                             ![\[L(w) = \sum_{i=1}^N [y_i \log{\pi(x_i)} + (1-y_i) \log{(1 - \pi(x_i)})]\]](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-7b5b46100544dfadefe100d2f5ce9736_l3.svg)

 从而对

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-71661cb137bd34c6ab95ab635582a958_l3.svg)

求极大值，得到

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-356e473b3185b432024c4643855f1b9d_l3.svg)

的估计值。求极值的方法可以是梯度下降法，梯度上升法等。

### 3.基本操作

​	我们仍然以iris数据集为例进行分析。iris以鸢尾花的特征作为数据来源，数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性，是在数据挖掘、数据分类中非常常用的测试集、训练集。

#### 3.1  导入需要的包

```scala
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors,Vector}
```

#### 3.2 读取数据：

​	首先，读取Spark自带数据集，同样，采用LIBSVM格式导入数据

```scala
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
```

 然后，把数据打印看一下：

```scala
data.collect()
```

#### 3.3 构建模型

 接下来，首先进行数据集的划分，这里划分60%的训练集和40%的测试集：

```scala
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)
```

然后，构建逻辑斯蒂模型，用set的方法设置参数，比如说分类的数目，这里可以实现多分类逻辑斯蒂模型：

```scala
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(10)
  .run(training)
```

接下来，调用多分类逻辑斯蒂模型用的predict方法对测试数据进行预测，并把结果保存在MulticlassMetrics中。这里的模型全名为LogisticRegressionWithLBFGS，加上了LBFGS，表示**Limited-memory BFGS**。其中，BFGS是求解非线性优化问题（L(w)求极大值）的方法，是一种秩-2更新，以其发明者Broyden, Fletcher, Goldfarb和Shanno的姓氏首字母命名

```scala
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}
```

这里，采用了test部分的数据每一行都分为标签label和特征features，然后利用map方法，对每一行的数据进行model.predict(features)操作，获得预测值。并把预测值和真正的标签放到predictionAndLabels中。我们可以打印出具体的结果数据来看一下.可以看出，大部分的预测是对的。但第4行的预测与实际标签不同。

#### 3.4 模型评估

 最后，我们把模型预测的准确性打印出来：

```scala
val metrics = new MulticlassMetrics(predictionAndLabels)
val accuracy = metrics.accuracy
println(s"Accuracy = $accuracy")
```

#### 3.5 模型保存与使用

```scala
model.save(sc, "file:///home/hadoop/target/tmp/scalaLogisticRegressionWithLBFGSModel")
val sameModel = LogisticRegressionModel.load(sc,
  "file:///home/hadoop/target/tmp/scalaLogisticRegressionWithLBFGSModel")
```



## 四、决策树分类器

### 1.方法简介

 决策树（decision tree）是一种基本的分类与回归方法，这里主要介绍用于分类的决策树。决策树模式呈树形结构，其中每个内部节点表示一个属性上的测试，每个分支代表一个测试输出，每个叶节点代表一种类别。学习时利用训练数据，根据损失函数最小化的原则建立决策树模型；预测时，对新的数据，利用决策树模型进行分类。

### 2.基本原理

决策树学习通常包括3个步骤：特征选择、决策树的生成和决策树的剪枝。

#### 2.1 特征选择

​	特征选择在于选取对训练数据具有分类能力的特征，这样可以提高决策树学习的效率。通常特征选择的准则是信息增益（或信息增益比、基尼指数等），每次计算每个特征的信息增益，并比较它们的大小，选择信息增益最大（信息增益比最大、基尼指数最小）的特征。下面我们重点介绍一下特征选择的准则：信息增益。

​	首先定义信息论中广泛使用的一个度量标准——熵（entropy），它是表示随机变量不确定性的度量。熵越大，随机变量的不确定性就越大。而信息增益（informational entropy）表示得知某一特征后使得信息的不确定性减少的程度。简单的说，一个属性的信息增益就是由于使用这个属性分割样例而导致的期望熵降低。信息增益、信息增益比和基尼指数的具体定义如下：

**信息增益**：特征A对训练数据集D的信息增益

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-7f455e08dfaae9ec9caba851ba310853_l3.svg)

，定义为集合D的经验熵

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-bc325ed92774ae3c02215c8b3f0df281_l3.svg)

与特征A给定条件下D的经验条件熵

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-5e34864d2036f06dceb0fc344df6ef8d_l3.svg)

之差，即

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-d6a3162e051baab08b5f008e55d3b1b6_l3.svg)

 **信息增益比**：特征A对训练数据集D的信息增益比

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-7cb9a49dc7e23a4812e6dd357f6cce10_l3.svg)

定义为其信息增益

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-7f455e08dfaae9ec9caba851ba310853_l3.svg)

与训练数据集D关于特征A的值的熵

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-8af01987546ded14a9504596fdb86603_l3.svg)

之比，即

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-4f53c71e52f9e202140279fe444b56b8_l3.svg)

其中，

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-283d43a09f3d01239c839119ca540392_l3.svg)

，n是特征A取值的个数。

**基尼指数**：分类问题中，假设有K个类，样本点属于第K类的概率为

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-0a14c8d9009f64ff6011ea57052e19e0_l3.svg)

，则概率分布的基尼指数定义为

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-7b06955be44dc54e541b867eea64b5cf_l3.svg)

#### 2.2 决策树的生成

​	从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子结点，再对子结点递归地调用以上方法，构建决策树；直到所有特征的信息增均很小或没有特征可以选择为止，最后得到一个决策树。

​	决策树需要有停止条件来终止其生长的过程。一般来说最低的条件是：当该节点下面的所有记录都属于同一类，或者当所有的记录属性都具有相同的值时。这两种条件是停止决策树的必要条件，也是最低的条件。在实际运用中一般希望决策树提前停止生长，限定叶节点包含的最低数据量，以防止由于过度生长造成的过拟合问题。

#### 2.3决策树的剪枝

​	决策树生成算法递归地产生决策树，直到不能继续下去为止。这样产生的树往往对训练数据的分类很准确，但对未知的测试数据的分类却没有那么准确，即出现过拟合现象。解决这个问题的办法是考虑决策树的复杂度，对已生成的决策树进行简化，这个过程称为剪枝。

​	决策树的剪枝往往通过极小化决策树整体的损失函数来实现。一般来说，损失函数可以进行如下的定义：

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-8abbaa336774e0ca68d817c7a6e6978e_l3.svg)

 其中，T为任意子树，

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-a7b403272785da327a65e67b3abd260c_l3.svg)

为对训练数据的预测误差（如基尼指数），

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-ead5c2e2af1dbcc7231b994683d5ddc2_l3.svg)

为子树的叶结点个数，

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-095e997ee46e6271f962c1745b8cec82_l3.svg)

为参数，

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-afcbbec24bcbda1b1521842acdf3186d_l3.svg)

为参数是

 ![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9257e6e35d3269c35abbe174de34e5a6_l3.svg)

时的子树T的整体损失，参数

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9257e6e35d3269c35abbe174de34e5a6_l3.svg)

权衡训练数据的拟合程度与模型的复杂度。对于固定的

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9257e6e35d3269c35abbe174de34e5a6_l3.svg)

，一定存在使损失函数

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-afcbbec24bcbda1b1521842acdf3186d_l3.svg)

最小的子树，将其表示为

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9e01c94f75124de404135eab68009560_l3.svg)

。当

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9257e6e35d3269c35abbe174de34e5a6_l3.svg)

大的时候，最优子树

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9e01c94f75124de404135eab68009560_l3.svg)

偏小；当

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9257e6e35d3269c35abbe174de34e5a6_l3.svg)

小的时候，最优子树

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9e01c94f75124de404135eab68009560_l3.svg)

偏大。

### 3.基本操作

​	我们以iris数据集为例进行分析。iris以鸢尾花的特征作为数据来源，数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性，是在数据挖掘、数据分类中非常常用的测试集、训练集。

#### 3.1 导入需要的包：

```scala
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
```

#### 3.2 读取数据：

​	还是先读取数据

```scala
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
```

 然后，我们把数据打印看一下：

```scala
data.foreach { x => println(x) }
```

#### 3.3 构建模型

 接下来，首先进行数据集的划分，这里划分70%的训练集和30%的测试集：

```scala
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
```

然后，调用决策树的trainClassifier方法构建决策树模型，设置参数，比如分类数、信息增益的选择、树的最大深度等：

```scala
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 5
val maxBins = 32

val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  impurity, maxDepth, maxBins)
```

 接下来我们调用决策树模型的predict方法对测试数据集进行预测

```scala
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
```

#### 3.4 模型评估

 最后，我们把模型预测的准确性打印出来：

```scala
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
println("Test Error = " + testErr)
println("Learned classification tree model:\n" + model.toDebugString)
```

#### 3.5 模型保存与使用

```scala
model.save(sc, "file:///home/hadoop/target/tmp/myDecisionTreeClassificationModel")
val sameModel = DecisionTreeModel.load(sc, "file:///home/hadoop/target/tmp/myDecisionTreeClassificationModel")
```



## 五、支持向量机SVM分类器

### 1.方法简介

​	支持向量机SVM是一种二分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器。支持向量机学习方法包含3种模型：线性可分支持向量机、线性支持向量机及非线性支持向量机。当训练数据线性可分时，通过硬间隔最大化，学习一个线性的分类器，即线性可分支持向量机；当训练数据近似线性可分时，通过软间隔最大化，也学习一个线性的分类器，即线性支持向量机；当训练数据线性不可分时，通过使用核技巧及软间隔最大化，学习非线性支持向量机。线性支持向量机支持L1和L2的正则化变型。

### 2.基本原理

​	支持向量机，因其英文名为support vector machine，故一般简称SVM。SVM从线性可分情况下的最优分类面发展而来。最优分类面就是要求分类线不但能将两类正确分开(训练错误率为0)，且使分类间隔最大。SVM考虑寻找一个满足分类要求的超平面，并且使训练集中的点距离分类面尽可能的远，也就是寻找一个分类面使它两侧的空白区域(margin)最大。这两类样本中离分类面最近，且平行于最优分类面的超平面上的点，就叫做支持向量（下图中红色的点）。

![svm](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/12/svm.png)svm

假设超平面可描述为：

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-449556e56e383176183a533644819e9a_l3.svg)

其分类间隔等于

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-8ae60887563a37e93c1666a0426a198e_l3.svg)

。其学习策略是使数据间的间隔最大化，最终可转化为一个凸二次规划问题的求解。

分类器的损失函数（hinge loss铰链损失）如下所示：

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-8d050f8d4bbe5840bcd49262ae5ba918_l3.svg)

默认情况下，线性SVM是用L2 正则化来训练的，但也支持L1正则化。在这种情况下，这个问题就变成了一个线性规划。

 线性SVM算法输出一个SVM模型。给定一个新的数据点，比如说

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-9c2601179e19ecd5f28305e8c9ed83dc_l3.svg)

，这个模型就会根据

![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-d696bea1276912917a52faebe707dd7b_l3.svg)

的值来进行预测。默认情况下，如果

 ![img](http://dblab.xmu.edu.cn/blog/wp-content/ql-cache/quicklatex.com-12f65de211c6a5d7a326102066edc214_l3.svg)

，则输出预测结果为正（因为我们想要损失函数最小，如果预测为负，则会导致损失函数大于1），反之则预测为负。

### 2.基本操作

​	接下来，我们将用SVM对训练数据进行训练，然后用训练得到的模型对测试集进行预测，并计算错误率。仍然以iris数据集为例进行分析。iris以鸢尾花的特征作为数据来源，数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性，是在数据挖掘、数据分类中非常常用的测试集、训练集。

#### 2.1 导入需要的包：

 首先，我们导入需要的包：

```scala
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
```

#### 2.2 读取数据：

​	然后，导入数据

```scala
val data = MLUtils.loadLibSVMFile(sc, "file:///usr/local/spark/data/mllib/sample_libsvm_data.txt")
```

#### 2.3 构建模型

​	因为SVM只支持2分类，所以我们要进行一下数据抽取，这里我们通过filter过滤掉第2类的数据，只选取第0类和第1类的数据。然后，我们把数据集划分成两部分，其中训练集占60%，测试集占40%：

```scala
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)
```

 接下来，通过训练集构建模型SVMWithSGD。这里的SGD即著名的随机梯度下降算法（Stochastic Gradient Descent）。设置迭代次数为1000，除此之外还有stepSize（迭代步伐大小），regParam（regularization正则化控制参数），miniBatchFraction（每次迭代参与计算的样本比例），initialWeights（weight向量初始值）等参数可以进行设置。

```scala
val numIterations = 100
val model = SVMWithSGD.train(training, numIterations)
```

#### 2.4 模型评估

 接下来，我们清除默认阈值，这样会输出原始的预测评分，即带有确信度的结果。

```scala
model.clearThreshold()           // 清空阈值
val scoreAndLabels = test.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}
```

 那如果设置了阈值，则会把大于阈值的结果当成正预测，小于阈值的结果当成负预测。 最后，我们构建评估矩阵，把模型预测的准确性打印出来：

```scala
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()
println("Area under ROC = " + auROC)
```

其中， `SVMWithSGD.train()` 方法默认的通过把正则化参数设为1来执行来范数。如果我们想配置这个算法，可以通过创建一个新的 `SVMWithSGD`对象然后调用他的setter方法来进行重新配置。下面这个例子，我们构建了一个正则化参数为0.1的L1正则化SVM方法 ，然后迭代这个训练算法2000次。

```scala
import org.apache.spark.mllib.optimization.L1Updater
val svmAlg = new SVMWithSGD()
svmAlg.optimizer.
     |       setNumIterations(2000).
     |       setRegParam(0.1).
     |       setUpdater(new L1Updater)
val modelL1 = svmAlg.run(training)
```