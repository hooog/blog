---
title: PCA主成分分析+SVM人脸识别准确率97%+
date: 2018-04-10 01:05:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn 
---
<!--more-->

## 加载数据

这里使用的测试数据共包含40位人员照片，每个人10张照片。也可登陆http://www.cl.cam.ac.uk/research/dtg/attarchive/facesataglance.html 查看400张照片的缩略图。


```python
import time 
import logging
from sklearn.datasets import fetch_olivetti_faces

logging.basicConfig(level = logging.INFO, format="%(asctime)s %(message)s") # 这里INFO必须大写

data_home = 'code/datasets/'
logging.info("开始加载数据")
faces = fetch_olivetti_faces(data_home=data_home)
logging.info("加载完成")
```

这里做下简单的解释：

加载的图片保存在faces变量里，sklaern已经把每张照片处理成剪切掉头发部分并且64x64大小且人脸居中显示。在真实生产环境中这一步很重要，否则模型将被大量的噪声干扰（即照片背景，变化的发型等，这些特征都应该排除在输入特征之外）。最后要成功下载数据集还需要安装Python图片图里工具Pillow否则无法对图片解码。下面输出下数据的概要信息：


```python
import  numpy as np

X = faces.data
y = faces.target

targets = np.unique(faces.target)
target_names = np.array(["p%d" % t for t in targets]) #给每个人做标签
n_targets = target_name.shape[0]
n_samples, h, w = faces.images.shape

print('Samples count:{}\nTarget count:{}'.format(n_samples, n_targets))
print('Image size:{}x{}\nData shape:{}'.format(w, h, X.shape))
```

    Samples count:400
    Target count:40
    Image size:64x64
    Data shape:(400, 4096)


由输出可知，共有40人，照片总量400，输入特征(64x64=4096)个。

为了直观观察数据，从每个人物的照片里随机选择一张显示，定义下画图工具：

其中输入参数images是一个二维数据，每一行都是一个图片数据。在加载数据时，fech_ollivetti_faces()函数已经自动做了预处理，图片的每个像素的RBG值都转换成了[0,1]浮点数。因此，画出来的照片也是黑白的。子图片识别领域一般用黑白照片就可以了，减少计算量的同时也更加准确。


```python
%matplotlib inline
from matplotlib import pyplot as plt
def plot_gallery(images, titles, h, w, n_row=2, n_col=5):
#     显示图片阵列：
    plt.figure(figsize=(2*n_col, 2.2*n_row),dpi=140)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.01)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.axis('off')

```


```python
n_row = 2
n_col = 6

sample_images = None
sample_titles = []
for i in range(n_targets):
    people_images = X[y==i]  # 注意这里传入i
    people_sample_index = np.random.randint(0, people_images.shape[0], 1)
    people_sample_image = people_images[people_sample_index, :]
    if sample_images is not None:
        sample_images = np.concatenate((sample_images, people_sample_image), axis=0)
    else:
        sample_images =people_sample_image
    sample_titles.append(target_names[i])   # 这里target_names是在前面生成的标签
    
plot_gallery(sample_images, sample_titles, h, w, n_row, n_col)

#代码中X[y=i]可以选择除特定的所有照片，随机选出来的照片放在sample.images数组对象里，最后调用之前定义的函数把照片画出来。
```

![](https://ws4.sinaimg.cn/large/006tKfTcly1fq6vy8axmgj31bo0gs0xb.jpg)


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```

## 支持向量机第一次尝试

直接食用支持向量机来实现人脸识别：


```python
from time import time
from sklearn.svm import SVC

t = time()
clf = SVC(class_weight='balanced')
clf.fit(X_train, y_train)
print("耗时：{}秒".format(time() - t))
```

    耗时：1.0220119953155518秒


### **1、接着对人脸数据进行预测：使用`confusion_matrix`查看准确性：**


```python
from sklearn.metrics import confusion_matrix

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
print("confusion_matrix:\n")
# np.set_printoptions(threshold=np.nan)
print(cm[:10])
```

    confusion_matrix:
    
    [[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]


上面`np.set_printoptions()`是为了确保cm完整输出，这是因为这个数组是40x40的，默认情况下不会全部输出。??

### **2、使用`classification_report`查看准确性**

但是很明显输出结果效果很差。 因为`confusion_matrix`理想的输出是矩阵的对角线上有数组，其他地方都为0，而且这里很多图片都被预测成索引为12的类别了。我买再来看下`classification_report`的结果：


```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names = target_names)) #这里y_test和y_pred不要颠倒。
```

                 precision    recall  f1-score   support
    
             p0       0.00      0.00      0.00         1
             p1       0.00      0.00      0.00         3
             p2       0.00      0.00      0.00         2
             p3       0.00      0.00      0.00         1
             p4       0.00      0.00      0.00         1
             p5       0.00      0.00      0.00         1
             p6       0.00      0.00      0.00         4
             p7       0.00      0.00      0.00         2
             p8       0.00      0.00      0.00         4
             p9       0.00      0.00      0.00         2
            p10       0.00      0.00      0.00         1
            p11       0.00      0.00      0.00         0
            p12       0.00      0.00      0.00         4
            p13       0.00      0.00      0.00         4
            p14       0.00      0.00      0.00         1
            p15       0.00      0.00      0.00         1
            p16       0.00      0.00      0.00         3
            p17       0.00      0.00      0.00         2
            p18       0.00      0.00      0.00         2
            p19       0.00      0.00      0.00         2
            p20       0.00      0.00      0.00         1
            p21       0.00      0.00      0.00         2
            p22       0.00      0.00      0.00         3
            p23       0.00      0.00      0.00         2
            p24       0.00      0.00      0.00         3
            p25       0.00      0.00      0.00         3
            p26       0.00      0.00      0.00         2
            p27       0.00      0.00      0.00         2
            p28       0.00      0.00      0.00         0
            p29       0.00      0.00      0.00         2
            p30       0.00      0.00      0.00         2
            p31       0.00      0.00      0.00         3
            p32       0.00      0.00      0.00         2
            p33       0.00      0.00      0.00         2
            p34       0.00      0.00      0.00         0
            p35       0.00      0.00      0.00         2
            p36       0.00      0.00      0.00         3
            p37       0.00      0.00      0.00         1
            p38       0.00      0.00      0.00         2
            p39       0.00      0.00      0.00         2
    
    avg / total       0.00      0.00      0.00        80
    


    /Users/hadoop/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/hadoop/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
      'recall', 'true', average, warn_for)


结果不出预料，果然很差。

这是因为：**这里把每个像素都作为一个输入特征来处理，这样那个数据噪声太严重了，模型根本没有办法对训练数据集进行拟合。这里有4096个特征向量，可是数据集大小才400个，比特征个数少了太多，而且分出20%作为测试集。这种情况下根本无法进行准确的训练和预测。**

## 使用PCA来处理数据集

解决上述问题的办法有两种，一个是加大数据样本量（在这里这个不太现实），或者使用PCA给数据降维，值选择前k个最重要的特征。

这里我们根据PCA算法来计算失真程度来确定k值。

在sklearn里，可以从PCA模型的`explained_variance_ratio_`变量里获取经PCA处理后的数据还原率。这是一个数组，所有元素求和即可知道选择的$k$值的数据还原率。随着$k$的增大，数值会无限接近于1。

利用这一特征，可以让$k$取值10～300之间，每个30取一次样。针对这里的情况选择失真度小于5%即可。


```python
from sklearn.decomposition import PCA

print("Exploring explained variance ratio for dataset ...")
candidate_components = range(10, 300, 30)
explained_ratios = []
t = time()
for c in candidate_components:
    pca = PCA(n_components=c)
    X_pca = pca.fit_transform(X)
    explained_ratios.append(np.sum(pca.explained_variance_ratio_))
print('Done in {0:.2f}s'.format(time()-t))
```

    Exploring explained variance ratio for dataset ...
    Done in 2.17s



```python
plt.figure(figsize=(8, 5), dpi=100)
plt.grid()
plt.plot(candidate_components, explained_ratios)
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained variance ratio for PCA')
plt.yticks(np.arange(0.5, 1.05, .05))
plt.xticks(np.arange(0, 300, 20));
```

![](https://ws2.sinaimg.cn/large/006tKfTcly1fq6vyorh25j30jj0ctdh6.jpg)

由上图可知，若要保留95%的数据还原率，$k$值选择120即可。为了更直观的看不同$k$值的区别，这里画出来体验下：


```python
def title_prefix(prefix, title):
    return "{}: {}".format(prefix, title)

n_row = 1
n_col = 5

sample_images = sample_images[0:5]
sample_titles = sample_titles[0:5]

plotting_images = sample_images
plotting_titles = [title_prefix('orig', t) for t in sample_titles]
candidate_components = [120, 75, 37, 19, 8]
for c in candidate_components:
    print("Fitting and projecting on PCA(n_components={}) ...".format(c))
    t = time()
    pca = PCA(n_components=c)
    pca.fit(X)
    X_sample_pca = pca.transform(sample_images)
    X_sample_inv = pca.inverse_transform(X_sample_pca)
    plotting_images = np.concatenate((plotting_images, X_sample_inv), axis=0)
    sample_title_pca = [title_prefix('{}'.format(c), t) for t in sample_titles]
    plotting_titles = np.concatenate((plotting_titles, sample_title_pca), axis=0)
    print("Done in {0:.2f}s".format(time() - t))

print("Plotting sample image with different number of PCA conpoments ...")
plot_gallery(plotting_images, plotting_titles, h, w,
    n_row * (len(candidate_components) + 1), n_col)
```

    Fitting and projecting on PCA(n_components=120) ...
    Done in 0.18s
    Fitting and projecting on PCA(n_components=75) ...
    Done in 0.14s
    Fitting and projecting on PCA(n_components=37) ...
    Done in 0.11s
    Fitting and projecting on PCA(n_components=19) ...
    Done in 0.07s
    Fitting and projecting on PCA(n_components=8) ...
    Done in 0.06s
    Plotting sample image with different number of PCA conpoments ...


![](https://ws1.sinaimg.cn/large/006tKfTcly1fq6vzviawhj31411bntk5.jpg)

接下来选择$k=120$作为`PCA`的参数对数据集和测试集进行特征提取，然后调用`GridSearchCV`选出最优参数


```python
n_components = 120

print("Fitting PCA by using training data ...")
t = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("Done in {0:.2f}s".format(time() - t))

print("Projecting input data for PCA ...")
t = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Done in {0:.2f}s".format(time() - t))
```

    Fitting PCA by using training data ...
    Done in 0.16s
    Projecting input data for PCA ...
    Done in 0.01s



```python
from sklearn.model_selection import GridSearchCV

print("Searching the best parameters for SVC ...")
param_grid = {'C': [1, 5, 10, 50],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=2, n_jobs=4)# 参数n_jobs=4表示启动4个进程
clf = clf.fit(X_train_pca, y_train)
print("Best parameters found by grid search:")
print(clf.best_params_)
```

    Searching the best parameters for SVC ...
    Fitting 3 folds for each of 20 candidates, totalling 60 fits
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] C=1, gamma=0.0005 ...............................................
    [CV] ................................ C=1, gamma=0.0001, total=   0.1s
    [CV] C=1, gamma=0.0005 ...............................................
    [CV] ................................ C=1, gamma=0.0001, total=   0.1s
    [CV] C=1, gamma=0.0005 ...............................................
    [CV] ................................ C=1, gamma=0.0001, total=   0.1s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ................................ C=1, gamma=0.0005, total=   0.1s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ................................ C=1, gamma=0.0005, total=   0.1s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ................................ C=1, gamma=0.0005, total=   0.1s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ................................. C=1, gamma=0.001, total=   0.1s
    [CV] C=5, gamma=0.0001 ...............................................
    [CV] ................................. C=1, gamma=0.001, total=   0.1s
    [CV] C=5, gamma=0.0005 ...............................................
    [CV] ................................. C=1, gamma=0.001, total=   0.1s
    [CV] C=1, gamma=0.005 ................................................
    [CV] .................................. C=1, gamma=0.01, total=   0.1s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ................................ C=5, gamma=0.0001, total=   0.1s
    [CV] C=5, gamma=0.0001 ...............................................
    [CV] ................................ C=5, gamma=0.0005, total=   0.1s
    [CV] C=5, gamma=0.001 ................................................
    [CV] ................................. C=1, gamma=0.005, total=   0.1s
    [CV] C=1, gamma=0.005 ................................................
    [CV] .................................. C=1, gamma=0.01, total=   0.1s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ................................ C=5, gamma=0.0001, total=   0.1s
    [CV] C=5, gamma=0.0005 ...............................................
    [CV] ................................. C=5, gamma=0.001, total=   0.1s
    [CV] C=5, gamma=0.001 ................................................
    [CV] ................................. C=1, gamma=0.005, total=   0.1s
    [CV] C=1, gamma=0.005 ................................................
    [CV] ................................ C=5, gamma=0.0005, total=   0.1s
    [CV] C=5, gamma=0.0005 ...............................................
    [CV] .................................. C=1, gamma=0.01, total=   0.1s
    [CV] C=5, gamma=0.0001 ...............................................
    [CV] ................................. C=5, gamma=0.001, total=   0.1s
    [CV] C=5, gamma=0.001 ................................................
    [CV] ................................. C=1, gamma=0.005, total=   0.1s
    [CV] ................................ C=5, gamma=0.0005, total=   0.1s
    [CV] C=5, gamma=0.005 ................................................
    [CV] C=5, gamma=0.01 .................................................
    [CV] ................................ C=5, gamma=0.0001, total=   0.1s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] ................................. C=5, gamma=0.001, total=   0.1s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ................................. C=5, gamma=0.005, total=   0.1s
    [CV] C=5, gamma=0.005 ................................................
    [CV] .................................. C=5, gamma=0.01, total=   0.1s
    [CV] C=5, gamma=0.01 .................................................
    [CV] ............................... C=10, gamma=0.0001, total=   0.1s
    [CV] C=10, gamma=0.0005 ..............................................
    [CV] ................................ C=10, gamma=0.001, total=   0.1s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ................................. C=5, gamma=0.005, total=   0.1s
    [CV] C=5, gamma=0.005 ................................................
    [CV] ............................... C=10, gamma=0.0005, total=   0.1s
    [CV] C=10, gamma=0.0005 ..............................................
    [CV] .................................. C=5, gamma=0.01, total=   0.1s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] ................................ C=10, gamma=0.001, total=   0.1s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ................................. C=5, gamma=0.005, total=   0.1s
    [CV] C=5, gamma=0.01 .................................................
    [CV] ............................... C=10, gamma=0.0001, total=   0.1s
    [CV] C=10, gamma=0.0001 ..............................................
    [CV] ............................... C=10, gamma=0.0005, total=   0.1s
    [CV] C=10, gamma=0.0005 ..............................................
    [CV] ................................ C=10, gamma=0.001, total=   0.1s
    [CV] C=10, gamma=0.005 ...............................................
    [CV] .................................. C=5, gamma=0.01, total=   0.1s
    [CV] C=10, gamma=0.005 ...............................................
    [CV] ............................... C=10, gamma=0.0001, total=   0.1s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ............................... C=10, gamma=0.0005, total=   0.1s
    [CV] C=50, gamma=0.0005 ..............................................
    [CV] ................................ C=10, gamma=0.005, total=   0.1s
    [CV] C=50, gamma=0.001 ...............................................
    [CV] ................................ C=10, gamma=0.005, total=   0.1s
    [CV] C=10, gamma=0.005 ...............................................
    [CV] ................................. C=10, gamma=0.01, total=   0.1s
    [CV] C=50, gamma=0.0001 ..............................................
    [CV] ............................... C=50, gamma=0.0005, total=   0.1s
    [CV] C=50, gamma=0.0005 ..............................................
    [CV] ................................ C=50, gamma=0.001, total=   0.1s
    [CV] C=50, gamma=0.001 ...............................................
    [CV] ............................... C=50, gamma=0.0001, total=   0.1s
    [CV] ................................ C=10, gamma=0.005, total=   0.1s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ............................... C=50, gamma=0.0005, total=   0.1s
    [CV] C=50, gamma=0.0001 ..............................................
    [CV] C=50, gamma=0.0005 ..............................................
    [CV] ................................ C=50, gamma=0.001, total=   0.1s
    [CV] ................................. C=10, gamma=0.01, total=   0.1s
    [CV] C=10, gamma=0.01 ................................................
    [CV] C=50, gamma=0.005 ...............................................
    [CV] ............................... C=50, gamma=0.0001, total=   0.1s
    [CV] C=50, gamma=0.0001 ..............................................
    [CV] ............................... C=50, gamma=0.0005, total=   0.1s
    [CV] C=50, gamma=0.001 ...............................................
    [CV] ................................. C=10, gamma=0.01, total=   0.1s
    [CV] ................................ C=50, gamma=0.005, total=   0.1s
    [CV] C=50, gamma=0.005 ...............................................
    [CV] C=50, gamma=0.005 ...............................................
    [CV] ............................... C=50, gamma=0.0001, total=   0.1s
    [CV] ................................ C=50, gamma=0.001, total=   0.1s
    [CV] ................................ C=50, gamma=0.005, total=   0.1s
    [CV] ................................ C=50, gamma=0.005, total=   0.1s
    [CV] C=50, gamma=0.01 ................................................
    [CV] ................................. C=50, gamma=0.01, total=   0.0s
    [CV] C=50, gamma=0.01 ................................................
    [CV] ................................. C=50, gamma=0.01, total=   0.0s
    [CV] C=50, gamma=0.01 ................................................
    [CV] ................................. C=50, gamma=0.01, total=   0.0s
    Best parameters found by grid search:
    {'C': 10, 'gamma': 0.0005}


    [Parallel(n_jobs=4)]: Done  60 out of  60 | elapsed:    1.9s finished


接着使用这一模型对测试集进行预测，并分别使用`confusion_matrix`和`classification_report`查看其效果


```python
t = time()
y_pred = clf.best_estimator_.predict(X_test_pca)
cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
print("Done in {0:.2f}.\n".format(time()-t))
print("confusion matrix:")
np.set_printoptions(threshold=np.nan)
print(cm[:10])
```

    Done in 0.01.
    
    confusion matrix:
    [[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]



```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=target_names)) #这里注意y_test和y_pred位置不要颠倒
```

                 precision    recall  f1-score   support
    
             p0       1.00      1.00      1.00         1
             p1       1.00      1.00      1.00         3
             p2       1.00      0.50      0.67         2
             p3       1.00      1.00      1.00         1
             p4       1.00      1.00      1.00         1
             p5       1.00      1.00      1.00         1
             p6       1.00      0.75      0.86         4
             p7       1.00      1.00      1.00         2
             p8       1.00      1.00      1.00         4
             p9       1.00      1.00      1.00         2
            p10       1.00      1.00      1.00         1
            p11       1.00      1.00      1.00         4
            p12       1.00      1.00      1.00         4
            p13       1.00      1.00      1.00         1
            p14       1.00      1.00      1.00         1
            p15       0.75      1.00      0.86         3
            p16       1.00      1.00      1.00         2
            p17       1.00      1.00      1.00         2
            p18       1.00      1.00      1.00         2
            p19       1.00      1.00      1.00         1
            p20       1.00      1.00      1.00         2
            p21       1.00      1.00      1.00         3
            p22       1.00      1.00      1.00         2
            p23       1.00      1.00      1.00         3
            p24       0.75      1.00      0.86         3
            p25       1.00      1.00      1.00         2
            p26       1.00      1.00      1.00         2
            p27       1.00      1.00      1.00         2
            p28       1.00      1.00      1.00         2
            p29       1.00      1.00      1.00         3
            p30       1.00      1.00      1.00         2
            p31       1.00      1.00      1.00         2
            p32       1.00      1.00      1.00         2
            p33       1.00      1.00      1.00         3
            p34       1.00      1.00      1.00         1
            p35       1.00      1.00      1.00         2
            p36       1.00      1.00      1.00         2
    
    avg / total       0.98      0.97      0.97        80
    


    /Users/hadoop/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1428: UserWarning: labels size, 37, does not match size of target_names, 40
      .format(len(labels), len(target_names))


效果非常乐观，但是仍有个问题：

**`怎么确定p0～p37分别对应的是哪个一个人？`**
