### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！
AI的开发离不开算法那我们就接下来开始学习算法吧！

#### 支持向量机（SVM）

支持向量机(Support Vector Machine，SVM)是Corinna Cortes和Vapnik在1995年首先提出的，是一种监督式学习的方法，可广泛地应用于统计分类以及回归分析。支持向量机属于一般化线性分类器，这族分类器的特点是他们能够同时最小化经验误差与最大化几何边缘区，因此支持向量机也被称为最大边缘区分类器。

在机器学习中，支持向量机（SVM，还支持矢量网络）是与相关的学习算法有关的监督学习模型，可以分析数据，识别模式，用于分类和回归分析。
支持向量机方法是建立在统计学习理论的VC维理论和结构风险最小原理基础上的，根据有限的样本信息在模型的复杂性（即对特定训练样本的学习精度）和学习能力（即无错误地识别任意样本的能力）之间寻求最佳折中，以求获得最好的推广能力 。


在机器学习中，支持向量机（SVM，还支持矢量网络）是与相关的学习算法有关的监督学习模型，可以分析数据，识别模式，用于分类和回归分析。给定一组训练样本，每个标记为属于两类，一个SVM训练算法建立了一个模型，分配新的实例为一类或其他类，使其成为非概率二元线性分类。一个SVM模型的例子，如在空间中的点，映射，使得所述不同的类别的例子是由一个明显的差距是尽可能宽划分的表示。新的实施例则映射到相同的空间中，并预测基于它们落在所述间隙侧上属于一个类别。
除了进行线性分类，支持向量机可以使用所谓的核技巧，它们的输入隐含映射成高维特征空间中有效地进行非线性分类。


支持向量机将向量映射到一个更高维的空间里，在这个空间里建立有一个最大间隔超平面。在分开数据的超平面的两边建有两个互相平行的超平面。建立方向合适的分隔超平面使两个与之平行的超平面间的距离最大化。其假定为，平行超平面间的距离或差距越大，分类器的总误差越小。
<p align="center">
<img width="530" align="center" src="../../images/219.jpg" />
</p>


#### 应用示例
```python
# Data driven farmer goes to the Rodeo
import numpy as npimport pylab as plfrom sklearn 
import svmfrom sklearn 
import linear_modelfrom sklearn 
import treeimport pandas as pddef 

plot_results_with_hyperplane(clf, clf_name, df, plt_nmbr):
    x_min, x_max = df.x.min() - .5, df.x.max() + .5
    y_min, y_max = df.y.min() - .5, df.y.max() + .5

    # step between points. i.e. [0, 0.02, 0.04, ...]
    step = .02
    # to plot the boundary, we're going to create a matrix of every possible point
    # then label each point as a wolf or cow using our classifier
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # this gets our predictions back into a matrix
    Z = Z.reshape(xx.shape)

    # create a subplot (we're going to have more than 1 plot on a given image)
    pl.subplot(2, 2, plt_nmbr)
    # plot the boundaries
    pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)

    # plot the wolves and cows
    for animal in df.animal.unique():
        pl.scatter(df[df.animal==animal].x,
                   df[df.animal==animal].y,
                   marker=animal,
                   label="cows" if animal=="x" else "wolves",
                   color='black')
    pl.title(clf_name)
    pl.legend(loc="best")data = open("cows_and_wolves.txt").read()
    data = [row.split('\t') for row in data.strip().split('\n')]
    animals = []for y, row in enumerate(data):
    for x, item in enumerate(row):
        # x's are cows, o's are wolves
        if item in ['o', 'x']:
            animals.append([x, y, item])df = pd.DataFrame(animals,
            columns=["x", "y", "animal"])df['animal_type'] = df.animal.apply(lambda x: 0 if x=="x" else 1)
            # train using the x and y position 
            coordiantestrain_cols = ["x", "y"]
            clfs = {
            "SVM": svm.SVC(),
            "Logistic" : linear_model.LogisticRegression(),
            "Decision Tree": tree.DecisionTreeClassifier(),}
            plt_nmbr = for clf_name, clf in clfs.iteritems():
            clf.fit(df[train_cols], df.animal_type)
            plot_results_with_hyperplane(clf, clf_name, df, plt_nmbr)
            plt_nmbr += pl.show()
            }
```
相比于神经网络这样更先进的算法，支持向量机有两大主要优势：更高的速度、用更少的样本（千以内）取得更好的表现。这使得该算法非常适合文本分类问题。
