### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！

机器学习主要有三种方式：监督学习，无监督学习与半监督学习。

（1）监督学习：从给定的训练数据集中学习出一个函数，当新的数据输入时，可以根据函数预测相应的结果。监督学习的训练集要求是包括输入和输出，也就是特征和目标。训练集中的目标是有标注的。如今机器学习已固有的监督学习算法有可以进行分类的，例如贝叶斯分类，SVM，ID3，C4.5以及分类决策树，以及现在最火热的人工神经网络，例如BP神经网络，RBF神经网络，Hopfield神经网络、深度信念网络和卷积神经网络等。人工神经网络是模拟人大脑的思考方式来进行分析，在人工神经网络中有显层，隐层以及输出层，而每一层都会有神经元，神经元的状态或开启或关闭，这取决于大数据。同样监督机器学习算法也可以作回归，最常用便是逻辑回归。

（2）无监督学习：与有监督学习相比，无监督学习的训练集的类标号是未知的，并且要学习的类的个数或集合可能事先不知道。常见的无监督学习算法包括聚类和关联，例如K均值法、Apriori算法。

（3）半监督学习：介于监督学习和无监督学习之间,例如EM算法。

如今的机器学习领域主要的研究工作在三个方面进行：1）面向任务的研究，研究和分析改进一组预定任务的执行性能的学习系统；2）认知模型，研究人类学习过程并进行计算模拟；3）理论的分析，从理论的层面探索可能的算法和独立的应用领域算法。

#### 自组织映射(Self-Organizing Map)

自组织映射(Self-Organizing Maps, SOM)算法作为一种聚类和高维可视化的无监督学习算法, 是通过模拟人脑对信 号处理的特点而发展起来的一种人工神经网络。该模型由芬兰赫尔辛基大学教授 Teuvo Kohonen 于 1981 年提出后,现在 已成为应用最广泛的自组织神经网络方法,其中的 WTA(Winner Takes All)竞争机制反映了自组织学习最根本的特征。

自组织映射它的思想很简单，本质上是一种只有输入层--隐藏层的神经网络。隐藏层中的一个节点代表一个需要聚成的类。训练时采用“竞争学习”的方式，每个输入的样例在隐藏层中找到一个和它最匹配的节点，称为它的激活节点，也叫“winning neuron”。 紧接着用随机梯度下降法更新激活节点的参数。同时，和激活节点临近的点也根据它们距离激活节点的远近而适当地更新参数。

所以，SOM的一个特点是，隐藏层的节点是有拓扑关系的。这个拓扑关系需要我们确定，如果想要一维的模型，那么隐藏节点依次连成一条线；如果想要二维的拓扑关系，那么就行成一个平面，如下图所示（也叫Kohonen Network）：
<p align="center">
<img width="500" align="center" src="../../images/353.jpg" />
</p>

既然隐藏层是有拓扑关系的，所以我们也可以说，SOM可以把任意维度的输入离散化到一维或者二维(更高维度的不常见)的离散空间上。 Computation layer里面的节点与Input layer的节点是全连接的。

拓扑关系确定后，开始计算过程，大体分成几个部分：

1. 初始化：每个节点随机初始化自己的参数。每个节点的参数个数与Input的维度相同。

2. 对于每一个输入数据，找到与它最相配的节点。假设输入时D维的， 即 X={x_i, i=1,...,D}，那么判别函数可以为欧几里得距离：
<p align="center">
<img width="300" align="center" src="../../images/354.jpg" />
</p>
3. 找到激活节点I(x)之后，我们也希望更新和它临近的节点。令S_ij表示节点i和j之间的距离，对于I(x)临近的节点，分配给它们一个更新权重：
<p align="center">
<img width="300" align="center" src="../../images/355.jpg" />
</p>
简而言之就是临近的节点根据距离的远近，更新程度要会减弱。

然后就是更新节点的参数了。按照梯度下降法更新：

<p align="center">
<img width="300" align="center" src="../../images/356.jpg" />
</p>

迭代，直到收敛。

与K-Means的比较

同样是无监督的聚类方法，自组织映射与K-Means的不同之处：

* K-Means需要事先定下类的个数，也就是K的值。 自组织映射则不用，隐藏层中的某些节点可以没有任何输入数据属于它。所以，K-Means受初始化的影响要比较大。

* K-means为每个输入数据找到一个最相似的类后，只更新这个类的参数。自组织映射则会更新临近的节点。所以K-mean受noise data的影响比较大，自组织映射的准确性可能会比k-means低（因为也更新了临近节点）。

* 自组织映射的可视化比较好。优雅的拓扑关系图 。

#### 应用示例

```python
import numpy
from sklearn.decomposition import RandomizedPCA


class SOM():
    def __init__(self, x, y):        
        self.map = []
        self.n_neurons = x*y
        self.sigma = x
        self.template = numpy.arange(x*y).reshape(self.n_neurons,1)
        self.alpha = 0.6
        self.alpha_final = 0.1
        self.shape = [x,y]
        self.epoch = 0
        
    def train(self, X, iter, batch_size=1):
        if len(self.map) == 0:
            x,y = self.shape
            # first we initialize the map
            self.map = numpy.zeros((self.n_neurons, len(X[0])))
            
            # then we the pricipal components of the input data
            eigen = RandomizedPCA(10).fit_transform(X.T).T
            
            # then we set different point on the map equal to principal components to force diversification
            self.map[0] = eigen[0]
            self.map[y-1] = eigen[1]
            self.map[(x-1)*y] = eigen[2]
            self.map[x*y - 1] = eigen[3]
            for i in range(4, 10):
                self.map[numpy.random.randint(1, self.n_neurons)] = eigen[i]
                
        self.total = iter
        
        # coefficient of decay for learning rate alpha
        self.alpha_decay = (self.alpha_final/self.alpha)**(1.0/self.total)
        
        # coefficient of decay for gaussian smoothing
        self.sigma_decay = (numpy.sqrt(self.shape[0])/(4*self.sigma))**(1.0/self.total)
        
        samples = numpy.arange(len(X))
        numpy.random.shuffle(samples)
    
        for i in xrange(iter):
            idx = samples[i:i + batch_size]
            self.iterate(X[idx])
    
    def transform(self, X):
        # We simply compute the dot product of the input with the transpose of the map to get the new input vectors
        res = numpy.dot(numpy.exp(X),numpy.exp(self.map.T))/numpy.sum(numpy.exp(self.map), axis=1)
        res = res / (numpy.exp(numpy.max(res)) + 1e-8)
        return res
     
    def iterate(self, vector):  
        x, y = self.shape
        
        delta = self.map - vector
        
        # Euclidian distance of each neurons with the example
        dists = numpy.sum((delta)**2, axis=1).reshape(x,y)
        
        # Best maching unit
        idx = numpy.argmin(dists)
        print "Epoch ", self.epoch, ": ", (idx/x, idx%y), "; Sigma: ", self.sigma, "; alpha: ", self.alpha
        
        # Linearly reducing the width of Gaussian Kernel
        self.sigma = self.sigma*self.sigma_decay
        dist_map = self.template.reshape(x,y)     
        
        # Distance of each neurons in the map from the best matching neuron
        dists = numpy.sqrt((dist_map/x - idx/x)**2 + (numpy.mod(dist_map,x) - idx%y)**2).reshape(self.n_neurons, 1)
        #dists = self.template - idx
        
        # Applying Gaussian smoothing to distances of neurons from best matching neuron
        h = numpy.exp(-(dists/self.sigma)**2)      
         
        # Updating neurons in the map
        self.map -= self.alpha*h*delta
       
        # Decreasing alpha
        self.alpha = self.alpha*self.alpha_decay
        
        self.epoch = self.epoch + 1 

```

```python
        
           
from PIL import Image
from tools import make_tile
import gzip
import cPickle

def load_mnist():   
    f = gzip.open("mnist/mnist.pkl.gz", 'rb')
    train, valid, test = cPickle.load(f)
    f.close()  
    return train[0][:20000],train[1][:20000]
    
def demo():
    # Get data
    X, y = load_mnist()
    cl = SOM(20, 20)
    cl.train(X, 2000)  
    
    # Plotting hidden units
    W = cl.map
    W = make_tile(W, img_shape= (28,28), tile_shape=(20,20))
    img = Image.fromarray(W)
    img.save("som_results.png")
    
    
    # creating new inputs
    X = cl.transform(X)
   
    # we can plot "landscape" 3d to view the map in 3d
    landscape = cl.transform(numpy.ones((1,28**2)))
    return cl.map, X, y,landscape
    
    
if __name__ == '__main__':
    demo()

```
