### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！

机器学习主要有三种方式：监督学习，无监督学习与半监督学习。

（1）监督学习：从给定的训练数据集中学习出一个函数，当新的数据输入时，可以根据函数预测相应的结果。监督学习的训练集要求是包括输入和输出，也就是特征和目标。训练集中的目标是有标注的。如今机器学习已固有的监督学习算法有可以进行分类的，例如贝叶斯分类，SVM，ID3，C4.5以及分类决策树，以及现在最火热的人工神经网络，例如BP神经网络，RBF神经网络，Hopfield神经网络、深度信念网络和卷积神经网络等。人工神经网络是模拟人大脑的思考方式来进行分析，在人工神经网络中有显层，隐层以及输出层，而每一层都会有神经元，神经元的状态或开启或关闭，这取决于大数据。同样监督机器学习算法也可以作回归，最常用便是逻辑回归。

（2）无监督学习：与有监督学习相比，无监督学习的训练集的类标号是未知的，并且要学习的类的个数或集合可能事先不知道。常见的无监督学习算法包括聚类和关联，例如K均值法、Apriori算法。

（3）半监督学习：介于监督学习和无监督学习之间,例如EM算法。

如今的机器学习领域主要的研究工作在三个方面进行：1）面向任务的研究，研究和分析改进一组预定任务的执行性能的学习系统；2）认知模型，研究人类学习过程并进行计算模拟；3）理论的分析，从理论的层面探索可能的算法和独立的应用领域算法。

#### 多层感知器(Multilayer Perceptron)

多层感知器（Multilayer Perceptron,缩写MLP）是一种前向结构的人工神经网络，映射一组输入向量到一组输出向量。MLP可以被看作是一个有向图，由多个的节点层所组成，每一层都全连接到下一层。除了输入节点，每个节点都是一个带有非线性激活函数的神经元（或称处理单元）。一种被称为反向传播算法的监督学习方法常被用来训练MLP。MLP是感知器的推广，克服了感知器不能对线性不可分数据进行识别的弱点

若每个神经元的激活函数都是线性函数，那么，任意层数的MLP都可被约简成一个等价的单层感知器。

实际上，MLP本身可以使用任何形式的激活函数，譬如阶梯函数或逻辑乙形函数（logistic sigmoid function），但为了使用反向传播算法进行有效学习，激活函数必须限制为可微函数。由于具有良好可微性，很多S函数，尤其是双曲正切函数（Hyperbolic tangent）及逻辑函数，被采用为激活函数。

通常MLP用来进行学习的反向传播算法，在模式识别的领域中算是标准监督学习算法，并在计算神经学及并行分布式处理领域中，持续成为被研究的课题。MLP已被证明是一种通用的函数近似方法，可以被用来拟合复杂的函数，或解决分类问题。

MLP在80年代的时候曾是相当流行的机器学习方法，拥有广泛的应用场景，譬如语音识别、图像识别、机器翻译等等，但自90年代以来，MLP遇到来自更为简单的支持向量机的强劲竞争。近来，由于深度学习的成功，MLP又重新得到了关注。

通常一个单一隐藏层的多层感知机（或人工神经网络—ANN）可以用图表现为：

<p align="center">
<img width="300" align="center" src="../../images/317.jpg" />
</p>


正式的单一隐藏层的MLP可以表现为：<img width="100" align="center" src="../../images/318.jpg" />，其中D是输入向量x的大小，L是输出向量f(x)的大小，矩阵表现为：<img width="300" align="center" src="../../images/319.jpg" />, b是偏差向量，W是权重矩阵，G和s是激活函数。

向量<img width="260" align="center" src="../../images/320.jpg" />构成隐藏层。<img width="160" align="center" src="../../images/321.jpg" />是连接输入向量和隐藏层的权重矩阵。Wi代表输入单元到第i个隐藏单元的权重。一般选择tanh作为s的激活函数，使用<img width="260" align="center" src="../../images/322.jpg" />或者使用逻辑sigmoid函数，<img width="300" align="center" src="../../images/323.jpg" />。

这里我们使用Tanh因为一般它训练速度更快（有时也有利于解决局部最优）。tanh和sigmoid都是标量到标量函数，但通过点积运算向量和张量自然延伸（将向量分解成元素，生成同样大小的向量）。

输出向量通过以下公式得到<img width="200" align="center" src="../../images/324.jpg" />。

我们此前在使用逻辑回归区分MNIST数字时提到过这一公式。如前，在多类区分中，通过使用softmax作为G的函数，可以获得类成员的概率。

训练一个MLP，我们学习模型所有的参数，这里我们使用随机梯度下降和批处理。要学习的参数为：<img width="200" align="center" src="../../images/325.jpg" />。

梯度<img width="70" align="center" src="../../images/326.jpg" />可以使用反向传播算法获得（连续微分的特殊形式），Theano可以自动计算这一微分过程。

从逻辑回归到多层感知机我们将聚焦单隐藏层的多层感知机。 我们从构建一个单隐藏层的类开始。之后只要在此基础之上加一个逻辑回归层就构建了MLP。

```python
class HiddenLayer(object):  
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,  
                 activation=T.tanh):  
        """ 
        Typical hidden layer of a MLP: units are fully-connected and have 
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out) 
        and the bias vector b is of shape (n_out,). 
 
        NOTE : The nonlinearity used here is tanh 
 
        Hidden unit activation is given by: tanh(dot(input,W) + b) 
 
        :type rng: numpy.random.RandomState 
        :param rng: a random number generator used to initialize weights 
 
        :type input: theano.tensor.dmatrix 
        :param input: a symbolic tensor of shape (n_examples, n_in) 
 
        :type n_in: int 
        :param n_in: dimensionality of input 
 
        :type n_out: int 
        :param n_out: number of hidden units 
 
        :type activation: theano.Op or function 
        :param activation: Non linearity to be applied in the hidden 
                           layer 
        """  
        self.input = input  
```
隐藏层i权重的初始值应当根据激活函数以对称间断的方式取得样本。

对于tanh函数，区间在<img width="270" align="center" src="../../images/327.jpg" />

对于sigmoid函数，区间在<img width="270" align="center" src="../../images/328.jpg" />

这种初始化方式保证了在训练早期，每一个神经元在它的激活函数内操作，信息可以便利的向上（输入到输出）或反向（输出到输入）传播。

#### 应用示例

```python
""" 
This tutorial introduces the multilayer perceptron using Theano. 
 
 A multilayer perceptron is a logistic regressor where 
instead of feeding the input to the logistic regression you insert a 
intermediate layer, called the hidden layer, that has a nonlinear 
activation function (usually tanh or sigmoid) . One can use many such 
hidden layers making the architecture deep. The tutorial will also tackle 
the problem of MNIST digit classification. 
 
.. math:: 
 
    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))), 
 
References: 
 
    - textbooks: "Pattern Recognition and Machine Learning" - 
                 Christopher M. Bishop, section 5 
 
"""  
  
from __future__ import print_function  
  
__docformat__ = 'restructedtext en'  
  
  
import os  
import sys  
import timeit  
  
import numpy  
  
import theano  
import theano.tensor as T  
  
  
from logistic_sgd import LogisticRegression, load_data  
  
  
# start-snippet-1  
class HiddenLayer(object):  
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,  
                 activation=T.tanh):  
        """ 
        Typical hidden layer of a MLP: units are fully-connected and have 
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out) 
        and the bias vector b is of shape (n_out,). 
 
        NOTE : The nonlinearity used here is tanh 
 
        Hidden unit activation is given by: tanh(dot(input,W) + b) 
 
        :type rng: numpy.random.RandomState 
        :param rng: a random number generator used to initialize weights 
 
        :type input: theano.tensor.dmatrix 
        :param input: a symbolic tensor of shape (n_examples, n_in) 
 
        :type n_in: int 
        :param n_in: dimensionality of input 
 
        :type n_out: int 
        :param n_out: number of hidden units 
 
        :type activation: theano.Op or function 
        :param activation: Non linearity to be applied in the hidden 
                           layer 
        """  
        self.input = input  
        # end-snippet-1  
  
        # `W` is initialized with `W_values` which is uniformely sampled  
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))  
        # for tanh activation function  
        # the output of uniform if converted using asarray to dtype  
        # theano.config.floatX so that the code is runable on GPU  
        # Note : optimal initialization of weights is dependent on the  
        #        activation function used (among other things).  
        #        For example, results presented in [Xavier10] suggest that you  
        #        should use 4 times larger initial weights for sigmoid  
        #        compared to tanh  
        #        We have no info for other function, so we use the same as  
        #        tanh.  
        if W is None:  
            W_values = numpy.asarray(  
                rng.uniform(  
                    low=-numpy.sqrt(6. / (n_in + n_out)),  
                    high=numpy.sqrt(6. / (n_in + n_out)),  
                    size=(n_in, n_out)  
                ),  
                dtype=theano.config.floatX  
            )  
            if activation == theano.tensor.nnet.sigmoid:  
                W_values *= 4  
  
            W = theano.shared(value=W_values, name='W', borrow=True)  
  
        if b is None:  
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  
            b = theano.shared(value=b_values, name='b', borrow=True)  
  
        self.W = W  
        self.b = b  
  
        lin_output = T.dot(input, self.W) + self.b  
        self.output = (  
            lin_output if activation is None  
            else activation(lin_output)  
        )  
        # parameters of the model  
        self.params = [self.W, self.b]  
  
  
# start-snippet-2  
class MLP(object):  
    """Multi-Layer Perceptron Class 
 
    A multilayer perceptron is a feedforward artificial neural network model 
    that has one layer or more of hidden units and nonlinear activations. 
    Intermediate layers usually have as activation function tanh or the 
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the 
    top layer is a softmax layer (defined here by a ``LogisticRegression`` 
    class). 
    """  
  
    def __init__(self, rng, input, n_in, n_hidden, n_out):  
        """Initialize the parameters for the multilayer perceptron 
 
        :type rng: numpy.random.RandomState 
        :param rng: a random number generator used to initialize weights 
 
        :type input: theano.tensor.TensorType 
        :param input: symbolic variable that describes the input of the 
        architecture (one minibatch) 
 
        :type n_in: int 
        :param n_in: number of input units, the dimension of the space in 
        which the datapoints lie 
 
        :type n_hidden: int 
        :param n_hidden: number of hidden units 
 
        :type n_out: int 
        :param n_out: number of output units, the dimension of the space in 
        which the labels lie 
 
        """  
  
        # Since we are dealing with a one hidden layer MLP, this will translate  
        # into a HiddenLayer with a tanh activation function connected to the  
        # LogisticRegression layer; the activation function can be replaced by  
        # sigmoid or any other nonlinear function  
        self.hiddenLayer = HiddenLayer(  
            rng=rng,  
            input=input,  
            n_in=n_in,  
            n_out=n_hidden,  
            activation=T.tanh  
        )  
  
        # The logistic regression layer gets as input the hidden units  
        # of the hidden layer  
        self.logRegressionLayer = LogisticRegression(  
            input=self.hiddenLayer.output,  
            n_in=n_hidden,  
            n_out=n_out  
        )  
        # end-snippet-2 start-snippet-3  
        # L1 norm ; one regularization option is to enforce L1 norm to  
        # be small  
        self.L1 = (  
            abs(self.hiddenLayer.W).sum()  
            + abs(self.logRegressionLayer.W).sum()  
        )  
  
        # square of L2 norm ; one regularization option is to enforce  
        # square of L2 norm to be small  
        self.L2_sqr = (  
            (self.hiddenLayer.W ** 2).sum()  
            + (self.logRegressionLayer.W ** 2).sum()  
        )  
  
        # negative log likelihood of the MLP is given by the negative  
        # log likelihood of the output of the model, computed in the  
        # logistic regression layer  
        self.negative_log_likelihood = (  
            self.logRegressionLayer.negative_log_likelihood  
        )  
        # same holds for the function computing the number of errors  
        self.errors = self.logRegressionLayer.errors  
  
        # the parameters of the model are the parameters of the two layer it is  
        # made out of  
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params  
        # end-snippet-3  
  
        # keep track of model input  
        self.input = input  
  
  
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,  
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):  
    """ 
    Demonstrate stochastic gradient descent optimization for a multilayer 
    perceptron 
 
    This is demonstrated on MNIST. 
 
    :type learning_rate: float 
    :param learning_rate: learning rate used (factor for the stochastic 
    gradient 
 
    :type L1_reg: float 
    :param L1_reg: L1-norm's weight when added to the cost (see 
    regularization) 
 
    :type L2_reg: float 
    :param L2_reg: L2-norm's weight when added to the cost (see 
    regularization) 
 
    :type n_epochs: int 
    :param n_epochs: maximal number of epochs to run the optimizer 
 
    :type dataset: string 
    :param dataset: the path of the MNIST dataset file from 
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz 
 
 
   """  
    datasets = load_data(dataset)  
  
    train_set_x, train_set_y = datasets[0]  
    valid_set_x, valid_set_y = datasets[1]  
    test_set_x, test_set_y = datasets[2]  
  
    # compute number of minibatches for training, validation and testing  
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size  
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size  
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size  
  
    ######################  
    # BUILD ACTUAL MODEL #  
    ######################  
    print('... building the model')  
  
    # allocate symbolic variables for the data  
    index = T.lscalar()  # index to a [mini]batch  
    x = T.matrix('x')  # the data is presented as rasterized images  
    y = T.ivector('y')  # the labels are presented as 1D vector of  
                        # [int] labels  
  
    rng = numpy.random.RandomState(1234)  
  
    # construct the MLP class  
    classifier = MLP(  
        rng=rng,  
        input=x,  
        n_in=28 * 28,  
        n_hidden=n_hidden,  
        n_out=10  
    )  
  
    # start-snippet-4  
    # the cost we minimize during training is the negative log likelihood of  
    # the model plus the regularization terms (L1 and L2); cost is expressed  
    # here symbolically  
    cost = (  
        classifier.negative_log_likelihood(y)  
        + L1_reg * classifier.L1  
        + L2_reg * classifier.L2_sqr  
    )  
    # end-snippet-4  
  
    # compiling a Theano function that computes the mistakes that are made  
    # by the model on a minibatch  
    test_model = theano.function(  
        inputs=[index],  
        outputs=classifier.errors(y),  
        givens={  
            x: test_set_x[index * batch_size:(index + 1) * batch_size],  
            y: test_set_y[index * batch_size:(index + 1) * batch_size]  
        }  
    )  
  
    validate_model = theano.function(  
        inputs=[index],  
        outputs=classifier.errors(y),  
        givens={  
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],  
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]  
        }  
    )  
  
    # start-snippet-5  
    # compute the gradient of cost with respect to theta (sorted in params)  
    # the resulting gradients will be stored in a list gparams  
    gparams = [T.grad(cost, param) for param in classifier.params]  
  
    # specify how to update the parameters of the model as a list of  
    # (variable, update expression) pairs  
  
    # given two lists of the same length, A = [a1, a2, a3, a4] and  
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each  
    # element is a pair formed from the two lists :  
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]  
    updates = [  
        (param, param - learning_rate * gparam)  
        for param, gparam in zip(classifier.params, gparams)  
    ]  
  
    # compiling a Theano function `train_model` that returns the cost, but  
    # in the same time updates the parameter of the model based on the rules  
    # defined in `updates`  
    train_model = theano.function(  
        inputs=[index],  
        outputs=cost,  
        updates=updates,  
        givens={  
            x: train_set_x[index * batch_size: (index + 1) * batch_size],  
            y: train_set_y[index * batch_size: (index + 1) * batch_size]  
        }  
    )  
    # end-snippet-5  
  
    ###############  
    # TRAIN MODEL #  
    ###############  
    print('... training')  
  
    # early-stopping parameters  
    patience = 10000  # look as this many examples regardless  
    patience_increase = 2  # wait this much longer when a new best is  
                           # found  
    improvement_threshold = 0.995  # a relative improvement of this much is  
                                   # considered significant  
    validation_frequency = min(n_train_batches, patience // 2)  
                                  # go through this many  
                                  # minibatche before checking the network  
                                  # on the validation set; in this case we  
                                  # check every epoch  
  
    best_validation_loss = numpy.inf  
    best_iter = 0  
    test_score = 0.  
    start_time = timeit.default_timer()  
  
    epoch = 0  
    done_looping = False  
  
    while (epoch < n_epochs) and (not done_looping):  
        epoch = epoch + 1  
        for minibatch_index in range(n_train_batches):  
  
            minibatch_avg_cost = train_model(minibatch_index)  
            # iteration number  
            iter = (epoch - 1) * n_train_batches + minibatch_index  
  
            if (iter + 1) % validation_frequency == 0:  
                # compute zero-one loss on validation set  
                validation_losses = [validate_model(i) for i  
                                     in range(n_valid_batches)]  
                this_validation_loss = numpy.mean(validation_losses)  
  
                print(  
                    'epoch %i, minibatch %i/%i, validation error %f %%' %  
                    (  
                        epoch,  
                        minibatch_index + 1,  
                        n_train_batches,  
                        this_validation_loss * 100.  
                    )  
                )  
  
                # if we got the best validation score until now  
                if this_validation_loss < best_validation_loss:  
                    #improve patience if loss improvement is good enough  
                    if (  
                        this_validation_loss < best_validation_loss *  
                        improvement_threshold  
                    ):  
                        patience = max(patience, iter * patience_increase)  
  
                    best_validation_loss = this_validation_loss  
                    best_iter = iter  
  
                    # test it on the test set  
                    test_losses = [test_model(i) for i  
                                   in range(n_test_batches)]  
                    test_score = numpy.mean(test_losses)  
  
                    print(('     epoch %i, minibatch %i/%i, test error of '  
                           'best model %f %%') %  
                          (epoch, minibatch_index + 1, n_train_batches,  
                           test_score * 100.))  
  
            if patience <= iter:  
                done_looping = True  
                break  
  
    end_time = timeit.default_timer()  
    print(('Optimization complete. Best validation score of %f %% '  
           'obtained at iteration %i, with test performance %f %%') %  
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))  
    print(('The code for file ' +  
           os.path.split(__file__)[1] +  
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)  
  
  
if __name__ == '__main__':  
    test_mlp()  

```
