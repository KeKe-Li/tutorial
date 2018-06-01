### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！

机器学习主要有三种方式：监督学习，无监督学习与半监督学习。

（1）监督学习：从给定的训练数据集中学习出一个函数，当新的数据输入时，可以根据函数预测相应的结果。监督学习的训练集要求是包括输入和输出，也就是特征和目标。训练集中的目标是有标注的。如今机器学习已固有的监督学习算法有可以进行分类的，例如贝叶斯分类，SVM，ID3，C4.5以及分类决策树，以及现在最火热的人工神经网络，例如BP神经网络，RBF神经网络，Hopfield神经网络、深度信念网络和卷积神经网络等。人工神经网络是模拟人大脑的思考方式来进行分析，在人工神经网络中有显层，隐层以及输出层，而每一层都会有神经元，神经元的状态或开启或关闭，这取决于大数据。同样监督机器学习算法也可以作回归，最常用便是逻辑回归。

（2）无监督学习：与有监督学习相比，无监督学习的训练集的类标号是未知的，并且要学习的类的个数或集合可能事先不知道。常见的无监督学习算法包括聚类和关联，例如K均值法、Apriori算法。

（3）半监督学习：介于监督学习和无监督学习之间,例如EM算法。

如今的机器学习领域主要的研究工作在三个方面进行：1）面向任务的研究，研究和分析改进一组预定任务的执行性能的学习系统；2）认知模型，研究人类学习过程并进行计算模拟；3）理论的分析，从理论的层面探索可能的算法和独立的应用领域算法。

#### 脉冲神经网络(Spiking Neural Network)
脉冲神经网络Spiking neuralnetworks (SNNs)是第三代神经网络模型，其模拟神经元更加接近实际，除此之外，把时间信息的影响也考虑其中。思路是这样的，动态神经网络中的神经元不是在每一次迭代传播中都被激活（而在典型的多层感知机网络中却是），而是在它的膜电位达到某一个特定值才被激活。当一个神经元被激活，它会产生一个信号传递给其他神经元，提高或降低其膜电位。

脉冲神经网络，其模拟神经元更加接近实际，除此之外，把时间信息的影响也考虑其中。思路是这样的，动态神经网络中的神经元不是在每一次迭代传播中都被激活（而在典型的多层感知机网络中却是），而是在它的膜电位达到某一个特定值才被激活。当一个神经元被激活，它会产生一个信号传递给其他神经元，提高或降低其膜电位。
在脉冲神经网络中，神经元的当前激活水平（被建模成某种微分方程）通常被认为是当前状态，一个输入脉冲会使当前这个值升高，持续一段时间，然后逐渐衰退。出现了很多编码方式把这些输出脉冲序列解释为一个实际的数字，这些编码方式会同时考虑到脉冲频率和脉冲间隔时间。

借助于神经科学的研究，人们可以精确的建立基于脉冲产生时间神经网络模型。这种新型的神经网络采用脉冲编码(spike coding)，通过获得脉冲发生的精确时间，这种新型的神经网络可以进行获得更多的信息和更强的计算能力。

Alan Lloyd Hodgkin 和 Andrew Huxley在1952年提出了第一个脉冲神经网络模型，这个模型描述了动作电位是怎样产生并传播的。但是，脉冲并不是在神经元之间直接传播的，它需要在突触间隙间交换一种叫“神经递质”的化学物质。这种生物体的复杂性和可变性导致了许多不同的神经元模型的产生。

从信息论的观点来看，找到一种可以解释脉冲，也就是动作电位的模型是个问题。所以，神经科学的一个基本问题就是确定神经元是否通过时间编码来交流。时间编码表明单一的神经元可以取代上百个S型隐藏层节点。


这种神经网络大体上可以和传统的人工神经网络一样被用在信息处理中，而且脉冲神经网络可以对一个虚拟昆虫寻找食物的问题建模，而不需要环境的先验知识。并且，由于它更加接近现实的性能，使它可以用来学习生物神经系统的工作，电生理学的脉冲和脉冲神经网络在电脑上的模拟输出相比，确定了拓扑学和生物神经学的假说的可能性。

在实践中脉冲神经网络和已被证明的理论之间还存在一个主要的不同点。脉冲神经网络已被证明在神经科学系统中有作用，而在工程学中还无建树，一些大规模的神经网络已经被设计来利用脉冲神经网络中发现的脉冲编码，这些网络根据储备池计算的原则，但是现实中，大规模的脉冲神经网络计算由于所需计算资源多而产能小，发展受限，造成了只有很少的大规模脉冲神经网络被用来解决复杂的计算问题，而这些之前都是由第二代神经网络解决的。第二代神经网络模型中难以加入时间，脉冲神经网络（特别当算法定义为离散时间时）相当容易观察其动力学特征。我们很难建立一个具有稳定行为的模型来实现一个特定功能。

#### 应用示例
```python
import chainer, math
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check

class SELU(function.Function):
	def __init__(self, alpha, lam):
		self.alpha = float(alpha)
		self.lam = float(lam)

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() == 1)
		x_type, = in_types
		type_check.expect(x_type.dtype.kind == 'f')

	def forward_cpu(self, x):
		y = x[0].copy()
		neg_indices = x[0] <= 0
		y[neg_indices] = self.alpha * (np.exp(y[neg_indices]) - 1)
		y *= self.lam
		return y,

	def forward_gpu(self, x):
		y = cuda.elementwise(
			'T x, T alpha, T lam', 'T y',
			'y = x > 0 ? (T)(lam * x) : (T)(lam * alpha * (exp(x) - 1))',
			'elu_fwd')(x[0], self.alpha, self.lam)
		return y,

	def backward_cpu(self, x, gy):
		gx = gy[0].copy()
		neg_indices = x[0] <= 0
		gx[neg_indices] *= self.alpha * np.exp(x[0][neg_indices])
		gx *= self.lam
		return gx,

	def backward_gpu(self, x, gy):
		gx = cuda.elementwise(
			'T x, T gy, T alpha, T lam', 'T gx',
			'gx = x > 0 ? (T)(lam * gy) : (T)(lam * gy * alpha * exp(x))',
			'elu_bwd')(
				x[0], gy[0], self.alpha, self.lam)
		return gx,


def selu(x, alpha=1.6732632423543772848170429916717, lam=1.0507009873554804934193349852946):
	return SELU(alpha, lam)(x)

def dropout_selu(x, ratio=0.1, alpha=-1.7580993408473766):
	if chainer.config.train == False:
		return x

	q = 1.0 - ratio

	xp = cuda.get_array_module(*x)
	if xp == np:
		d = np.random.rand(*x[0].shape) >= ratio
	else:
		d = xp.random.rand(*x[0].shape, dtype=np.float32) >= ratio

	a = math.pow(q + alpha ** 2 * q * (1 - q), -0.5)
	b = -a * (1 - q) * alpha

	return a * (x * d + alpha * (1 - d)) + b
```
