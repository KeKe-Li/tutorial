### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！

机器学习主要有三种方式：监督学习，无监督学习与半监督学习。

（1）监督学习：从给定的训练数据集中学习出一个函数，当新的数据输入时，可以根据函数预测相应的结果。监督学习的训练集要求是包括输入和输出，也就是特征和目标。训练集中的目标是有标注的。如今机器学习已固有的监督学习算法有可以进行分类的，例如贝叶斯分类，SVM，ID3，C4.5以及分类决策树，以及现在最火热的人工神经网络，例如BP神经网络，RBF神经网络，Hopfield神经网络、深度信念网络和卷积神经网络等。人工神经网络是模拟人大脑的思考方式来进行分析，在人工神经网络中有显层，隐层以及输出层，而每一层都会有神经元，神经元的状态或开启或关闭，这取决于大数据。同样监督机器学习算法也可以作回归，最常用便是逻辑回归。

（2）无监督学习：与有监督学习相比，无监督学习的训练集的类标号是未知的，并且要学习的类的个数或集合可能事先不知道。常见的无监督学习算法包括聚类和关联，例如K均值法、Apriori算法。

（3）半监督学习：介于监督学习和无监督学习之间,例如EM算法。

如今的机器学习领域主要的研究工作在三个方面进行：1）面向任务的研究，研究和分析改进一组预定任务的执行性能的学习系统；2）认知模型，研究人类学习过程并进行计算模拟；3）理论的分析，从理论的层面探索可能的算法和独立的应用领域算法。

#### 径向基函数网络(Radial Basis Function Network)

在数学建模领域，径向基函数网络（Radial basis function network，缩写 RBF network）是一种使用径向基函数作为激活函数的人工神经网络。
RBF神经网络是基于人脑的神经元细胞对外界反应的局部性而提出的新颖的、有效的前馈式神经网络，具有良好的局部逼近特性。它的数学理论基础成形于1985年由Powell首先提出的多变量插值的径向基函数，1988年被Broomhead和Lowe应用到神经网络设计领域，最终形成了RBF神经网络。
径向基函数网络的输出是输入的径向基函数和神经元参数的线性组合。径向基函数网络具有多种用途，包括包括函数近似法、时间序列预测、分类和系统控制。他们最早由布鲁姆赫德（Broomhead）和洛维（Lowe）在1988年建立。

RBF神经网络是一种三层前馈神经网络。第一层为输入层，由信号源节点构成，将网络与外界环境连结起来，节点数由输入信号的维数确定；第二层为隐含层（径向基层），其节点由径向基函数构成，实现输入空间到隐层空间的非线性变换；第三层为输出层（线性层），对输入模式做出响应，其节点由隐含层节点给出的基函数的线性组合来计算。

<p align="center">
<img width="500" align="center" src="../../images/331.jpg" />
</p>

径向基神经网络的激活函数采用径向基函数，通常定义为空间任一点到某一中心之间欧氏距离的单调函数。径向基神经网络的激活函数是以输入向量和权值向量之间的距离<img width="60" align="center" src="../../images/332.jpg" /> 为自变量的。径向神经网络的激活函数一般表达式为
<p align="center">
<img width="300" align="center" src="../../images/333.jpg" />
</p>

随着权值和输入向量之间距离的减少，网络输出是递增的，当输入向量和权值向量一致时，神经元输出1。b为阈值，用于调整神经元的灵敏度。利用径向基神经元和线性神经元可以建立广义回归神经网络，该种神经网络适用于函数逼近方面的应用；径向基神经元和竞争神经元可以组件概率神经网络，此种神经网络适用于解决分类问题。输出层和隐含层所完成的任务是不同的，因而它们的学习策略也不相同。输出层是对线性权进行调整，采用的是线性优化策略，因而学习速度较快。而隐函数是对激活函数（格林函数或高斯函数，一般为高斯函数）的参数进行调整，采用的是非线性优化策略，因而学习速度较慢。

尽管RBF网络的输出是隐单元输出的线性加权和，学习速度加快，但并不等于径向基神经网络就可以取代其他前馈网络。这是因为径向神经网络很可能需要比BP网络多得多的隐含层神经元来完成工作。

径向基神经网络中需要求解的参数有三个基函数的中心、方差以及隐含层到输出层的权值。根据径向基函数中心选取方法的不同，RBF网络有多种学习方法。下面介绍自组织选取中心的RBF神经网络学习法。此方法由两个阶段组成：

* 自组织学习阶段，此阶段为无监督学习过程，求解隐含层基函数的中心与方差。

* 监督学习阶段，此阶段求解隐含层到输出层之间的权值。

径向基神经网络中常用的径向基函数是高斯函数，因此径向基神经网络的激活函数可表示为：
<p align="center">
<img width="360" align="center" src="../../images/334.jpg" />
</p>

由此可得，径向基神经网络的结构可得到网络的输出为：

<p align="center">
<img width="360" align="center" src="../../images/335.jpg" />
</p>

其中<img width="30" align="center" src="../../images/336.jpg" />为第p个输入样本。h为隐含层的结点数。

如果d是样本的期望输出值，那么基函数的方差可表示为：
<p align="center">
<img width="300" align="center" src="../../images/337.jpg" />
</p>

基于K-均值聚类方法求取基函数中心c:

* 网络初始化 随机选取h个训练样本作为聚类中心<img width="30" align="center" src="../../images/338.jpg" />
* 将输入的训练样本集合按最近邻规则分组，按照<img width="30" align="center" src="../../images/336.jpg" /> 与中心为<img width="30" align="center" src="../../images/338.jpg" />之间的欧式距离将<img width="30" align="center" src="../../images/336.jpg" />分配到输入样本的各个聚类集合<img width="30" align="center" src="../../images/339.jpg" />之中。
* 重新调整聚类中心 计算各个聚类集合<img width="30" align="center" src="../../images/339.jpg" /> 中训练样本的平均值，即新的聚类中心<img width="30" align="center" src="../../images/338.jpg" />， 如果新的聚类中心不再发生变化，所得到的<img width="30" align="center" src="../../images/338.jpg" />就是RBF神经网络最终的基函数中心，否则返回上一步进行下一轮求解.

求解方差<img width="30" align="center" src="../../images/340.jpg" />:

* 该RBF神经网络的基函数为高斯函数，因此方差<img width="30" align="center" src="../../images/340.jpg" />可由下式求解得出:
<p align="center">
<img width="300" align="center" src="../../images/341.jpg" />
</p>

其中<img width="60" align="center" src="../../images/342.jpg" />是所选取中心之间的最大距离.

计算隐含层和输出层之间的权值：

* 用最小二乘法直接计算得到：
<p align="center">
<img width="500" align="center" src="../../images/343.jpg" />
</p>


#### 应用示例
```python
from __future__ import division

import random
import pylab
import math


SAMPLES = 75
EPOCHS = 100

TESTS = 12
RUNS = 3
MOD = 12


def h(x):
  """Function to approximate: y = 0.5 + 0.4sin(2πx)."""
  # note: pylab.sin can accept a numpy.ndarray, but math.sin cannot
  return 0.5 + 0.4*pylab.sin(pylab.pi*2*x)

def noise(x):
  """Add uniform noise in intervale [-0.1, 0.1]."""
  return x + random.uniform(-0.1, 0.1)

def sample(n):
  """Return sample of n random points uniformly distributed in [0, 1]."""
  a = [random.random() for x in range(n)]
  a.sort()
  return a

def gaussian(radial, x):
  """Return gaussian radial function.
  Args:
    radial: (num, num) of gaussian (base, width^2) pair
    x: num of input
  Returns:
    num of gaussian output
  """
  base, width2 = radial
  power = -1 / width2 / 2 * (x-base)**2
  y = pylab.exp(power)
  return y


def output(radials, weights, x):
  """Return set of linearly combined gaussian functions.
  Args:
    radials: [(num, num) of (base, width^2) pairs
    weights: [num] of radial weights, |weights| -1 = |radials|
    x: num of input
  Returns:
    num of linear combination of radial functions.
  """
  y = 0
  for radial, weight in zip(radials, weights[:-1]):
    y += gaussian(radial, x) * weight
  # add bias
  y += weights[-1]
  return y


def update_weights(eta, weights, radials, x, y, d):
  """Update weight vector.
  Returns:
    [num] of updated weight vector, len = |weights|
  """
  new_weights = []
  err = d-y
  for radial, weight in zip(radials, weights[:-1]):
    w = weight + (eta * err * gaussian(radial, x))
    new_weights.append(w)
  # update bias
  w = weights[-1] + (eta * err)
  new_weights.append(w)
  return new_weights


def k_means(input, k):
  """Return n Gaussian centers computed by K-means algorithm from sample x.
  Args:
    input: [num] of input vector
    k: int number of bases, <= |set(input)|
  Returns:
    [(num, [num])] k-size list of (center, input cluster) pairs.
  """
  # initialize k bases as randomly selected unique elements from input
  bases = random.sample(set(input), k)

  # place all inputs in the first cluster to initialize
  clusters = [ (x, 0) for x in input ]
  updated = True

  while(updated):
    updated=False
    for i in range(0, len(clusters)):
      x, m = clusters[i]
      distances = [(abs(b-x), j) for j, b in enumerate(bases)]
      d, j = min(distances)
      # update to move x to a new base cluster
      if m != j:
        updated = True
        clusters[i] = (x, j)

    # update bases
    if updated:
      base_sums = [ [0,0] for s in range(k)]
      for x, m in clusters:
        base_sums[m][0] += x
        base_sums[m][1] += 1
      # check for divide by zero errors
      new_bases = []
      for s, n in base_sums:
        # avoid rare edge case, <1% @ n=25
        # division by zero: select a new base from input
        if n == 0:
          base = random.sample(set(input), 1)[0]
        else:
          base = s / n
        new_bases.append(base)
      bases = new_bases

  # generate returned value
  response = [ (b, []) for b in bases ]
  for x, m in clusters:
    response[m][1].append(x)
    
  return response
      

def variance_width(k_meaned_x):
  """Return mean, variance pairs computed from k_means(x, k).
  Args:
    k_meaned_x: [(num, [num])] of (base, input cluster) pairs
  Returns:
    [(num, num)] of (center, width^2) pairs.
  """
  response = []
  for base, cluster in k_meaned_x:
    if len(cluster) > 1:
      var = sum([(base-x)**2 for x in cluster]) / len(cluster)
      # this actually produces excellent approximations
      # var = sum([(base-x)**2 for x in cluster])
    else:
      var = None
    response.append((base, var))

  # set |cluster| widths to mean variance of other clusters
  vars = [v for b, v in response if v]
  if len(vars) == 0:
    raise Exception("No variance: cannot compute mean variance")
  else:
    var_mean = sum(vars) / len(vars)

  for i in range(len(response)):
    base, var = response[i]
    if not var:
      response[i] = (base, var_mean)

  return response


def shared_width(k_meaned_x):
  """Return shared gaussian widths computed from k_means(x, k).
  Args:
    k_meaned_x: [(num, [num])] of (base, input cluster) pairs
  Returns:
    [(num, num)] of (center, width^2) pairs.
  """
  assert(len(k_meaned_x) > 1)
  # ignore clusters
  bases = [b for b, cluster in k_meaned_x]
  # compute distances between adjancent bases
  s_bases = bases[:]
  s_bases.sort()
  distances = map(lambda p: abs(p[0]-p[1]), zip(s_bases, s_bases[1:]))
  max_d = max(distances)
  sigma_sq = (max_d / 2**0.5)**2
  # map to outputs 
  response = [(b, sigma_sq) for b in bases]
  return response


def plot_instance(name, x, ideal_y, measured_y, trained_y, new_x, estimated_y):
  """Plot function graph, save to file.
  Effect: saves png file of plot to currect directory.
  NOTE: use local graph variable
  Args:
    name: str of plot name, used in file name like "name.png"
    x: [num] input vector
    ideal_y: [num] ideal output vector
    measured_y: [num] noisy output vector
    trained_y: [num] trained output vector
    new_x: [num] new input sample not used in training
    estimated_y: [num] estimated output from trained RBN
  """
  # plot graph
  pylab.rc('text', usetex=True)
  pylab.rc('font', family='serif')
  pylab.xlabel('$x$')
  pylab.ylabel('$y = 0.5 + 0.4\sin(2 \pi x)$')
  pylab.title('RBF Network: %s' % name)
  pylab.plot(x, ideal_y, 'g', label="Ideal")
  pylab.plot(x, measured_y, 'bo', label="Measured")
  pylab.plot(x, trained_y, 'y', label="Trained")
  pylab.plot(new_x, estimated_y, 'r', label="Generalized")
  pylab.legend()
  #  pylab.grid(True)
  filename = name
  filename = filename.replace(' ', '_').replace('\\', '').replace('$', '')
  filename = filename.replace(',', '')
  # save figure
  pylab.savefig("%s.png" % filename)
  # clear this figure
  # note: use http://matplotlib.sourceforge.net/users/artists.html#artist-tutorial
  #  in the future
  pylab.clf()
  pylab.cla()

  
def error(actual, expected):
  """Return error from actual to expected.
  Args
    actual: [num] of sampled output
    expected: [num] of expected ouput, ||expected|| = ||actual||
  Returns:
    num of average distance between actual and expected
  """
  sum_d = 0
  for a, e in zip(actual, expected):
    sum_d += abs(a-e)
  err = sum_d / len(expected)
  return err


def run_test(eta, k, tests=TESTS, runs=RUNS, f_width=variance_width, graph_mod=MOD):
  """Run an RBF training test set; plot, return errors from results.
  Args:
    eta: num of training rate
    k: num of bases
    tests: num of sample set iterations
    runs: num of network generation iterations
    f_width: function to generate radial widths
    graph_mod: num of after how many iterations to plot a graph
  Returns:
    {str: [num]} such that n = (tests*runs) and:
      "sample_err": [num] of n sampling errors
      "train_err": [num] of n training errors
      "gen_err": [num] of n estimation errors
  """
  
  results = {
    "sample_err": [],
    "train_err": [],
    "gen_err": [],
    }

  f_name = f_width.__name__.capitalize().split('_')[0]
  for test in range(1,tests+1):

    print "## K=%d, eta=%.2f, Test=%d" % (k, eta, test)

    # compute input samples
    input = sample(SAMPLES)
    test_input = sample(SAMPLES)
    # compute desired and ideal outputs
    ideal_y = map(h, input)
    test_ideal_y = map(h, test_input)
    measured_y = map(noise, ideal_y)

    # estimate each sample three times
    for run in range(1,runs+1):
      # initialize K radials
      radials = f_width(k_means(input, k))
      # k+1 weights, last weight is bias
      weights = [random.uniform(-0.5, 0.5) for x in range(k+1)]
      # train all epochs
      for i in range(EPOCHS):
        # train one epoch
        for x, d in zip(input, measured_y):
          y = output(radials, weights, x)
          weights = update_weights(eta, weights, radials, x, y, d)

      # examine results
      trained_y = map(lambda x: output(radials, weights, x), input)
      estimated_y = map(lambda x: output(radials, weights, x), test_input)
      sample_err = error(measured_y, ideal_y)
      train_err = error(trained_y, measured_y)
      gen_err = error(estimated_y, test_ideal_y)
      
      # save results
      results["sample_err"].append(sample_err)
      results["train_err"].append(train_err)
      results["gen_err"].append(gen_err)

#      print "Run: %d, Sample: %.4f, Train: %.4f, General: %.4f" \
#        % (run, sample_err, train_err, gen_err)

      # graph some set of results
      iteration = (test-1)*runs + run
      if (iteration % graph_mod) == 0:
#        print "Graphing Test=%d, Run=%d" % (test, run)
        name = "%s $K=%d, \eta =%.2f, E=%.3f$ (%d-%d)" % \
          (f_name, k, eta, gen_err, test, run)
        plot_instance( \
          name, input, ideal_y, measured_y, trained_y, test_input, estimated_y)
  return results


def stats(values):
  """Return tuple of common statistical measures.
  Returns:
    (num, num, num, num) as (mean, std, min, max)
   """
  mean = sum(values) / len(values)
  sum_sqs = reduce(lambda x, y: x + y*y, values)
  var = sum([(mean-x)**2 for x in values]) / len(values)
  var = (sum_sqs - len(values)*mean**2) / len(values)
  std = var**0.5
  min_var, max_var = min(values), max(values)
  return (mean, std, min_var, max_var)


def main():
  random.seed()

  # need final report
  for f_width in (variance_width, shared_width):
    for eta in (0.01, 0.02):
      for k in (5, 10, 15, 20, 25):
      
        print ""
        print "BEGIN PARAMETER TEST SUITE"
        print "K=%d, eta=%.2f, f_width=%s, Tests=%d, Runs=%d" % \
          (k, eta, f_width.__name__, TESTS, RUNS)
        print "+++++++++++++++++++++++++++++++++++"
        r = run_test(k=k, eta=eta, f_width=f_width)
        print "+++++++++++++++++++++++++++++++++++"
        print "RESULTS"
        print "K=%d, eta=%.2f, f_width=%s, Tests=%d, Runs=%d" % \
          (k, eta, f_width.__name__, TESTS, RUNS)
        for name, values in r.items():
          print name
          print "mean=%.4f, std=%.4f, min=%.4f, max=%.4f" % \
            stats(values)
        print "+++++++++++++++++++++++++++++++++++"
               

if __name__ == "__main__":
  main()
```
