### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！
AI的开发离不开算法那我们就接下来开始学习算法吧！

#### 岭回归（Ridge Regression）

岭回归(ridge regression, Tikhonov regularization)是一种专用于共线性数据分析的有偏估计回归方法，实质上是一种改良的最小二乘估计法，通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价获得回归系数更为符合实际、更可靠的回归方法，对病态数据的拟合要强于最小二乘法。

岭回归，又称脊回归、吉洪诺夫正则化（Tikhonov regularization），是对不适定问题（ill-posed problem)进行回归分析时最经常使用的一种正则化方法。
对于有些矩阵，矩阵中某个元素的一个很小的变动，会引起最后计算结果误差很大，这种矩阵称为“病态矩阵”。有些时候不正确的计算方法也会使一个正常的矩阵在运算中表现出病态。对于高斯消去法来说，如果主元（即对角线上的元素）上的元素很小，在计算时就会表现出病态的特征。

回归分析中常用的最小二乘法是一种无偏估计。对于一个适定问题，X通常是列满秩的：
<p align="center">
<img width="50" align="center" src="../../images/160.jpg" />
</p>
采用最小二乘法，定义损失函数为残差的平方，最小化损失函数:
<p align="center">
<img width="70" align="center" src="../../images/161.jpg" />
</p>
也可以采用梯度下降法进行求解优化，也可以采用如下公式进行直接求解：

<p align="center">
<img width="100" align="center" src="../../images/162.jpg" />
</p>

当X不是列满秩时，或者某些列之间的线性相关性比较大时，<img width="40" align="center" src="../../images/164.jpg" />的行列式接近于0，即<img width="40" align="center" src="../../images/164.jpg" />接近于奇异，上述问题变为一个不适定问题，此时，计算<img width="50" align="center" src="../../images/165.jpg" />时误差会很大，传统的最小二乘法缺乏稳定性与可靠性。

因而，我们需要将不适定问题转化为适定问题：我们为上述损失函数加上一个正则化项，变为
<p align="center">
<img width="100" align="center" src="../../images/166.jpg" />
</p>

其中，我们定义<img width="100" align="center" src="../../images/167.jpg" />，于是：
<p align="center">
<img width="100" align="center" src="../../images/168.jpg" />
</p>
