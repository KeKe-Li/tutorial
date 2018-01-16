### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！
AI的开发离不开算法那我们就接下来开始学习算法吧！
回归方法是对数值型连续随机变量进行预测和建模的监督学习算法。其特点是标注的数据集具有数值型的目标变量。回归的目的是预测数值型的目标值。


常用的回归方法包括:
* 线性回归：使用超平面拟合数据集 
* 最近邻算法：通过搜寻最相似的训练样本来预测新样本的值
* 决策树和回归树：将数据集分割为不同分支而实现分层学习
* 集成方法：组合多个弱学习算法构造一种强学习算法，如随机森林（RF）和梯度提升树（GBM）等
* 深度学习：使用多层神经网络学习复杂模型

#### 线性回归

线性回归是最简单的回归方法，它的目标是使用超平面拟合数据集，即学习一个线性模型以尽可能准确的预测实值输出标记。
线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法，运用十分广泛。其表达形式为y = w'x+e，e为误差服从均值为0的正态分布。
回归分析中，只包括一个自变量和一个因变量，且二者的关系可用一条直线近似表示，这种回归分析称为一元线性回归分析。如果回归分析中包括两个或两个以上的自变量，且因变量和自变量之间是线性关系，则称为多元线性回归分析。

#### 单变量模型
模型
<p>
  <span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><msup><mi>w</mi><mi>T</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi></mrow><annotation encoding="application/x-tex">f(x)=w^Tx+b</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.8413309999999999em;"></span><span class="strut bottom" style="height:1.0913309999999998em;vertical-align:-0.25em;"></span><span class="base textstyle uncramped"><span class="mord mathit" style="margin-right:0.10764em;">f</span><span class="mopen">(</span><span class="mord mathit">x</span><span class="mclose">)</span><span class="mrel">=</span><span class="mord"><span class="mord mathit" style="margin-right:0.02691em;">w</span><span class="msupsub"><span class="vlist"><span style="top:-0.363em;margin-right:0.05em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle uncramped mtight"><span class="mord mathit mtight" style="margin-right:0.13889em;">T</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span></span><span class="mord mathit">x</span><span class="mbin">+</span><span class="mord mathit">b</span></span></span></span>
</p>

线性模型(linear model)
简单, 易于建模, 但却蕴含着机器学习的重要思想.由于w直观地表达了各属性在预测中的重要性, 所以线性模型有着很好的可解释性(comprehensibility).
目标函数（最小二乘参数估计）

