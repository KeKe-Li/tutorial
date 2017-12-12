### Leeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！
AI的开发离不开算法那我们就接下来开始学习算法吧！

#### 粗糙集理论
粗糙集理论是一种新的处理模糊和不确定性知识的数学工具，是建立在分类机制的基础上的，它将分类理解为在特定空间上的等价关系，而等价关系构成了对该空间的划分。粗糙集理论将知识理解为对数据的划分，每一被划分的集合称为概念。粗糙集理论的主要思想是利用已知的知识库，将不精确或不确定的知识用已知的知识库中的知识来（近似）刻画。其主要思想是在保持分类能力不变的前提下，通过知识约简，导出问题的决策或分类规则。粗糙集分析方法中用到的数据类型为离散型数据，对于连续型数据必须在处理前离散化。


粗糙集理论基本概念
定义1： 一个信息系统是一个四元组，可表示为：

<p align="center">
<img width="100" align="center" src="../../images/30.jpg" />
</p>

 其中U为对象的非空有限集合;A为属性的非空有限集合;V为属性的值域集;f为信息函数,<img width="100" align="center" src="../../images/31.jpg" />.
 如果<img width="100" align="center" src="../../images/33.jpg" />,<img width="100" align="center" src="../../images/32.jpg" />,C为条件属性集,D为决策属性集，则把信息系统<img width="100" align="center" src="../../images/30.jpg" />称为决策系统，用<img width="100" align="center" src="../../images/34.jpg" />或<img width="100" align="center" src="../../images/35.jpg" />来表示，其中d为单一的决策属性。从数据库的角度来看，决策系统就是一张表，其中U是记录集合，A是字段集合，每一个对象对应一条记录，这样决策系统又可称为决策表。
 
定义2：在决策系统<img width="100" align="center" src="../../images/36.jpg" />中，对于<img width="50" align="center" src="../../images/37.jpg" />，则B在U上的不可分辨关系定义为：<img width="300" align="center" src="../../images/38.jpg" />，<img width="50" align="center" src="../../images/39.jpg" />把U划分为k个等价类，<img width="100" align="center" src="../../images/40.jpg" />表示等价关系,<img width="100" align="center" src="../../images/41.jpg" />的所有等价类组成的等价类族，即有:
<img width="100" align="center" src="../../images/42.jpg" />.
