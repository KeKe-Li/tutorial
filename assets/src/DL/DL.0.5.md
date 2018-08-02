### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！
AI的开发离不开算法那我们就接下来开始学习算法吧！

#### 分层时间记忆(Hierarchical Temporal Memory)

分层时间记忆算法(Hierarchical Temporal Memory)，全称HTM Cortical Learning Algorithms是由《人工智能的未来》(On Intelligence)一书作者Jeff Hawkins创建的Numenta公司发表的新一代人工智能算法。HTM算法旨在模拟新大脑皮层的工作原理，将复杂的问题转化为模式匹配与预测。正如它的名字HTM一样，该算法与普通的神经网络算法有诸多的不同之处。HTM强调对“神经元”进行分层级，强调信息模式的空间特性与时间特性。目前Numenta公司已经推出基于HTM算法的python平台和可进行视觉识别的软件工具箱。

传统的人工智能算法大多是针对特定的任务目标而设计的。HTM算法与之不同 ，它注重先把问题转化成模式匹配与预测的问题再解决。这使提出人工智能的“统一理论”成为可能。HTM算法是建立在大量解剖学和神经科学的基础上的。HTM算法认为人类之所以具有智能，新大脑皮层是不可缺少的必要条件，并且由其承担高级脑活动。我们的大脑的运行机制是将接受到的各种模式与记忆中模式进行匹配，并对下一刻将会接收到的信息作出预测及反应，如此循环往复。这正是其时效性(Temporal)的体现。
HTM算法表面上与神经网络算法有相似之处，其实质是完全不同的。这就好比一般电路与门电路的区别。将模拟“神经元”按照新大脑皮层的结构连接之后就会产生与一般神经网路完全不同的效果。一般的神经网络注重前馈，而HTM算法更注重信息的双向交流，这也是由于神经解剖学发现反馈突触数量不亚于前馈的原因。而反馈并不能得到大多数人的重视。

HTM算法也是一种拥有记忆性和可学习性的算法。它相对于其他学习算法更注重对神经网络的训练而不是架构。HTM算法认为只要经过合理的训练，该算法适用于解决大多数人工智能问题。对于不同的任务目标实验者需要将数据先空间化和时间化再对HTM网络进行训练即可。

其实HTM算法是源于生物学理论，意味着它来自神经解剖学和神经生理学，是解释生物学的新皮质如何运作的。我们有时说HTM 理论是“约束于生物学”，和“启发于生物学”相反，后者是机器学习领域常用的说法。HTM 理论必须与新皮质的生物学细节相容，并且不能依赖生物组织中不可能实现的原理。比如，考虑锥体细胞，这是新皮质中最常见的神经元类型。锥体细胞有称作树突的树状延伸，通过数千个突触连接。神经学家知道树突是激活的处理单元，通过突触的交流是动态的、内在随机的过程。锥体细胞是新皮质的核心信息处理元件，突触是记忆的基础。所以，为了理解新皮质是如何运作的，我们需要一种适应神经元和突触的本质特征的理论。人工神经网络（ANN）一般是对没有树突并且只有很少的高度精确的突触和特征的神经元建模，真实的神经元并非如此。这种人工神经元与生物神经元不相容，因此不可能发展成与脑的运作原理相同的网络。这个评论不是说 ANN 没用，只是它们的运作原理和生物神经网络的不同。你将会看到，HTM 理论解释了为什么神经元有数千个突触和激活树突。我们认为，这些和很多其他生物学特征对智能系统来说是必不可少、无法忽视的。


HTM的原理：稀疏分布表征理论,运用的表征方法称作稀疏分布表征，简称 SDR。SDR 是包含数千比特的向量。在任何时候，一小部分比特置 1，其余的置 0。HTM 理论会解释为什么在 SDR 中，总是有一小部分零散的比特置 1，以及这些置 1 的比特的比重必须小，通常小于 2%。SDR 中的比特对应着新皮质中的神经元。稀疏分布表征有一些意义重大并且不可思议的特性。为了方便比较，考虑在可编程计算机中的表征方法。单词存储在计算机中的含义不是单词内在的。如果给你看计算机内存中某处的 64 比特，你并不能知道它表示什么。在程序运行的某一时刻，这些比特可能是表示一个意思，在另一时刻，可能又表示别的意思。无论是哪种情况，只能依赖物理地址，
而非比特本身。对于 SDR，表征的比特编码了自身的语义特征，即表达和含义是一致的。如果两个 SDR 有相同的比特位置 1，它们就共有某种语义特征。如果两个 SDR 有越多相同的比特位置 1，它们在语义上就越相似。SDR 解释了人脑是如何做语义归纳的，它正是这种表征方法的内在特性。另一个展示稀疏表征的特有能力的例子是，一组神经元可以同时激活多种表征并且不引起混乱。这就好比在计算机内存中的某处，不仅仅只能容纳一个值，而是同时容纳二十个值，并且不引起混乱。我们称这种独一无二的特性为“联合性”。在 HTM 理论中它被用来同时做多个预测。稀疏分布表征的运用是 HTM 理论的关键所在。我们认为，所有真正的智能系统必须要运用稀疏分布表征。为了轻松理解 HTM 理论，你需要培养一种面向 SDR 的数学特性和表征特性的直觉。



每个 HTM 系统都需要“感知器官”，我们称之为“编码器”。每种编码器负责把某类数据（数字、事件、温度、图像或者 GPS 坐标）转化成一个稀疏分布表征，以便 HTM 学习算法进一步处理。每种编码器都是为明确的数据类型专门设计的，往往有很多途径可以把输入信息转化为 SDR，就如同哺乳动物们的视网膜构造五花八门。只要感知信息被编码成适当的 SDR，HTM 学习算法就可以处理了。

基于 HTM 理论的机器智能令人兴奋的一个方面是，我们可以创造一些生物学上不存在的具有类似功能的编码器。比如，我们可以创造接收 GPS 坐标的编码器并把数据转化成 SDR。这种编码器允许 HTM 系统直接通过空间位置检测运动。HTM 系统可以进一步对运动分类，预测未来的位置，侦查运动中的异常。能够运用非人类感官的能力启发了智能机器可能的发展方向。智能机器不仅仅在人类事务上表现出优越性，还将处理那些人类难以感知或者无能为力的问题。


每个 HTM 系统都需要“感知器官”，我们称之为“编码器”。每种编码器负责把某类数据（数字、事件、温度、图像或者 GPS 坐标）转化成一个稀疏分布表征，以便 HTM 学习算法进一步处理。每种编码器都是为明确的数据类型专门设计的，往往有很多途径可以把输入信息转化为 SDR，就如同哺乳动物们的视网膜构造五花八门。只要感知信息被编码成适当的 SDR，HTM 学习算法就可以处理了。

基于 HTM 理论的机器智能令人兴奋的一个方面是，我们可以创造一些生物学上不存在的具有类似功能的编码器。比如，我们可以创造接收 GPS 坐标的编码器并把数据转化成 SDR。这种编码器允许 HTM 系统直接通过空间位置检测运动。HTM 系统可以进一步对运动分类，预测未来的位置，侦查运动中的异常。能够运用非人类感官的能力启发了智能机器可能的发展方向。智能机器不仅仅在人类事务上表现出优越性，还将处理那些人类难以感知或者无能为力的问题。

#### 应用示例
```python

from carver.htm.config import config
from carver.htm.synapse import CONNECTED_CUTOFF
from carver.htm.segment import Segment

#one column out of n should fire:
desiredLocalActivity = config.getint('constants','desiredLocalActivity')

def pool_spatial(htm):
    '''
    A couple notable deviations:
    *column overlap boost and cutoff are swapped from pseudocode, details inline
        see _spatial_overlap
    *time and inputData removed from code - used a data producer model, linked to htm 
    *getBestMatchingSegment now takes an argument for whether it is a nextStep segment or a sequence one
        inspired by binarybarry on http://www.numenta.com/phpBB2/viewtopic.php?t=1403
    '''
    
    _spatial_overlap(htm)
    
    activeColumns = _spatial_inhibition(htm)
            
    inhibitionRadius = _spatial_learning(htm, activeColumns)
    
    htm.inhibitionRadius = inhibitionRadius

def pool_temporal(htm, updateSegments, learning=True):
    updateSegments = _temporal_phase1(htm, learning, updateSegments)
            
    updateSegments = _temporal_phase2(htm, updateSegments, learning)
    
    if learning:
        updateSegments = _temporal_phase3(htm, updateSegments)
    
    return updateSegments
    
def _spatial_overlap(htm):
    'Overlap, p 35'
    
    for col in htm.columns:
        col.overlap = len(col.old_firing_synapses())
            
        #The paper has conflicting information in the following lines.
        #The text implies boost before cutoff, the code: cutoff then boost. I 
        #chose boost first because I think the boost should help a column 
        #overcome the cutoff.
        col.overlap *= col.boost
        
        if col.overlap < col.MIN_OVERLAP:
            col.overlap = 0
    
def _spatial_inhibition(htm):
    'Inhibition, p 35'
    activeColumns = []
    for col in htm.columns:
        kthNeighbor = col.kth_neighbor(desiredLocalActivity)
        minLocalActivity = kthNeighbor.overlap
        
        if col.overlap > 0 and col.overlap >= minLocalActivity:
            activeColumns.append(col)
            col.active = True
        else:
            col.active = False
    
    return activeColumns

def _spatial_learning(htm, activeColumns):
    'Learning, p 36'
    for col in activeColumns:
        for s in col.synapses:
            if s.was_firing():
                s.permanence_increment()
            else:
                s.permanence_decrement()
            
    for col in htm.columns:
        col.dutyCycleMin = 0.01 * col.neighbor_duty_cycle_max()
        col.dutyCycleActive = col.get_duty_cycle_active()
        col.boost = col.next_boost()
        
        col.dutyCycleOverlap = col.get_duty_cycle_overlap()
        if col.dutyCycleOverlap < col.dutyCycleMin:
            col.increase_permanences(0.1 * CONNECTED_CUTOFF)
        
    return htm.average_receptive_field_size()

def _temporal_phase1(htm, learning, updateSegments):
    '''
    Phase 1, p40
    @param htm: htm network object
    @param learning: boolean describing whether the network is learning now
    @param updateSegments: hash from cell to a list of segments to update when cell becomes active
    '''
    
    for col in htm.columns_active():
        buPredicted = False
        learningCellChosen = False
        for cell in col.cells:
            if cell.predicted:
                seg = cell.findSegmentWasActive(nextStep=True)
                
                #distal dendrite segments = sequence memory
                if seg and seg.distal:
                    buPredicted = True
                    cell.active = True
                    
                    #Learning Phase 1, p 41
                    if learning and seg.wasActiveFromLearningCells:
                        learningCellChosen = True
                        cell.learning = True
                    
        if not buPredicted:
            for cell in col.cells:
                cell.active = True
                
        #Learning Phase 1, p41
        if learning and not learningCellChosen:
            cell, seg = col.bestCell(nextStep=True)
            cell.learning = True
            
            if seg is None:
                seg = cell.create_segment(htm, nextStep=True)
                
            updateSegments.add(cell, seg, timeDelta=-1)
            
    return updateSegments
            
def _temporal_phase2(htm, updateSegments, learning):
    'Phase 2, p40'
    for cell in htm.cells:
        for seg in cell.segments:
            if seg.active:
                cell.predicting = True
                
                if learning:
                    updateSegments.add(cell, seg, timeDelta=0)
            
        #for each cell, grab the best segment. right now, this does not prevent 
        #duplication of learning on the best segment
        if learning and cell.predicting:
            bestSeg = cell.bestMatchingSegment(nextStep=False)
            if bestSeg is None:
                bestSeg = cell.create_segment(htm, nextStep=False)
                
            bestSeg.round_out_synapses(htm)
            
            updateSegments.add(cell, bestSeg, timeDelta=-1)
    
    return updateSegments

def _temporal_phase3(htm, updateSegments):
    'Phase 3, p42'
    for cell in htm.cells:
        if cell.learning:
            for synapseStates in updateSegments[cell]:
                Segment.adapt_up(synapseStates)
            updateSegments.reset(cell)
        elif not cell.predicting and cell.predicted:
            for synapseStates in updateSegments[cell]:
                Segment.adapt_down(synapseStates)
            updateSegments.reset(cell)
            
    return updateSegments
```
