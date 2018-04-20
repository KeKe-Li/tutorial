### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！
AI的开发离不开算法那我们就接下来开始学习算法吧！

#### Eclat算法(Eclat Algorithm)
Eclat算法是一种深度优先算法,采用垂直数据表示形式,在概念格理论的基础上利用基于前缀的等价关系将搜索空间(概念格)划分为较小的子空间(子概念格)。

#### Eclat算法原理
* 垂直数据表示

支持度的计算需要访问数据库。在大多数算法中,考虑数据库中事务(即记录)的表示形式。从概念上讲,这样的数据库可以用一个二进制的二维矩阵来表示,矩阵的每一行代表数据库的一条事务,每一列代表项目。
传统的数据挖掘算法多采用水平数据表示,在水平数据表示中,数据库的一条事务由事务标识符(TID)和项目组成。事务由TID唯一标识,一条事务可以包含一个项目或多个项目。Apriori、FP?Growth等算法都是采用此种表示方法。
定义1(Tidset)?设有项目X,包含项目X的所有事务的标识符的集合称为项目X的Tidset。在这种数据表示方法中,数据库的事务由项目和该项目的Tidset组成,该事务由项目唯一标识。Tidset垂直数据表示:数据库中的每一条记录由一个项目及其所出现过的所有事务记录的列表(即Tidset表)构成。这样使得任何一个频繁项集的支持度计数都可以通过对Tidset交集运算求得。

* 支持度计数方法

Eclat算法采用方法二计算支持度。对候选k项集进行支持度计算时,不需再次扫描数据库,仅在一次扫描数据库后得到每个1项集的支持度,而候选k项集的支持度就是在对k-1项集进行交集操作后得到的该k项集Tidset中元素的个数。

* 概念格理论

Eclat算法在概念格理论的基础上,利用基于前缀的等价关系将搜索空间(概念格)划分为较小的子空间(子概念格),各子概念格采用自底向上的搜索方法独立产生频繁项集。

#### eclat算法不足
在Eclat算法中,它由2个集合的并集产生新的候选集,通过计算这2个项集的Tidset的交集快速得到候选集的支持度,因此,当Tidset的规模庞大时将出现以下问题:
* 1.求Tidset的交集的操作将消耗大量时间,影响了算法的效率;
* 2.Tidset的规模相当庞大,消耗系统大量的内存。


#### 应用案例
```python
import sys
import itertools
import math

tidlists = {1: {}};
min_sup = float(sys.argv[1])
min_conf = float(sys.argv[2])

print('Reading dataset.')
dataset = open(sys.argv[3])
tid = 0
for line in dataset:
    line = line.strip()
    if len(line) == 0:
        continue

    items = line.split(' ')

    for item in items:
        item = (item,)
        if item not in tidlists[1]:
            tidlists[1][item] = set()
        tidlists[1][item].add(tid)

    tid += 1
dataset.close()
print('Dataset reading done.')
transactions = tid
min_sup_count = min_sup * transactions

n_items = len(tidlists[1])
print('Number of items: {}.'.format(n_items))

tidlists[1] = {k:v for k,v in tidlists[1].items() if len(v) >= min_sup_count}

n_frequent_items = len(tidlists[1])
print('Number of requent items: {}. Removed {}.'.format(n_frequent_items, n_items - n_frequent_items))

def has_same_prefix(itemset1, itemset2, n):
    for i in range(0, min(len(itemset1), len(itemset2))):
        if n == 0:
            return True

        if itemset1[i] != itemset2[i]:
            return False
        n -= 1
    return n == 0

def eclat():
    k = 2
    while True:
        print('Searching for {}-itemsets.'.format(k))
        tidlists[k] = {}
        combination_counter = 0
        #n_combinations = math.factorial(len(tidlists[k-1]))/(2 * math.factorial(len(tidlists[k-1])-2))
        #print('Possible number of combinations: {}.'.format(n_combinations))
        for itemset1, itemset2 in itertools.combinations(tidlists[k-1].keys(), r=2):
            combination_counter += 1
            if not has_same_prefix(itemset1, itemset2, k-2):
                continue

            tidlist1, tidlist2 = tidlists[k-1][itemset1], tidlists[k-1][itemset2]
            intersection = tidlist1.intersection(tidlist2)
            if len(intersection) < min_sup_count:
                continue

            tidlists[k][tuple(list(itemset1) + [itemset2[len(itemset2)-1]])] = intersection


        if len(tidlists[k]) == 0:
            del tidlists[k]
            break


        print('Number of frequent {}-itemsets: {}.'.format(k, len(tidlists[k])))
        k += 1
        
    out = open(sys.argv[4], 'w')
    # Print results
    out.write('itemsets\n')
    for i in range(1, k):
        #print('Frequent {}-itemsets'.format(i))
        for (itemset, tidlist) in tidlists[i].items():
            out.write(' '.join(itemset1) + '\n')
            out.write(str(len(tidlist)/transactions) + '\n')

    out.write('rules\n')
    for i in range(1, k):
        #print('Frequent {}-itemsets'.format(i))
        for (itemset, tidlist) in tidlists[i].items():
            subsets = []
            for j in range(1, len(itemset)):
                subsets = subsets + list(itertools.combinations(itemset, j))

            for subset in subsets:
                difference = set(itemset).difference(subset)
                confidence = len(tidlist)/len(tidlists[len(subset)][tuple(subset)])
                if confidence < min_conf:
                    continue
                out.write(' '.join(subset) + '\n')
                out.write(' '.join(difference) + '\n')
                out.write(str(confidence) + '\n')

    out.close()
    
eclat()

``
