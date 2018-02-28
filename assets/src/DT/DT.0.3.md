### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！
AI的开发离不开算法那我们就接下来开始学习算法吧！


#### ID3算法(Iterative Dichotomiser 3)

ID3算法是决策树的一种，它是基于奥卡姆剃刀原理的，即用尽量用较少的东西做更多的事。ID3算法,即Iterative Dichotomiser3，迭代二叉树三代，是Ross Quinlan发明的一种决策树算法，这个算法的基础就是上面提到的奥卡姆剃刀原理，越是小型的决策树越优于大的决策树，尽管如此，也不总是生成最小的树型结构，而是一个启发式算法。


在信息论中，期望信息越小，那么信息增益就越大，从而纯度就越高。ID3算法的核心思想就是以信息增益来度量属性的选择，选择分裂后信息增益最大的属性进行分裂。
  
#### 应用案例

```python
using namespace std;
void ReadData() //读入数据
{
    ifstream fin("F:\\data.txt");
    ;i<NUM;i++)
    {
      ;j<;j++)
        {
            fin>>DataTable[i][j];
            cout<<DataTable[i][j]<<"\t";
        }
      cout<<endl;
    }
    fin.close();
}

double ComputLog(double &p) //计算以2为底的log
{
    ||p==)
    ;
    else
    {
        );
        return result;
    }
}

double ComputInfo(double &p) //计算信息熵
{
    //cout<<"The value of p is: "<<p<<endl;
    -p;
    /p;
    /q;
    return (p*ComputLog(m)+q*ComputLog(n));
}

void CountInfoNP(int begin,int end,int &CountP,int &CountN) //搜索的起始位置、终止位置、计数变量
{
    CountP=;
    CountN=;
    for(int i=begin;i<=end;i++)
        ]=="Yes")
            CountP++;
        else
            CountN++;
}

bool CompareData(string &data,int &count,string &result) //判断该属性值是否出现过
{
    ;k<count;k++)
        if(data==DataValueWeight[k].AttriValueName) //如果该值出现过，则将其出现次数加一
            {
                DataValueWeight[k].ValueWeight+=;
                if(result=="Yes")
                    DataValueWeight[k].ValuePWeight+=;
                else
                    DataValueWeight[k].ValueNWeight+=;
                //cout<<"Exist Here"<<endl;
                return false;
            }
    return true; //如果该值没有出现过，则返回真值
}


```
