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

int SearchData(const int &begin,const int &end,const int &k) //对于第k列进行检索
{
    //cout<<"Enter SearchData() "<<begin<<" "<<end<<" "<<k<<endl;
    ;
    ;i<VALUENUM;i++)
        {
            DataValueWeight[i].ValueWeight=;
            DataValueWeight[i].ValueNWeight=;
            DataValueWeight[i].ValuePWeight=;
        }

    for(int i=begin;i<=end;i++)
        if(i==begin)
           {
             DataValueWeight[count].AttriValueName=DataTable[i][k];
             DataValueWeight[count].ValueWeight+=;
             ]=="Yes")
                DataValueWeight[count].ValuePWeight+=;
             else
                DataValueWeight[count].ValueNWeight+=;

             count++;
           }
        else
        {
            string data=DataTable[i][k];
            ];
            if(CompareData(data,count,result)) //如果该值没有出现过
            {
                DataValueWeight[count].AttriValueName=data;
                DataValueWeight[count].ValueWeight+=;

                ]=="Yes")
                    DataValueWeight[count].ValuePWeight+=;
                else
                    DataValueWeight[count].ValueNWeight+=;
                count++;
            }
        }

     //for(int s=0;s<count;s++)
     // cout<<"Hello: "<<DataValueWeight[s].AttriValueName<<"\t"<<DataValueWeight[s].ValueWeight<<
     // "\t"<<DataValueWeight[s].ValuePWeight<<" \t"<<DataValueWeight[s].ValueNWeight<<endl;

    ;i<count;i++)
    {
        )
            DataValueWeight[i].ValueNWeight=DataValueWeight[i].ValueWeight/DataValueWeight[i].ValueNWeight;
        else
            DataValueWeight[i].ValueNWeight=;

        )
            DataValueWeight[i].ValuePWeight=DataValueWeight[i].ValueWeight/DataValueWeight[i].ValuePWeight;
        else
            DataValueWeight[i].ValuePWeight=;
        //cout<<"N: "<<DataValueWeight[i].ValueNWeight<<" P: "<<DataValueWeight[i].ValuePWeight<<endl;
    }
    return count;
}

int PickAttri()
{
    ;
    int pos;

    ;i<;i++)
    if(InfoResult[i].AttriI>max)
    {
        pos=i;
        max=InfoResult[i].AttriI;
    }
    return pos;
}
int SortByAttriValue(int &begin,int &end,int &temp,int *position)
{

    for(int i=begin;i<=end;i++) //将相应的数据拷贝到另一个阵列
        ;j<=;j++)
        {
            int posy=i-begin;
            CopyDataTable[posy][j]=DataTable[i][j];
        }
//cout<<"have a look"<<endl;

    /*cout<<"*************Show Result First****************"<<endl;
    cout<<InfoResult[temp].AttriName<<endl;
    for(int i=begin;i<=end;i++)
    {
        for(int j=0;j<=5;j++)
            cout<<DataTable[i][j]<<"\t";
        cout<<endl;
    }*/

    ,high=end-begin;
    ;
    ;
    position[]=begin;
    ;i<InfoResult[temp].AttriKind;i++)
    {
        for(int j=low;j<=high;j++)
            if(CopyDataTable[j][temp]==DataValueWeight[i].AttriValueName)
               {
                    int pos=count+begin;

                    ;k<;k++)
                        DataTable[pos][k]=CopyDataTable[j][k];
                    count++;
               }
        position[countpos]=count+begin;
        countpos++;
    }

    /*cout<<"*************Show Result Second****************"<<endl;
    cout<<InfoResult[temp].AttriName<<endl;
    for(int i=begin;i<=end;i++)
    {
        for(int j=0;j<=5;j++)
            cout<<DataTable[i][j]<<"\t";
        cout<<endl;
    }
    cout<<"


";*/
    return countpos;
}

void BuildTree(int begin,int end,Node *parent)
{
    ,CountN=;
    CountInfoNP(begin,end,CountP,CountN);

    cout<<"************************The data be sorted**************************"<<endl;
    for(int i=begin;i<=end;i++)
    {
        ;j<=;j++)
            cout<<DataTable[i][j]<<"\t";
        cout<<endl;
    }
    cout<<"


";

    cout<<parent->AttriName<<" have a look: "<<CountP<<endl;
    ||CountN==) //该子集当中只包含Yes或者No时为叶子节点，返回调用处；
    {
        cout<<"creat leaf node"<<endl;
        Node* t=new Node(); //建立叶子节点
        )
            t->AttriName="No";
        else
            t->AttriName="Yes";
        parent->Children.push_back(t); //插入孩子节点
        return;
    }
    else
    {
        double p=(double)CountP/(CountP+CountN);
        double InfoH=ComputInfo(p); //获得信息熵

        ;k<;k++) //循环计算各个属性的条件信息熵，并计算出互信息
        {
            int KindOfValue=SearchData(begin,end,k);
            +end-begin;
            ;j<KindOfValue;j++) //计算出属性的每种取值的权重的倒数
                DataValueWeight[j].ValueWeight=DataValueWeight[j].ValueWeight/sum;

            ;
            ].ValueNWeight!=&&DataValueWeight[].ValuePWeight!=)
                InfoGain=DataValueWeight[].ValueWeight*(ComputLog(DataValueWeight[].ValueNWeight)/DataValueWeight[].ValueNWeight+ComputLog(DataValueWeight[].ValuePWeight)/DataValueWeight[].ValuePWeight);

            ;j<KindOfValue;j++) //计算条件信息
            &&DataValueWeight[j].ValuePWeight!=)
                InfoGain+=DataValueWeight[j].ValueWeight*(ComputLog(DataValueWeight[j].ValueNWeight)/DataValueWeight[j].ValueNWeight+ComputLog(DataValueWeight[j].ValuePWeight)/DataValueWeight[j].ValuePWeight);

            InfoResult[k].AttriI=InfoH-InfoGain; //计算互信息
            InfoResult[k].AttriKind=KindOfValue;
        }
        int temp=PickAttri(); //选出互信息最大的属性作为节点建树
        Node* t=new Node();
        t->AttriName=InfoResult[temp].AttriName;
        SearchData(begin,end,temp);
        ;k<InfoResult[temp].AttriKind;k++)
        {
            string name=DataValueWeight[k].AttriValueName;
            t->AttriValue.push_back(name);
        }
        t->parent=parent;
        parent->Children.push_back(t); //孩子节点压入vector当中
        int position[NUMOFPOS];

        cout<<"before SortByAttriValue Begin: "<<begin<<",END: "<<end<<endl;

        SortByAttriValue(begin,end,temp,position); //将数据按照选定属性的取值不同进行划分
        int times=InfoResult[temp].AttriKind;
        ;l<=times;l++)
            cout<<position[l]<<" ";
        cout<<endl;
        ;k<times;k++)
            {
                int head,rear;
                head=position[k];
                ;
                rear=position[hire]-;
                ;l<=times;l++)
                cout<<position[l]<<" ";
                cout<<endl;
                cout<<"Head: "<<head<<" ,Rear: "<<rear<<endl;
                BuildTree(head,rear,t);
            }
    }
}

void ShowTree(Node *root)
{

    if(root->AttriName=="Yes"||root->AttriName=="No")
    {
        cout<<root->AttriName<<endl;
        return;
    }
    else
    {
        cout<<root->AttriName<<endl;
        for(vector<string>::iterator itvalue=root->AttriValue.begin();itvalue!=root->AttriValue.end();itvalue++)
        {
            string value=*itvalue;
            cout<<value<<" ";
        }
        cout<<endl;
        for(vector<Node*>::iterator itnode=root->Children.begin();itnode!=root->Children.end();itnode++)
        {
            Node *t=*itnode;
            ShowTree(t);
        }
    }
}
int main()
{
    InfoResult[].AttriName="天气";
    InfoResult[].AttriName="气温";
    InfoResult[].AttriName="湿度";
    InfoResult[].AttriName="风";
    ReadData();
    Node *Root=new Node;
    BuildTree(,NUM-,Root);

    //vector<Node>::iterator it=Root.Children.begin();
    ShowTree(Root);
    /*Node t=*it;
    cout<<t.AttriName<<endl;
    for(vector<string>::iterator itvalue=t.AttriValue.begin();itvalue!=t.AttriValue.end();itvalue++)
    cout<<*itvalue<<endl;
    it=t.Children.begin();

    t=*it;
    cout<<t.AttriName<<endl;*/
    //ShowTree(t);
    //cout<<"Root: "<<t.AttriName<<" ,Value: "<<*(t.AttriValue.begin())<<endl;
    ;
}
```
